import json
import logging
import math
import os
import subprocess
from typing import TextIO

import h5py
import librosa
import plotly.graph_objs as go
import soundfile
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Annotation
from pyannote.database.util import load_rttm
from pydub import AudioSegment

from src.audio_sample import AudioSample
from src.download_srf_audio import PODCAST_METADATA_FOLDER
from src.download_yt_audio import PODCAST_YT_METADATA_FOLDER
from src.util import DATASETS_PATH, SAMPLING_RATE, DATASETS_PATH_H5, XTTS_SAMPLE_RATE

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")

# SEGMENTED_AUDIO_PATH = DATASETS_PATH
SEGMENTED_AUDIO_PATH = "src/segmented_speech"
# SEGMENTED_AUDIO_PATH = "/scratch/stucksam"
SEGMENTED_AUDIO_JSON = "src/segmented_speech/segmented_speech.json"
MIN_SAMPLE_DURATION = 2.0
MAX_SAMPLE_DURATION = 15.0
N_WINDOW_SAMPLES = 400
N_STRIDING_SAMPLES = 320

logger = logging.getLogger(__name__)


def _get_parent_podcast_folder(pod_type: str = "SRF") -> str:
    # return "/cluster/home/stucksam/datasets"
    return f"src/{PODCAST_METADATA_FOLDER}" if pod_type == "SRF" else f"src/{PODCAST_YT_METADATA_FOLDER}"


def _get_podcast_metadata(podcast: str, pod_type: str = "SRF") -> dict:
    # run the pipeline on an audio file
    with open(f"{_get_parent_podcast_folder(pod_type)}/{podcast}.json", "r") as f:
        episodes = json.loads(f.read())
    return episodes


def diarize_podcast(podcast, pod_type: str = "SRF"):
    """
    Important considerations:
    1. stereo or multi-channel audio files are automatically downmixed to mono by averaging the channels.
    2. audio files sampled at a different rate are resampled to 16kHz automatically upon loading.

    Taken from: https://huggingface.co/pyannote/speaker-diarization-3.0
    :return:
    """
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token=HF_ACCESS_TOKEN)
    pipeline.to(torch.device("cuda"))

    logger.info(f"Doing podcast: {podcast}")

    episodes = _get_podcast_metadata(podcast, pod_type=pod_type)
    podcast_path = f"{_get_parent_podcast_folder(pod_type)}/{podcast}"

    for _uuid, episode in episodes.items():
        rttm_file = f"{podcast_path}/{_uuid}.rttm"
        if os.path.exists(rttm_file):
            logger.debug(f"skipping {_uuid} as it was already diarized")
            continue

        if not os.path.exists(f"{podcast_path}/{_uuid}.mp3"):
            logger.warning(f"skipping {_uuid} as it doesn't exist")
            continue

        logger.info(f"Diarization for episode: {_uuid}")
        try:
            with ProgressHook() as hook:
                diarization = pipeline(f"{podcast_path}/{_uuid}.mp3", hook=hook, min_speakers=2)

        except soundfile.LibsndfileError:
            logger.debug("Converting file to wav due to header issues")
            _ = subprocess.run(["ffmpeg", "-i", f"{podcast_path}/{_uuid}.mp3", f"{podcast_path}/{_uuid}.wav"],
                               capture_output=True, check=True)
            try:
                with ProgressHook() as hook:
                    diarization = pipeline(f"{podcast_path}/{_uuid}.wav", hook=hook, min_speakers=2)
            except soundfile.LibsndfileError as e:
                logger.warning(f"!!!! SOMETHING IS SERIOUSLY WRONG with this file: {str(e)}")
                continue

        # dump the diarization output to disk using RTTM format
        with open(rttm_file, "w") as rttm:
            diarization.write_rttm(rttm)


def combine_neighbouring_tracks(all_tracks, annotation: Annotation) -> tuple[list, dict]:
    merged = []
    merged_dict = {label: [] for label in annotation.labels()}
    to_skip = set()
    for i, track in enumerate(all_tracks):  # combine tracks with the same speaker
        track_copy = track.copy()
        if i in to_skip:
            continue

        for j, track2 in enumerate(all_tracks):  # combine all following tracks that belong to the same speaker.
            if j <= i:  # do not merge track with subsequent ones if previous track ends after current track (complete overlap case)
                # detect prev segment that completely overlaps current track + 0.1s
                if track2["end"] - track["end"] > 0.1 and track2["speaker"] != track["speaker"]:
                    break
                else:
                    continue

            else:  # check i + k until not same speaker
                if track["speaker"] == track2["speaker"] and track2["start"] - track["end"] <= MIN_SAMPLE_DURATION: # Silences of more than 2 seconds are not tolerated
                    track_copy["end"] = track2["end"]
                    to_skip.add(track2["track"])
                else:
                    break
        info = {
            "speaker": track_copy["speaker"],
            "start": track_copy["start"],
            "end": track_copy["end"],
            "dur": track_copy["end"] - track_copy["start"],
            "track": track_copy["track"],
        }
        merged.append(info)
        merged_dict[track["speaker"]].append(info)

    return merged, merged_dict


def __is_track_already_in_collection(col, track_id):
    return any(track["track"] == track_id for track in col)


def skip_overlapping_audio(tracks: list) -> list:
    non_overlapping_tracks = []
    reassign_check_index = -1
    skip_tracks = set()
    for i, track in enumerate(tracks):
        if track["track"] in skip_tracks:
            continue
        for j, compare_track in enumerate(tracks[i + 1:]):  # start one track after the currently checked track (i + 1)
            if compare_track["start"] >= track["end"] or track["end"] - compare_track["start"] < 0.1:
                logger.debug("Accurate cut detected, continuing...")
                if not __is_track_already_in_collection(non_overlapping_tracks, track["track"]):
                    non_overlapping_tracks.append(track)
                break  # stop iterating

            elif compare_track["start"] < track["end"]:
                logger.debug("Overlap detected")

                # Case track2 speaks over track but stops while track continues
                if compare_track["end"] <= track["end"]:
                    skip_tracks.add(compare_track["track"])

                    track_copy = track.copy()
                    track["end"] = compare_track["start"]
                    track["dur"] = track["end"] - track["start"]
                    non_overlapping_tracks.append(track)
                    # rest segment is larger Min Sample duration -> cut out sample
                    if track_copy["end"] - compare_track["end"] > MIN_SAMPLE_DURATION:
                        # cut track2 end -> track1 end into new segment, overlapping segment is skipped
                        track_copy["start"] = compare_track["end"]
                        # must not be skipped by next iteration, as such tag with negative number must be assigned
                        track_copy["track"] = reassign_check_index - i
                        track_copy["dur"] = track_copy["end"] - track_copy["start"]
                        tracks[i + 1 + j] = track_copy
                    else:
                        logger.debug("End of segment from track is smaller than 2 seconds, "
                                     "deleting track2 and last segment")

                else:  # case that track2 starts in previous and continues after, compare_track["end"] > track["end"]
                    compare_track_start = compare_track["start"]
                    # remaining speech after subtracting overlapping part should be > Min Sample Duration
                    if compare_track["end"] - track["end"] > MIN_SAMPLE_DURATION:
                        compare_track["start"] = track["end"]
                        compare_track["dur"] = compare_track["end"] - compare_track["start"]

                    else:  # compare track can be ignored because most of the speech is overlapping
                        skip_tracks.add(compare_track["track"])

                    track["end"] = compare_track_start
                    track["dur"] = track["end"] - track["start"]
                    if track["dur"] < MIN_SAMPLE_DURATION:
                        skip_tracks.add(track["track"])
                    else:
                        non_overlapping_tracks.append(track)
            else:
                break

    return non_overlapping_tracks


def cut_into_two_to_fifteen_sec_segments(tracks: list) -> list:
    ready_tracks = []
    for track in tracks:
        duration = track["dur"]

        # Skip tracks that are too short
        if duration < MIN_SAMPLE_DURATION:
            logger.debug(f"Skipping track {track['track']} with duration {round(duration, 4)}s")
            continue

        # If the track is already within the desired range, add it directly
        if duration <= MAX_SAMPLE_DURATION:
            ready_tracks.append(track)
            continue

        # Split the track into segments
        ready_tracks = split_track_into_segments(track, ready_tracks)

    return ready_tracks


def split_track_into_segments(track: dict, ready_tracks: list) -> list:
    """
    Splits a track into segments of maximum `MAX_SAMPLE_DURATION` seconds.
    Appends the resulting segments to `ready_tracks`.
    """
    start = track["start"]
    end = track["end"]
    duration = track["dur"]

    num_segments = math.ceil(duration / MAX_SAMPLE_DURATION)

    for i in range(num_segments):
        # Create a copy of the track to modify
        segment = track.copy()

        # Calculate segment start and end times
        segment_start = start + i * MAX_SAMPLE_DURATION
        segment_end = min(segment_start + MAX_SAMPLE_DURATION, end)

        # Ensure the last segment is at least `MIN_SAMPLE_DURATION`
        if end - segment_start < MIN_SAMPLE_DURATION:
            # Adjust the previous segment if it exists
            if ready_tracks:
                prev_segment = ready_tracks[-1]
                # change end to accurately match MIN_SAMPLE_DURATION of current segment -> e.g. 694 to 692 so that segment can be 2 seconds long
                prev_segment["end"] -= (MIN_SAMPLE_DURATION - (end - segment_start))
                prev_segment["dur"] = prev_segment["end"] - prev_segment["start"]

            # Adjust the current segment start time
            segment_start = end - MIN_SAMPLE_DURATION

        # Update segment properties
        segment["start"] = segment_start
        segment["end"] = segment_end
        segment["dur"] = segment_end - segment_start

        # Append the segment to the list
        ready_tracks.append(segment)

    return ready_tracks


def _load_json_meta(podcast: str) -> dict:
    with open(SEGMENTED_AUDIO_JSON, "r") as f:
        segmented_samples = json.loads(f.read())

    cut_audio_segments = {podcast: {}}  # TODO: This looks horrendous, fix in future
    if podcast in segmented_samples:
        for _id, episode in segmented_samples[podcast].items():
            cut_audio_segments[podcast][_id] = {}
            for _cut_id, cut in episode.items():
                cut_audio_segments[podcast][_id][_cut_id] = AudioSample.create_audio_sample_from_dict(cut)
    return cut_audio_segments


def _load_txt_meta(podcast: str) -> [TextIO, set]:
    """
    Built as sample_name -> track_id -> duration -> track_start -> track_end -> speaker -> no_outputs
    :param podcast:
    :return:
    """
    metadata_txt = os.path.join(SEGMENTED_AUDIO_PATH, f'{podcast}.txt')
    already_processed = set()
    if os.path.exists(f"{SEGMENTED_AUDIO_PATH}/{podcast}.txt"):
        with open(metadata_txt, 'rt', encoding='utf-8') as meta_data_file:
            already_processed = set([x.split('\t')[0] for x in meta_data_file.readlines()])
        return open(metadata_txt, 'at', encoding='utf-8'), already_processed

    else:
        return open(metadata_txt, 'wt', encoding='utf-8'), already_processed


def _load_h5(podcast: str) -> h5py.File:
    h5_file = os.path.join(DATASETS_PATH, f"{podcast}.hdf5")
    return h5py.File(h5_file, "a" if os.path.exists(h5_file) else "w")


def _load_sample(path: str):
    try:
        return AudioSegment.from_file(path + ".mp3")
    except soundfile.LibsndfileError:
        return AudioSegment.from_file(path + ".wav")


def _clean_track_segments(annotation: Annotation) -> list:
    _, tracks = as_dict_list(annotation)
    merged_tracks, _ = combine_neighbouring_tracks(tracks, annotation)
    non_overlap_tracks = skip_overlapping_audio(merged_tracks)
    return cut_into_two_to_fifteen_sec_segments(non_overlap_tracks)


def cut_segments_on_rttm(podcast: str, pod_type: str = "SRF", save_cuts_as_mp3: bool = False,
                         write_to_hdf5: bool = True, sample_rate: int = SAMPLING_RATE):
    logger.info(f"Cutting podcast {podcast}")
    metadata_json = _get_podcast_metadata(podcast, pod_type)
    metadata_txt, already_processed = _load_txt_meta(podcast)
    h5_file = _load_h5(podcast)
    parent_path = _get_parent_podcast_folder(pod_type)
    cut_audio_segments = _load_json_meta(podcast)

    try:
        for _uuid, episode in metadata_json.items():
            logger.info(f"Cutting episode {_uuid}")
            episode_path = f"{parent_path}/{podcast}/{_uuid}"
            if not os.path.exists(episode_path + ".rttm"):
                logger.info(f"rttm does not exist for {_uuid}, skipping")
                continue

            rttm = load_rttm(episode_path + ".rttm")
            sound = _load_sample(episode_path)

            cut_audio_segments[podcast][_uuid] = {}

            if not os.path.exists(f"{SEGMENTED_AUDIO_PATH}/{podcast}"):
                os.makedirs(f"{SEGMENTED_AUDIO_PATH}/{podcast}")

            episode_cut_sample_path = f"{SEGMENTED_AUDIO_PATH}/{podcast}/{_uuid}"

            if not os.path.exists(episode_cut_sample_path):
                os.makedirs(episode_cut_sample_path)

            for uri, annotation in rttm.items():
                ready_tracks = _clean_track_segments(annotation=annotation)
                start_id = 1000  # enables better id assignment with 1 being lower than 10 in files due to 1001 and 1010
                for i, track in enumerate(ready_tracks):
                    track["track"] = start_id + i
                    segment_name = f"{_uuid}_{start_id + i}"
                    file_name = f"{episode_cut_sample_path}/{segment_name}.wav"

                    if os.path.exists(f"{episode_cut_sample_path}/{segment_name}.wav"):
                        continue

                    start_ms = track["start"] * 1000
                    end_ms = track["end"] * 1000
                    segment = sound[start_ms: end_ms]
                    segment.export(file_name, format="wav")  # write segment to os
                    if save_cuts_as_mp3:
                        segment.export(file_name.replace("wav", "mp3"), format="mp3")  # write segment to os

                    speech, _ = librosa.load(file_name, sr=sample_rate)
                    n_outputs = int(speech.shape[0] / N_STRIDING_SAMPLES)
                    duration = round(track['dur'], 4)
                    speaker = track['speaker']
                    track_start = round(track['start'], 4)
                    track_end = round(track['end'], 4)

                    h5_entry = h5_file.create_dataset(segment_name, dtype=float, data=speech)
                    if write_to_hdf5:
                        h5_entry.attrs["dataset_name"] = podcast
                        h5_entry.attrs["speaker"] = speaker
                        h5_entry.attrs["duration"] = duration
                        h5_entry.attrs["track_start"] = track_start
                        h5_entry.attrs["track_end"] = track_end
                    h5_file.flush()

                    cut_audio_segments[podcast][_uuid][segment_name] = AudioSample(segment=track)
                    metadata_txt.write(
                        f"{segment_name}\t{track['track']}\t{duration}\t{track_start}\t{track_end}\t{speaker}\t{n_outputs}\n")

                    os.remove(file_name)  # keep disk space clean

    except Exception as e:
        logger.error(f"ERROR {type(e).__name__}: {str(e)}")

    metadata_txt.close()
    h5_file.close()

    for episode, samples in cut_audio_segments[podcast].items():
        for _id, info in samples.items():
            cut_audio_segments[podcast][episode][_id] = info.to_dict()

    with open(SEGMENTED_AUDIO_JSON, "r") as f:
        segmented_samples = json.loads(f.read())

    segmented_samples[podcast] = cut_audio_segments[podcast]

    with open(SEGMENTED_AUDIO_JSON, "w") as f:
        json.dump(segmented_samples, f)


def as_dict_list(annotation: Annotation):
    result = {label: [] for label in annotation.labels()}
    parsed_tracks = []
    for segment, track, label in annotation.itertracks(yield_label=True):
        info = {
            "speaker": label,
            "start": segment.start,
            "end": segment.end,
            "dur": segment.duration,
            "track": track,
        }
        result[label].append(info)
        parsed_tracks.append(info)
    return result, parsed_tracks



def resample_to_xtts_sr(pod_type: str = "SRF"):
    podcasts = [f.replace(".txt", "") for f in os.listdir(DATASETS_PATH) if f.endswith('.txt') and "_enriched" not in f]
    type_folder = _get_parent_podcast_folder(pod_type)
    for p in podcasts:
        if os.path.isdir(f"{DATASETS_PATH}/{type_folder}/{p}"):
            cut_segments_on_rttm(p, pod_type, sample_rate=XTTS_SAMPLE_RATE)


def plot_annotation(annotation: Annotation):
    data, _ = as_dict_list(annotation)
    # merged_tracks, data = combine_neighbouring_tracks(data2, annotation)
    fig = go.Figure(
        layout={
            'barmode': 'stack',
            'xaxis': {'automargin': True},
            'yaxis': {'automargin': True}
        }
    )
    for label, turns in data.items():
        durations, starts, ends = [], [], []
        for turn in turns:
            durations.append(turn["dur"])
            starts.append(turn["start"])
            ends.append(f"{turn['end']:.1f}")
        fig.add_bar(
            x=durations,
            y=[label] * len(durations),
            base=starts,
            orientation='h',
            showlegend=True,
            name=label,
            hovertemplate="<b>%{base:.2f} --> %{x:.2f}</b>",
        )

    fig.update_layout(
        title=annotation.uri,
        legend_title="Speakers",
        font={"size": 18},
        height=500,
        yaxis=go.layout.YAxis(showticklabels=False, showgrid=False),
        xaxis=go.layout.XAxis(title="Time (seconds)", showgrid=False),
    )
    fig.update_xaxes(rangemode="tozero")
    fig.update_traces(width=0.4)

    fig.show()


if __name__ == "__main__":
    """
    Thoughts: Only need to check for any overlaps and sort them to own list. RTTM is sequential so just check each track
    with all tracks following.

    Overlapping tracks need to be filtered by putting all overlapping tracks (e.g. 3 tracks when all three people
    speak at the same time) into its own list and discard them.

    Some RTTM files diarized the speech into multiple segments even though it was always the same person speaking -> need ot be combined when cutting
    Isolate other tracks and cut to 15s segments

    """
    rttm = load_rttm("analytics/H_384359e2-ace7-4a92-8c6d-ba07ccf41316.rttm")
    for uri, ano in rttm.items():
        # result = _clean_track_segments(annotation=ano)
        plot_annotation(annotation=ano)

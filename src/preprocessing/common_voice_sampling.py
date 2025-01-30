import json
import logging
import os

import h5py
import librosa
import pandas as pd
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline

from src.analytics.analyze_snf import get_distribution_of_property
from src.segment_speech import _load_h5
from src.transcribe_speech import MODEL_AUDIO_PHONEME
from src.util import SAMPLING_RATE, get_h5_file, setup_gpu_device

BATCH_SIZE = 16

datapath = "src/config"
cv_path = "/home/ubuntu/ma/commonvoice/cv-corpus-19.0-2024-09-13/de/clips"

logger = logging.getLogger(__name__)


def clean_common_voice():
    df = pd.read_csv(f"{datapath}/cv_train.tsv", sep="\t")
    is_null = df[df['gender'].isnull()].reset_index()
    for index, row in is_null.iterrows():
        clip_name = row['path']
        clip_path = f"{cv_path}/{clip_name}"

        if os.path.exists(clip_path):
            logger.info(f"Deleting {clip_name}...")
            os.remove(clip_path)


def get_speakers_by_dataset(dataset) -> set:
    with open(f"{datapath}/{dataset}.jsonl", "r") as f:
        meta_data = [json.loads(line) for line in f]
    return {line["speaker_id"] for line in meta_data}


def create_dataset_of_30_hours(file_name: str, aim_time: float, aim_men_perc: float, aim_fem_perc: float):
    dataset_name = file_name.replace(".tsv", "")
    file_path = os.path.join(datapath, "cv_train.tsv")

    df = pd.read_csv(file_path, sep="\t")
    df_non_null = df[df["gender"].notnull()].reset_index()
    df_de = df_non_null[df_non_null["accents"] == "Deutschland Deutsch"]

    df_dur = pd.read_csv(os.path.join(datapath, "clip_durations.tsv"), sep="\t")
    df_merged = pd.merge(df_de, df_dur, how="inner", left_on="path", right_on="clip")
    if "index" in df_merged.columns.tolist():
        df_merged = df_merged.drop(columns='index')
    df_merged = df_merged.sample(frac=1.0).reset_index(drop=True)

    df_speakers = df_merged.groupby(["client_id", "gender", "age", "accents"], as_index=False)["duration[ms]"].sum()

    current_time = 0.0
    fem_aim_time = aim_time * aim_fem_perc
    men_aim_time = aim_time * aim_men_perc
    fem_cur_time = 0.0
    men_cur_time = 0.0
    max_time_per_speaker = aim_time * 0.05  # at most 5% should be one speaker

    if "train" in dataset_name:
        used_speakers = set()
    elif "valid" in dataset_name:
        used_speakers = get_speakers_by_dataset("cv_train")
    else:
        used_speakers = get_speakers_by_dataset("cv_train") | get_speakers_by_dataset("cv_valid")

    df_set = pd.DataFrame(columns=df_merged.columns.tolist())
    for index, row in df_speakers.sample(frac=1).iterrows():  # random sampling

        speaker_duration = row["duration[ms]"] / 1000
        if row["client_id"] in used_speakers or speaker_duration > max_time_per_speaker or current_time + speaker_duration > aim_time:
            continue
        if row["gender"] == "male_masculine" and speaker_duration + men_cur_time <= men_aim_time * 1.01:
            men_cur_time += speaker_duration
        elif row["gender"] == "female_feminine" and speaker_duration + fem_cur_time <= fem_aim_time * 1.01:
            fem_cur_time += speaker_duration
        else:
            continue

        df_single = df_merged[df_merged["client_id"] == row["client_id"]]
        df_set = pd.concat([df_set, df_single], ignore_index=True)
        current_time = fem_cur_time + men_cur_time
        if current_time + 60 >= aim_time:
            break

    if "index" in df_set.columns.tolist():
        df_set = df_set.drop(columns='index')

    counts_gender = get_distribution_of_property(df_set, "gender")
    counts_age = get_distribution_of_property(df_set, "age")

    logger.info("Gender Split")
    logger.info(counts_gender)
    logger.info("Age Split")
    logger.info(counts_age)

    h5_file = _load_h5(dataset_name)
    json_data = []

    for index, row in df_set.iterrows():
        clip_path = os.path.join(cv_path, row['path'])
        clip_path_wav = clip_path.replace(".mp3", ".wav")

        entry = {"corpus_name": "CommonVoice",
                 "dataset_name": dataset_name,
                 "sample_name": row["path"].replace(".mp3", ""),
                 "classname": "de",
                 "class_nr": 7,
                 "output_len": -1,
                 "speaker_id": row["client_id"],
                 "text": row["sentence"],
                 "phonemes": "",
                 "gender": row["gender"],
                 "age": row["age"],
                 "accents": row["accents"]
                 }

        sound = AudioSegment.from_mp3(clip_path)
        sound.export(clip_path_wav, format="wav")  # write segment to os
        speech, _ = librosa.load(clip_path_wav, sr=SAMPLING_RATE)

        _ = h5_file.create_dataset(entry["sample_name"], dtype=float, data=speech)
        h5_file.flush()
        json_data.append(entry)
        os.remove(clip_path_wav)  # keep disk space clean

    # Open the file in write mode
    with open(os.path.join(datapath, f"{dataset_name}.jsonl"), "w", encoding="utf-8") as f:
        for entry in json_data:
            # Convert each dictionary to a JSON string and write it as a line
            json_line = json.dumps(entry)
            f.write(json_line + "\n")  # Add newline for the next JSON object


def cv_audio_to_phoneme(dataset: str, write_to_hdf5: bool = True) -> None:
    with open(os.path.join(datapath, f"{dataset}.jsonl"), "r", encoding="utf-8") as f:
        meta_data = [json.loads(line) for line in f]

    num_samples = len(meta_data)
    h5_file = get_h5_file(dataset)
    device, torch_dtype = setup_gpu_device()

    processor = Wav2Vec2Processor.from_pretrained(MODEL_AUDIO_PHONEME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_AUDIO_PHONEME)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    try:
        with h5py.File(h5_file, "r+") as h5:
            for start_idx in range(0, num_samples, BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, num_samples)
                # Load batch of audio data
                audio_batch = [h5[meta_data[i]["sample_name"]][:] for i in range(start_idx, end_idx)]

                results = pipe(audio_batch, batch_size=BATCH_SIZE)
                # Save results
                for idx, result in enumerate(results):
                    phoneme = result["text"].strip()
                    meta_data[start_idx + idx]["phonemes"] = phoneme
                    if write_to_hdf5:
                        h5[meta_data[start_idx + idx]["sample_name"]].attrs["phoneme"] = phoneme
                    logger.info(f"NAME: {meta_data[start_idx + idx]['sample_name']}, PHON: {phoneme}")

    except Exception as e:
        logger.error(f"ERROR: {type(e).__name__} with error {str(e)}")

    with open(os.path.join(datapath, f"{dataset}_enriched.jsonl"), "wt", encoding="utf-8") as f:
        for entry in meta_data:
            f.write(json.dumps(entry) + "\n")


if __name__ == '__main__':
    perc_fem = 53.3382 / 100
    perc_men = 46.6618 / 100
    aim_hours = 30.0 * 3600  # seconds

    perc_fem_test = 57.7566 / 100
    perc_men_test = 42.2434 / 100
    aim_hours_test = 4.0 * 3600

    perc_fem_valid = 52.3797 / 100
    perc_men_valid = 47.6203 / 100
    aim_hours_valid = 4.0 * 3600

    create_dataset_of_30_hours("cv_train.tsv", aim_hours, perc_men, perc_fem)
    create_dataset_of_30_hours("cv_valid.tsv", aim_hours_valid, perc_men_valid, perc_fem_valid)
    create_dataset_of_30_hours("cv_test.tsv", aim_hours_test, perc_men_test, perc_fem_test)

    cv_audio_to_phoneme("cv_train")
    cv_audio_to_phoneme("cv_valid")
    cv_audio_to_phoneme("cv_test")

import json
import logging
import os

import h5py
import torch

from src.data_points import DatasetDataPoint, DialectDataPoint

DATASETS_PATH_H5 = "/scratch/stucksam"
# DATASETS_PATH = "/cluster/home/stucksam/datasets"
DATASETS_PATH = "src/segmented_speech"
DIALECT_DATA_PATH = DATASETS_PATH + "/dialects"
ANALYTICS_PATH = "src/analytics"

SAMPLING_RATE = 16000
XTTS_SAMPLE_RATE = 22050
DIALECT_TO_TAG = {
    "Zürich": "ch_zh",
    "Innerschweiz": "ch_lu",
    "Wallis": "ch_vs",
    "Graubünden": "ch_gr",
    "Ostschweiz": "ch_sg",
    "Basel": "ch_bs",
    "Bern": "ch_be",
    "Deutschland": "de"
}
LANG_MAP_INV = {v: k for k, v in DIALECT_TO_TAG.items()}

logger = logging.getLogger(__name__)


def _get_metadata_file_name(podcast: str, enriched: bool = False) -> str:
    return f"{podcast}_enriched.txt" if enriched else f"{podcast}.txt"


def get_h5_file(podcast: str):
    return os.path.join(DATASETS_PATH, f"{podcast}.hdf5")


def get_metadata_path(podcast: str, enriched: bool = False):
    """
    Loads text file to Python object for metadata handling.
    :param podcast: Name of podcast or dialect that needs to be loaded
    :param enriched: Use metadata of hdf5 creation or the enriched one after adding properties such as DE-Text or DID
    :return: 
    """
    name_metadata_file = _get_metadata_file_name(podcast, enriched)
    return os.path.join(DATASETS_PATH, name_metadata_file)


def load_meta_data(meta_data_path: str) -> tuple[list[DatasetDataPoint | DialectDataPoint], int]:
    """
    Loads podcast metadata generated on hdf5 file creation or enriched during subsequent processes. It's expected
    that metadata files do no contain mixed Dialect separated (read: reduced) metadata and Dataset separated
    (read: detailed) metadata.
    :param meta_data_path: Path to metadata file which should be parsed
    :return:
    """
    sample_list = []
    with open(meta_data_path, "rt", encoding="utf-8") as meta_file:
        for line in meta_file:
            split_line = line.replace('\n', '').split('\t')
            if DIALECT_DATA_PATH in meta_data_path:
                sample_list.append(DialectDataPoint.load_single_datapoint(split_line))
            else:
                sample_list.append(DatasetDataPoint.load_single_datapoint(split_line))

    length_samples = len(sample_list)
    # random.shuffle(sample_list)
    logger.info(f"NO. OF SAMPLES: {length_samples}")
    return sample_list, length_samples


def write_meta_data(podcast: str, meta_data, enriched: bool = False):
    file_name = _get_metadata_file_name(podcast, enriched)
    with open(os.path.join(DATASETS_PATH, file_name), "wt", encoding="utf-8") as f:
        for line in meta_data:
            f.write(line.to_string())


def setup_gpu_device() -> tuple:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger.info(f"Training / Inference device is: {device}")
    return device, torch_dtype


def _calc_cut_segments(podcast: str) -> float:
    sample_list, num_samples = load_meta_data(get_metadata_path(podcast, enriched=False))
    total_duration = 0.0

    for cut in sample_list:
        total_duration += cut.duration
    logger.info("Podcast: " + podcast)
    logger.info(
        f"{round(total_duration, 4)} in seconds, {round(total_duration / 60, 4)} in minutes, {round(total_duration / 3600, 4)} in hours")
    logger.info(f"{num_samples} cuts, {round(total_duration / num_samples, 4)} on average per cut\n")

    return total_duration


def get_length_of_data() -> None:
    # List all .txt files in the directory and parse their names
    txt_files = [f for f in os.listdir(DATASETS_PATH) if f.endswith('.txt')]

    # Print the parsed filenames (without extension)
    total_duration = 0.0
    for file in txt_files:
        filename = os.path.splitext(file)[0]  # Removes the .txt extension
        total_duration += _calc_cut_segments(filename)
    logger.info(
        f"{round(total_duration, 4)} in seconds, {round(total_duration / 60, 4)} in minutes, {round(total_duration / 3600, 4)} in hours.")


def calc_duration_of_podcast() -> None:
    """
    Based on json file instead of .txt metadata. Old code can be deleted if not needed
    :return:
    """
    with open("src/segmented_speech/segmented_speech.json", "r") as f:
        podcasts = json.loads(f.read())
    total_duration = 0.0
    cuts = 0
    pod = podcasts["Zivadiliring"]
    for e, content in pod.items():
        for cut, cut_cont in content.items():
            total_duration += cut_cont['dur']
            cuts += 1
    samples = len(pod)

    logger.info(
        f"{total_duration} in seconds, {round(total_duration / 60, 4)} in minutes, {round(total_duration / 3600, 4)} in hours")
    logger.info(f"{samples} samples, {round(total_duration / samples, 4)} on average per sample")
    logger.info(f"{cuts} cuts, {round(total_duration / cuts, 4)} on average per cut")


def calc_orig_duration_of_podcast() -> None:
    yt = 0.0
    srf = 0.0

    srf_podcasts = [f for f in os.listdir("src/srf_audio_crawl") if f.endswith('.json')]
    yt_podcasts = [f for f in os.listdir("src/youtube_audio_crawl") if f.endswith('.json')]

    print("SRF:")
    for srf_podcast in srf_podcasts:
        pod_dur = 0.0
        with open(f"src/srf_audio_crawl/{srf_podcast}", "r") as f:
            episodes = json.loads(f.read())
        for _id, episode in episodes.items():
            if episode["download_available"]:
                pod_dur += episode["duration_ms"]
        pod_dur = pod_dur / 1000 / 3600
        srf += pod_dur
        print(f"{srf_podcast.replace('.json', '')}: {round(pod_dur, 4)}")

    print("\nYT:")
    for yt_podcast in yt_podcasts:
        pod_dur = 0.0
        with open(f"src/youtube_audio_crawl/{yt_podcast}", "r") as f:
            episodes = json.loads(f.read())
        for _id, episode in episodes.items():
            pod_dur += int(episode["duration_ms"])
        pod_dur = pod_dur / 1000 / 3600
        yt += pod_dur
        print(f"{yt_podcast.replace('.json', '')}: {round(pod_dur, 4)}")

    print(f"YT: {yt}")
    print(f"SRF: {srf}")

def check_hdf5_file():
    try:
        with h5py.File(f"{DATASETS_PATH}/Tagesgespräch.hdf5", "r") as h5:
            logger.info("was able to open")
            logger.info(h5["0d271d50-9dab-48bb-b56e-9a8da36244f1_1077"].attrs["duration"])

    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == '__main__':
    get_length_of_data()

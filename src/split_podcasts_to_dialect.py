import json
import logging
import os
from multiprocessing import Process

import h5py
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.data_points import DialectDataPoint
from src.transcribe_speech import MODEL_T5_TOKENIZER, MODEL_PATH_DE_CH, MISSING_TEXT
from src.util import load_meta_data, DIALECT_TO_TAG, DIALECT_DATA_PATH, get_metadata_path, get_h5_file, \
    setup_gpu_device

CLUSTER_DATA_PATH = "/cluster/data/deri"
# CLUSTER_DATA_PATH = "src/segmented_speech"
SNF_DATASET_PATH = f"{CLUSTER_DATA_PATH}/snf_tts"
SWISSDIAL_DATASET_PATH = f"{CLUSTER_DATA_PATH}/swissdial"

SWISSDIAL_CANTON_TO_DIALECT = {
    "gr": "Graubünden",
    "lu": "Innerschweiz",
    "zh": "Zürich",
    "vs": "Wallis",
    "sg": "Ostschweiz",
    "be": "Bern",
    "bs": "Basel",
    "ag": "Zürich",
}

logger = logging.getLogger(__name__)


def _get_dialect_meta_data_path(dialect: str) -> str:
    return os.path.join(DIALECT_DATA_PATH, f"{dialect}.txt")


def _write_dialect_meta_data(dialect: str, dialect_content: list[DialectDataPoint]) -> None:
    meta_data_dialect_path = _get_dialect_meta_data_path(dialect)
    with open(meta_data_dialect_path, "wt", encoding="utf-8") as f:
        f.writelines(line.to_string() for line in dialect_content)


def _get_dialect_files(dialect) -> tuple[list, str]:
    meta_data_dialect_path = _get_dialect_meta_data_path(dialect)
    h5_file_dialect = os.path.join(DIALECT_DATA_PATH, f"{dialect}.hdf5")

    meta_data_dialect, _ = load_meta_data(meta_data_dialect_path) if os.path.exists(
        meta_data_dialect_path) else ([], 0)
    return meta_data_dialect, h5_file_dialect


def start_podcast_move(podcast: str, dialect: str, samples: list) -> None:
    logger.info(f"Performing move for dialect '{dialect}'.")
    path_orig_h5 = get_h5_file(podcast)
    with h5py.File(path_orig_h5, "r") as h5_read:
        meta_data_dialect, h5_file_dialect = _get_dialect_files(dialect)

        with h5py.File(h5_file_dialect, "a") as h5_dialect:
            for entry in samples:
                if entry.sample_name in h5_dialect:
                    continue
                h5_content = h5_read[entry.sample_name]
                # specifically using [()] instead of [:] to reduce operation time as we are not slicing
                new_h5_entry = h5_dialect.create_dataset(entry.sample_name, dtype=float, data=h5_content[()])

                # Copy attributes such as DID, phoneme, mel spec etc.
                for attr_name, attr_value in h5_content.attrs.items():
                    new_h5_entry.attrs[attr_name] = attr_value
                h5_dialect.flush()
                entry.dataset_name = podcast
                meta_data_dialect.append(entry.convert_to_dialect_datapoint())

        _write_dialect_meta_data(dialect, meta_data_dialect)

    logger.info(f"Finished move for dialect '{dialect}'.")


def move_podcast_to_dialect(podcast: str) -> None:
    """
    Runs move of dialect data in parallel to reduce time
    :param podcast:
    :return:
    """
    logger.info(f"Starting concurrent move of podcast '{podcast}' to dialect hdf5.")

    # Load metadata and initialize dialects
    meta_data, _ = load_meta_data(get_metadata_path(podcast, enriched=True))
    dialects = {key: [] for key in DIALECT_TO_TAG.keys()}
    os.makedirs(DIALECT_DATA_PATH, exist_ok=True)

    # Group metadata by dialect
    for entry in meta_data:
        dialects[entry.did].append(entry)

    for dialect, entries in dialects.items():
        logger.info(f"{dialect}: {len(entries)}")

    processes = [
        Process(target=start_podcast_move, args=(podcast, dialect, samples))
        for dialect, samples in dialects.items() if samples  # if contains entries then make process
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()


def _load_snf_tsv_for_duration() -> dict:
    path = "src/config"
    train = f"{path}/train_all.tsv"
    test = f"{path}/test.tsv"
    valid = f"{path}/valid.tsv"
    sample_to_duration = {}
    for file in [train, test, valid]:
        df = pd.read_csv(file, sep="\t")
        for index, row in df.iterrows():
            # path thingy is something custom because I just copy ready made h5s, check your env and replace as needed
            sample_to_duration[row["path"].replace("/", "-").replace(".flac", "")] = round(float(row["duration"]), 4)
    return sample_to_duration


def create_datapoints_for_speaker(speaker: str, speaker_path: str, parse_duration: bool = False) -> list[DialectDataPoint]:
    data = []
    if parse_duration:
        sample_to_duration = _load_snf_tsv_for_duration()
    with open(f"{speaker_path}/metadata.txt", "rt", encoding="utf-8") as meta_file:
        for line in meta_file:
            split_line = line.split("|")
            sample_name = split_line[0]
            data.append(DialectDataPoint(
                dataset_name="SNF",
                sample_name=sample_name,
                duration=sample_to_duration[sample_name] if parse_duration else -1.0,
                speaker_id=speaker,
                de_text=split_line[1],
            ))
    return data


def load_speakers_to_dialect(dialect: str, speakers: list) -> None:
    logger.info(f"Starting move for dialect '{dialect}' for SNF.")
    meta_data_dialect, h5_file_dialect = _get_dialect_files(dialect)

    with h5py.File(h5_file_dialect, "a") as h5_dialect:
        for speaker in speakers:
            speaker_path = f"{SNF_DATASET_PATH}/speakers/{speaker}"
            meta_data_speaker = create_datapoints_for_speaker(speaker, speaker_path, True)
            with h5py.File(f"{speaker_path}/audio.h5", "r") as h5_read:
                for entry in meta_data_speaker:
                    # I want uniformity in hdf5 keys of type SAMPLE_CUTID with only one underscore or just SAMPLE
                    new_sample_name = entry.sample_name.split("-")[-1]
                    if new_sample_name in h5_dialect:
                        continue

                    h5_content = h5_read[entry.sample_name]
                    # specifically using [()] instead of [:] to reduce operation time as we are not slicing
                    new_h5_entry = h5_dialect.create_dataset(new_sample_name, dtype=float, data=h5_content[()])

                    # Create essential attributes
                    new_h5_entry.attrs["dataset_name"] = entry.dataset_name
                    new_h5_entry.attrs["speaker"] = entry.speaker_id
                    new_h5_entry.attrs["de_text"] = entry.de_text
                    new_h5_entry.attrs["did"] = dialect
                    h5_dialect.flush()

                    entry.sample_name = new_sample_name
                    meta_data_dialect.append(entry)

        _write_dialect_meta_data(dialect, meta_data_dialect)
        logger.info(f"Finished move for dialect '{dialect}'.")


def move_snf_to_dialect() -> None:
    with open(f"{SNF_DATASET_PATH}/speaker_to_dialect.json", "rt", encoding="utf-8") as f:
        speaker_to_dialect = json.loads(f.read())

    dialects = {key: [] for key in DIALECT_TO_TAG.keys()}
    os.makedirs(DIALECT_DATA_PATH, exist_ok=True)

    # Group metadata by dialect
    for speaker, dialect in speaker_to_dialect.items():
        dialects[dialect].append(speaker)

    processes = [
        Process(target=load_speakers_to_dialect, args=(dialect, samples))
        for dialect, samples in dialects.items() if samples  # if contains entries then make process
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()


def create_datapoint_for_canton(canton: str) -> list[DialectDataPoint]:
    data = []
    canton_path = f"{SWISSDIAL_DATASET_PATH}/{canton}"
    with open(f"{canton_path}/metadata.txt", "rt", encoding="utf-8") as meta_file:
        for line in meta_file:
            split_line = line.split("|")
            data.append(DialectDataPoint(
                dataset_name="SwissDial",
                sample_name=split_line[0],
                duration=-1.0,
                speaker_id=f"SPEAKER_ch_{canton}",
                de_text=split_line[1],
            ))
    return data


def move_canton_to_dialect(canton: str) -> None:
    dialect = SWISSDIAL_CANTON_TO_DIALECT[canton]
    logger.info(f"Starting move for dialect '{dialect}' for SwissDial.")

    meta_data = create_datapoint_for_canton(canton)
    canton_path = f"{SWISSDIAL_DATASET_PATH}/{canton}"

    with h5py.File(f"{canton_path}/audio.h5", "r") as h5_read:
        meta_data_dialect, h5_file_dialect = _get_dialect_files(dialect)

        with h5py.File(h5_file_dialect, "a") as h5_dialect:
            for entry in meta_data:

                # I want uniformity in hdf5 keys of type SAMPLE_CUTID with only one underscore
                new_sample_name = entry.sample_name.replace(f"ch_{canton}",
                                                            f"ch-{canton}")
                if new_sample_name in h5_dialect:
                    continue
                h5_content = h5_read[entry.sample_name]
                # specifically using [()] instead of [:] to reduce operation time as we are not slicing
                new_h5_entry = h5_dialect.create_dataset(new_sample_name, dtype=float, data=h5_content[()])

                # Create essential attributes
                new_h5_entry.attrs["dataset_name"] = entry.dataset_name
                new_h5_entry.attrs["speaker"] = entry.speaker_id
                new_h5_entry.attrs["de_text"] = entry.de_text
                new_h5_entry.attrs["did"] = dialect
                h5_dialect.flush()

                entry.sample_name = new_sample_name
                meta_data_dialect.append(entry)

        _write_dialect_meta_data(dialect, meta_data_dialect)

    logger.info(f"Finished move for dialect '{dialect}'.")


def move_swissdial_to_dialect() -> None:
    aargau = "ag"
    processes = [
        Process(target=move_canton_to_dialect, args=(canton,))
        for canton, dialect in SWISSDIAL_CANTON_TO_DIALECT.items() if canton != aargau
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    move_canton_to_dialect(aargau)  # wait for Zurich to be done so no issues with concurrency occur


def _collect_samples_per_dialect(meta_data_dialect) -> dict:
    podcasts = {}
    for entry in meta_data_dialect:
        if entry.dataset_name not in podcasts:
            podcasts[entry.dataset_name] = []
        podcasts[entry.dataset_name].append(entry)
    return podcasts


def create_ch_text_metadata():
    dialects = [key for key in DIALECT_TO_TAG.keys() if key != "Deutschland"]
    for dialect in dialects:
        meta_data_dialect, _ = _get_dialect_files(dialect)
        podcasts = _collect_samples_per_dialect(meta_data_dialect)

        ch_text_metadata = []

        for podcast, samples in podcasts.items():
            if podcast in ["SwissDial", "SNF"]:
                samples = execute_ch_transcription(samples, dialect)
                for sample in samples:
                    ch_text_metadata.append(sample)
            else:
                meta_data_podcast, _ = load_meta_data(get_metadata_path(podcast, enriched=True))
                podcast_samples = {}
                for podcast_sample in meta_data_podcast:
                    podcast_samples[podcast_sample.sample_name] = podcast_sample.ch_text

                for sample in samples:
                    sample.de_text = podcast_samples[sample.sample_name]
                    ch_text_metadata.append(sample)

        meta_data_dialect_path = os.path.join(DIALECT_DATA_PATH, f"ch_{dialect}.txt")
        with open(meta_data_dialect_path, "wt", encoding="utf-8") as f:
            f.writelines(line.to_string() for line in ch_text_metadata)


def execute_ch_transcription(samples: list[DialectDataPoint], dialect: str):
    num_samples = len(samples)
    device, _ = setup_gpu_device()
    batch_size = 6

    model = T5ForConditionalGeneration.from_pretrained(os.path.join(MODEL_PATH_DE_CH, "best-model"))
    tokenizer = T5Tokenizer.from_pretrained(MODEL_T5_TOKENIZER)
    tokenizer.add_tokens(["Ä", "Ö", "Ü"])

    model.to(device)
    model.eval()

    for iter_count, start_idx in enumerate(range(0, num_samples, batch_size)):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_texts = [f"[{DIALECT_TO_TAG[dialect]}]: {samples[i].de_text}" for i in range(start_idx, end_idx)]

        # Tokenize the batch of sentences
        inputs = tokenizer(batch_texts, return_tensors="pt", padding="longest", truncation=True, max_length=400)
        # Move input tensors to the device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        del inputs

        # Generate translations
        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids, max_length=300, attention_mask=attention_mask, num_beams=5,
                                        num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

        batch_translations = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        batch_translations = [x.replace("Ä ", "Ä").replace("Ü ", "Ü").replace("Ö ", "Ö").strip() for x in
                              batch_translations]
        del input_ids, attention_mask, output_ids
        torch.cuda.empty_cache()

        for idx, ch_text in enumerate(batch_translations):
            if ch_text == "":
                logger.error(
                    f"NO SWISS GERMAN TRANSCRIPT GENERATED FOR {samples[start_idx + idx].sample_name}")
                ch_text = MISSING_TEXT

            samples[start_idx + idx].de_text = ch_text
    return samples



if __name__ == '__main__':
    move_podcast_to_dialect("Zivadiliring")

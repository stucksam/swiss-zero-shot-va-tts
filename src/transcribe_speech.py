import logging
import os
from collections import Counter

import h5py
import torch
from joblib import load
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, T5ForConditionalGeneration, T5Tokenizer, \
    Wav2Vec2Processor, Wav2Vec2ForCTC, Pipeline, PreTrainedModel

from src.data_points import DatasetDataPoint
from src.util import load_meta_data, get_h5_file, setup_gpu_device, DIALECT_TO_TAG, \
    get_metadata_path, write_meta_data

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
MODEL_PATH = "src/model"
MODEL_PATH_DE_CH = f"{MODEL_PATH}/de_to_ch_large_2"
MODEL_PATH_DID = f"{MODEL_PATH}/text_clf_3_ch_de.joblib"
MODEL_AUDIO_PHONEME = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
MODEL_WHISPER_v3 = "openai/whisper-large-v3"
MODEL_T5_TOKENIZER = "google/t5-v1_1-large"

BATCH_SIZE = 32
META_WRITE_ITERATIONS = 25

MISSING_TEXT = "NO_TEXT"
MISSING_PHONEME = "NO_PHONEME"
NO_CH_TEXT = "NO_CH_TEXT"

logger = logging.getLogger(__name__)

phon_did_cls = {0: "Zürich", 1: "Innerschweiz", 2: "Wallis", 3: "Graubünden", 4: "Ostschweiz", 5: "Basel", 6: "Bern",
                7: "Deutschland"}


class MergingHelper:

    def __init__(self, sample):
        self.duration: float = sample.duration
        self.samples: list = [sample]


def append_missing_properties_to_h5(podcast: str, enriched=False) -> None:
    logger.info("Adding missing properties to hdf5 files.")
    meta_data, _ = load_meta_data(get_metadata_path(podcast, enriched=enriched))
    h5_file = get_h5_file(podcast)

    with h5py.File(h5_file, "r+") as h5:
        for entry in meta_data:
            changed_properties = False
            if "dataset_name" not in h5[entry.sample_name].attrs:
                h5[entry.sample_name].attrs["dataset_name"] = podcast
                changed_properties = True

            if "speaker" not in h5[entry.sample_name].attrs:
                h5[entry.sample_name].attrs["speaker"] = entry.speaker_id
                changed_properties = True

            if "duration" not in h5[entry.sample_name].attrs:
                h5[entry.sample_name].attrs["duration"] = entry.duration
                changed_properties = True

            if "track_start" not in h5[entry.sample_name].attrs:
                h5[entry.sample_name].attrs["track_start"] = entry.track_start
                changed_properties = True

            if "track_end" not in h5[entry.sample_name].attrs:
                h5[entry.sample_name].attrs["track_end"] = entry.track_end
                changed_properties = True

            if entry.de_text and entry.de_text != MISSING_TEXT and "de_text" not in h5[entry.sample_name].attrs:
                h5[entry.sample_name].attrs["de_text"] = entry.de_text
                changed_properties = True

            if entry.phoneme and entry.phoneme != MISSING_PHONEME and "phoneme" not in h5[entry.sample_name].attrs:
                h5[entry.sample_name].attrs["phoneme"] = entry.phoneme
                changed_properties = True

            if entry.did and "did" not in h5[entry.sample_name].attrs:
                h5[entry.sample_name].attrs["did"] = entry.did
                changed_properties = True

            if entry.ch_text and entry.ch_text != NO_CH_TEXT and "ch_text" not in h5[entry.sample_name].attrs:
                h5[entry.sample_name].attrs["ch_text"] = entry.ch_text
                changed_properties = True

            if changed_properties:
                h5.flush()

    logger.info("Finished adding properties.")


def _fix_any_long_german_segments(podcast: str, pipe: pipeline, write_to_hdf5: bool = True):
    """
    Sometimes whisper returns very random de-texts, containing only repetitions of the same world like "erst, erst, erst,
    erst, erst,..." etc. These segments are detected (generally len of > 390 characters) and re-run through whisper. Let
    them purposefully run through without batching
    :return:
    """
    meta_data, _ = load_meta_data(get_metadata_path(podcast, enriched=True))
    long_segments = [entry for entry in meta_data if len(entry.de_text) > 390 or entry.de_text in [MISSING_TEXT, "..."]]
    if not long_segments:
        return
    h5_file = get_h5_file(podcast)

    logger.info(f"\nTrying to fix any issues with whisper generated sentences for podcast {podcast}.\n")
    with h5py.File(h5_file, "r+" if write_to_hdf5 else "r") as h5:
        for segment in long_segments:
            result = pipe(h5[segment.sample_name][:])

            text = result["text"].strip()
            if len(text) > 390 or text == "...":
                logger.error(
                    f"AGAIN NO OR ERRONEOUS GERMAN TRANSCRIPT GENERATED FOR {segment.sample_name}, check if you want to remove this sample")
                text = MISSING_TEXT
            segment.de_text = text
            if write_to_hdf5:
                h5[segment.sample_name].attrs["de_text"] = text
                h5.flush()
            logger.info(f"NAME: {segment.sample_name}, TXT: {text}")

    write_meta_data(podcast, meta_data, True)


def _setup_german_transcription_model():
    device, torch_dtype = setup_gpu_device()
    # could do more finetuning under "or more control over the generation parameters, use the model + processor API directly: "
    # in https://github.com/huggingface/distil-whisper/blob/main/README.md
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_WHISPER_v3, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(MODEL_WHISPER_v3)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": "german", "no_repeat_ngram_size": 3}
    )


def _save_de_transcribe_results(results: list, samples_to_iterate: list[DatasetDataPoint], start_idx: int,
                                write_to_hdf5: bool, h5: h5py.File) -> list[DatasetDataPoint]:
    for idx, result in enumerate(results):
        text = result["text"].strip()
        if text == "":
            logger.error(
                f"NO GERMAN TRANSCRIPT GENERATED FOR {samples_to_iterate[start_idx + idx].sample_name}")
            text = MISSING_TEXT
        samples_to_iterate[start_idx + idx].de_text = text
        if write_to_hdf5:
            h5[samples_to_iterate[start_idx + idx].sample_name].attrs["de_text"] = text
        logger.info(f"NAME: {samples_to_iterate[start_idx + idx].sample_name}, TXT: {text}")

    if write_to_hdf5:
        h5.flush()

    return samples_to_iterate


def transcribe_audio_to_german(podcast: str, write_to_hdf5: bool = True,
                               overwrite_existing_samples: bool = True) -> None:
    # You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
    logger.info("Transcribing to WAV to German")
    h5_file = get_h5_file(podcast)

    if overwrite_existing_samples:
        meta_data, num_samples = load_meta_data(get_metadata_path(podcast, enriched=False))
        samples_to_iterate = meta_data
    else:
        meta_data, num_samples = load_meta_data(get_metadata_path(podcast, enriched=True))
        samples_to_iterate = [sample for sample in meta_data if sample.de_text == ""]
        num_samples = len(samples_to_iterate)

    pipe = _setup_german_transcription_model()
    with h5py.File(h5_file, "r+" if write_to_hdf5 else "r") as h5:
        for iter_count, start_idx in enumerate(range(0, num_samples, BATCH_SIZE)):
            # Define the batch range
            end_idx = min(start_idx + BATCH_SIZE, num_samples)
            logger.debug(f"Transcribing samples {start_idx} to {end_idx}...")
            # Load batch of audio data
            audio_batch = [h5[samples_to_iterate[i].sample_name][:] for i in range(start_idx, end_idx)]

            # Perform transcription
            results = pipe(audio_batch, batch_size=BATCH_SIZE)

            # Save results to collection
            samples_to_iterate = _save_de_transcribe_results(results, samples_to_iterate, start_idx, write_to_hdf5, h5)

            if iter_count % META_WRITE_ITERATIONS == 0:
                write_meta_data(podcast, meta_data, True)

    write_meta_data(podcast, meta_data, True)
    _fix_any_long_german_segments(podcast, pipe, write_to_hdf5)


def _fix_missing_phoneme(podcast: str, pipe: pipeline, write_to_hdf5):
    meta_data, _ = load_meta_data(get_metadata_path(podcast, enriched=True))
    to_check_segments = [entry for entry in meta_data if entry.phoneme == MISSING_PHONEME]
    if not to_check_segments:
        return
    h5_file = get_h5_file(podcast)

    logger.info(f"\nTrying to fix any issues with whisper generated phonemes for podcast {podcast}.\n")
    with h5py.File(h5_file, "r+" if write_to_hdf5 else "r") as h5:
        for segment in to_check_segments:
            audio = h5[segment.sample_name][:]
            result = pipe(audio)
            phoneme = result["text"].strip()

            if phoneme == "":
                logger.error(
                    f"AGAIN NO PHONEME TRANSCRIPT GENERATED FOR {segment.sample_name}, check if you want to remove this sample")
                phoneme = MISSING_PHONEME

            segment.phoneme = phoneme
            if write_to_hdf5:
                h5[segment.sample_name].attrs["phoneme"] = phoneme
                h5.flush()
            logger.info(f"NAME: {segment.sample_name}, PHON: {phoneme}")

    write_meta_data(podcast, meta_data, True)


def _setup_phoneme_model() -> Pipeline:
    device, torch_dtype = setup_gpu_device()

    processor = Wav2Vec2Processor.from_pretrained(MODEL_AUDIO_PHONEME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_AUDIO_PHONEME)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device
    )


def _save_phoneme_results(results: list, samples_to_iterate: list[DatasetDataPoint], start_idx: int,
                          write_to_hdf5: bool, h5: h5py.File) -> list[DatasetDataPoint]:
    for idx, result in enumerate(results):
        phoneme = result["text"].strip()
        if phoneme == "":
            logger.error(
                f"NO PHONEME TRANSCRIPT GENERATED FOR {samples_to_iterate[start_idx + idx].sample_name}")
            phoneme = MISSING_PHONEME
        samples_to_iterate[start_idx + idx].phoneme = phoneme
        if write_to_hdf5:
            h5[samples_to_iterate[start_idx + idx].sample_name].attrs["phoneme"] = phoneme
        logger.info(f"NAME: {samples_to_iterate[start_idx + idx].sample_name}, PHON: {phoneme}")

    if write_to_hdf5:
        h5.flush()

    return samples_to_iterate


def audio_to_phoneme(podcast: str, write_to_hdf5: bool = True, overwrite_existing_samples: bool = True) -> None:
    logger.info("Transcribing WAV to phoneme.")
    meta_data, num_samples = load_meta_data(get_metadata_path(podcast, enriched=True))
    if overwrite_existing_samples:
        samples_to_iterate = meta_data
    else:
        samples_to_iterate = [sample for sample in meta_data if sample.phoneme == ""]
        num_samples = len(samples_to_iterate)

    h5_file = get_h5_file(podcast)
    pipe = _setup_phoneme_model()

    with h5py.File(h5_file, "r+" if write_to_hdf5 else "r") as h5:
        for iter_count, start_idx in enumerate(range(0, num_samples, BATCH_SIZE)):
            end_idx = min(start_idx + BATCH_SIZE, num_samples)
            # Load batch of audio data
            audio_batch = [h5[samples_to_iterate[i].sample_name][:] for i in range(start_idx, end_idx)]

            # Run phoneme transcription
            results = pipe(audio_batch, batch_size=BATCH_SIZE)

            # Save results to collection
            samples_to_iterate = _save_phoneme_results(results, samples_to_iterate, start_idx, write_to_hdf5, h5)

            if iter_count % META_WRITE_ITERATIONS == 0:
                write_meta_data(podcast, meta_data, True)

    write_meta_data(podcast, meta_data, True)
    _fix_missing_phoneme(podcast, pipe, write_to_hdf5)


def _merge_phoneme_of_speaker_samples(combinations: list[MergingHelper]):
    texts = []
    for entries in combinations:
        texts.append(" ".join([cut.phoneme.replace(' ', '') for cut in entries.samples]))
    return texts


def _save_dialect_results(samples: list[MergingHelper], string_most_common: str, write_to_hdf5: bool,
                          h5: h5py.File) -> None:
    for combination in samples:
        for sample in combination.samples:
            sample.did = string_most_common  # same objects as in meta_data which is saved. I know not pretty, but I was lazy...
            if write_to_hdf5:
                h5[sample.sample_name].attrs["did"] = string_most_common
            logger.info(f"NAME: {sample.sample_name}, DID: {string_most_common}")

    if write_to_hdf5:
        h5.flush()


def dialect_identification_naive_bayes_majority_voting(podcast: str, write_to_hdf5: bool = True) -> None:
    logger.info("Run Dialect Identification based on phonemes with Majority Voting of 100s samples")
    meta_data, _ = load_meta_data(get_metadata_path(podcast, enriched=True))
    speaker_merged_phoneme = assign_samples_to_speaker(meta_data, max_length=100.0)
    h5_file = get_h5_file(podcast)

    text_clf = load(MODEL_PATH_DID)
    text_clf["clf"].set_params(n_jobs=BATCH_SIZE)

    with h5py.File(h5_file, "r+" if write_to_hdf5 else "r") as h5:
        # since Python 3.7 dicts are OrderPreserving, as such OK
        for episode, segments in speaker_merged_phoneme.items():
            for speaker, samples in segments.items():
                texts = _merge_phoneme_of_speaker_samples(samples)

                predicted = text_clf.predict(texts)
                logger.debug(len(texts))
                logger.debug(predicted)
                most_common = Counter(predicted).most_common(1)[0][0]  # Get the most common prediction
                string_most_common = phon_did_cls[most_common]
                logger.info(
                    f"Most common prediction for speaker {speaker}: {most_common}, which is {string_most_common}")

                # Save results to collection
                _save_dialect_results(samples, string_most_common, write_to_hdf5, h5)

    write_meta_data(podcast, meta_data, True)


def assign_samples_to_speaker(meta_data: list, max_length: float = 30.0) -> dict:
    """
    merge together samples max_length seconds of speakers.
    :return:
    """

    def _assign_to_merged(name, speaker):
        for combined_samples_for_given_duration in speaker_to_episodes[name][speaker]:
            if combined_samples_for_given_duration.duration + sample.duration <= max_length:
                combined_samples_for_given_duration.duration += sample.duration
                combined_samples_for_given_duration.samples.append(sample)
                return True
        return False

    speaker_to_episodes = {}
    for sample in meta_data:
        episode_name = sample.orig_episode_name
        speaker = sample.speaker_id

        if episode_name not in speaker_to_episodes:
            speaker_to_episodes[episode_name] = {}

        if speaker not in speaker_to_episodes[episode_name]:
            speaker_to_episodes[episode_name][speaker] = [MergingHelper(sample)]
        else:
            if not _assign_to_merged(episode_name, speaker):
                speaker_to_episodes[episode_name][speaker].append(MergingHelper(sample))

    return speaker_to_episodes


def _run_ch_de_batch(start_idx: int, batch_size: int, num_samples: int, samples_to_iterate: list[DatasetDataPoint],
                     tokenizer: T5Tokenizer, model: PreTrainedModel, device: str) -> list:
    end_idx = min(start_idx + batch_size, num_samples)
    batch_texts = [f"[{DIALECT_TO_TAG[samples_to_iterate[i].did]}]: {samples_to_iterate[i].de_text}" for i in
                   range(start_idx, end_idx)]  # Extract batched texts from meta_data

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
    return batch_translations


def _save_ch_de_results(batch_translations: list, samples_to_iterate: list[DatasetDataPoint], start_idx: int,
                        write_to_hdf5: bool, h5: h5py.File) -> list[DatasetDataPoint]:
    for idx, ch_text in enumerate(batch_translations):
        if ch_text == "":
            logger.error(
                f"NO SWISS GERMAN TRANSCRIPT GENERATED FOR {samples_to_iterate[start_idx + idx].sample_name}")
            ch_text = MISSING_TEXT

        samples_to_iterate[start_idx + idx].ch_text = ch_text
        if write_to_hdf5:
            h5[samples_to_iterate[start_idx + idx].sample_name].attrs["ch_text"] = ch_text
        logger.info(f"DE: {samples_to_iterate[start_idx + idx].de_text}, CH: {ch_text}")

    if write_to_hdf5:
        h5.flush()

    return samples_to_iterate


def transcribe_de_to_ch(podcast: str, write_to_hdf5: bool = True, overwrite_existing_samples: bool = True) -> None:
    """
    Instead of directly transcribing audio to CH-DE we chose the approach of first transcribing it to Standard German
    and then translate it to Swiss German.
    :param podcast:
    :param write_to_hdf5:
    :param overwrite_existing_samples:
    :return:
    """
    logger.info("Transcribing German text to Swiss German text.")
    meta_data, _ = load_meta_data(get_metadata_path(podcast, enriched=True))
    meta_data_non_de = [sample for sample in meta_data if sample.did != "Deutschland" and sample.de_text != MISSING_TEXT]
    num_samples = len(meta_data_non_de)
    if overwrite_existing_samples:
        samples_to_iterate = meta_data_non_de
    else:
        samples_to_iterate = [sample for sample in meta_data_non_de if sample.ch_text == ""]
        num_samples = len(samples_to_iterate)

    h5_file = get_h5_file(podcast)
    device, _ = setup_gpu_device()
    batch_size = 6

    model = T5ForConditionalGeneration.from_pretrained(os.path.join(MODEL_PATH_DE_CH, "best-model"))
    tokenizer = T5Tokenizer.from_pretrained(MODEL_T5_TOKENIZER)
    tokenizer.add_tokens(["Ä", "Ö", "Ü"])

    model.to(device)
    model.eval()

    with h5py.File(h5_file, "r+" if write_to_hdf5 else "r") as h5:
        for iter_count, start_idx in enumerate(range(0, num_samples, batch_size)):

            batch_translations = _run_ch_de_batch(start_idx, batch_size, num_samples, samples_to_iterate, tokenizer,
                                                  model, device)

            # Save results to collection
            samples_to_iterate = _save_ch_de_results(batch_translations, samples_to_iterate, start_idx, write_to_hdf5,
                                                     h5)
            if iter_count % META_WRITE_ITERATIONS == 0:
                write_meta_data(podcast, meta_data, True)

    for sample in meta_data:
        if sample.did == "Deutschland":
            sample.ch_text = NO_CH_TEXT
    write_meta_data(podcast, meta_data, True)


if __name__ == '__main__':
    transcribe_audio_to_german("Zivadiliring")

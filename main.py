import argparse
import logging
from datetime import datetime

from src.analytics.analyze_dialect_regions import show_did_in_single_token_distribution, show_did_in_single_plot
from src.analytics.analyze_token_distribution import run_token_srf_analysis
from src.audio_to_melspec import create_mel_spectrogram
from src.segment_speech import cut_segments_on_rttm, diarize_podcast
from src.split_podcasts_to_dialect import move_podcast_to_dialect
from src.transcribe_speech import transcribe_audio_to_german, audio_to_phoneme, transcribe_de_to_ch, \
    dialect_identification_naive_bayes_majority_voting, append_missing_properties_to_h5

DEFAULT_PODCAST = "Zivadiliring"
logger = logging.getLogger(__name__)


def execute_transcription(podcast: str = DEFAULT_PODCAST, pod_type: str = "SRF"):
    logger.info(f"Processing '{podcast}'")

    diarize_podcast(podcast, pod_type)
    cut_segments_on_rttm(podcast, pod_type, write_to_hdf5=True)
    # append_missing_properties_to_h5(podcast, enriched=False)
    transcribe_audio_to_german(podcast, False, overwrite_existing_samples=True)
    audio_to_phoneme(podcast, False)  # requires docker image if executed on Cloud Compute
    dialect_identification_naive_bayes_majority_voting(podcast, False)
    transcribe_de_to_ch(podcast, False, overwrite_existing_samples=False)
    create_mel_spectrogram(podcast, True)
    append_missing_properties_to_h5(podcast, enriched=True)
    move_podcast_to_dialect(podcast)


def main(podcast: str, pod_type: str = "SRF"):
    logging.basicConfig(filename=f"{datetime.now().strftime('%Y_%m_%d_%H_%M.log')}", level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logger.info("Started")

    # execute_transcription(podcast, pod_type)
    run_token_srf_analysis()
    # show_distributions_for_dialects()

    logger.info("Finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transcribe Podcast")
    group = parser.add_argument_group("podcast")
    group.add_argument("-p", "--podcast", type=str, default=DEFAULT_PODCAST,
                       help="Name of the podcast")
    group.add_argument("-s", "--source", type=str, choices=["SRF", "YT"], default="SRF",
                       help="Source of the podcast, either SRF or YouTube")

    args = parser.parse_args()
    main(podcast=args.podcast, pod_type=args.source)

import logging
import os

import h5py
import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from src.util import load_meta_data, DATASETS_PATH, get_metadata_path, get_h5_file

BATCH_SIZE = 16

logger = logging.getLogger(__name__)


def _convert_speech_to_mel_spec(speech, sr=16000) -> np.ndarray:
    # Create Mel spectrogram
    s = librosa.feature.melspectrogram(y=speech, sr=sr, fmax=sr / 2)  # explicit default behaviour for f_max, check docs
    # Convert to log scale (dB)
    s_db = librosa.power_to_db(s, ref=np.max)

    return s_db


def plot_mel_spectrogram(audio_path: str, output_dir: str, sr=16000):
    y, sr = librosa.load(audio_path, sr=sr)
    s = _convert_speech_to_mel_spec(y)

    # Create a plot for the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(s, sr=sr, x_axis='time', y_axis='mel')  # Load audio and limit duration to 15 seconds

    # Set title and labels
    plt.title('Mel-frequency spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    # Save the spectrogram as a PNG image
    file_name = os.path.splitext(os.path.basename(audio_path))[0] + '_mel_spectrogram.png'
    output_path = os.path.join(output_dir, file_name)
    plt.savefig(output_path)
    plt.close()  # Close the plot to avoid memory issues


def create_mel_spectrogram(podcast: str, write_to_hdf5: bool = True):
    meta_data, num_samples = load_meta_data(get_metadata_path(podcast, enriched=True))
    h5_file = get_h5_file(podcast)

    # TODO: make filter for already process samples
    try:
        with h5py.File(h5_file, "r+") as h5:
            for start_idx in range(0, num_samples, BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, num_samples)
                audio_batch = [h5[meta_data[i].sample_name][:] for i in range(start_idx, end_idx)]

                jobs = [joblib.delayed(_convert_speech_to_mel_spec)(audio) for audio in audio_batch]
                out = joblib.Parallel(n_jobs=BATCH_SIZE, verbose=1)(jobs)
                # Save results
                for idx, result in enumerate(out):
                    meta_data[start_idx + idx].mel_spec = result[idx]
                    if write_to_hdf5:
                        h5[meta_data[start_idx + idx].sample_name].attrs["mel_spec"] = result[idx]
                        h5.flush()
                    logger.info(f"NAME: {meta_data[start_idx + idx].sample_name}, MelSpec: DONE")

    except Exception as e:
        logger.error(f"ERROR: {type(e).__name__} with error {str(e)}")

    with open(os.path.join(DATASETS_PATH, f"{podcast}_enriched.txt"), "wt", encoding="utf-8") as f:
        for line in meta_data:
            f.write(line.to_string())


if __name__ == '__main__':
    # Example usage:
    audio_file = 'segmented_speech/2a1077ab-9c58-4f1c-b824-f4c72fe85f59_1001.mp3'
    output_folder = 'segmented_speech/'
    plot_mel_spectrogram(audio_file, output_folder)

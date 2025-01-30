import os
import shutil

import pandas as pd
import soundfile as sf

import librosa
from pydub import AudioSegment

HOME = "/cluster/home/stucksam/_speaker"
TARGET_SAMPLING_RATE = 22050
LANG_MAP = {
    "ch_be": "Bern",
    "ch_bs": "Basel",
    "ch_gr": "Graubünden",
    "ch_in": "Innerschweiz",
    "ch_os": "Ostschweiz",
    "ch_vs": "Wallis",
    "ch_zh": "Zürich",
    "de": "Deutschland"
}

LANG_MAP_INV = {v: k for k, v in LANG_MAP.items()}


dialect = "ch_zh"
file_list = [
    f"{HOME}/{dialect}/c85640d7841a25537605847ddc4c9ea8477ceddbbb06ad6060b529506a1567fe.mp3",
    f"{HOME}/{dialect}/e61779ae1c8f24ed49bf88554c1c45258e28792df591bc472cf3fad9c9df5892.mp3",
    f"{HOME}/{dialect}/f444d9c85fcda075d137d54152129c3508a0458509bcc5bf9bf88947ab5d59d3.mp3"
]


def convert_multiple():
    for file in file_list:
        convert_single(file)


def convert_single(name: str):
    wav_file = name.replace("mp3", "wav")
    sound = AudioSegment.from_file(name)
    sound.export(wav_file, format="wav")


def check_sampling_rate():
    for file_name in os.listdir(HOME + f"/{dialect}"):
        if ".wav" in file_name:
            _, sample_rate = librosa.load(HOME + f"/{dialect}"+ "/" + file_name,
                                              sr=None)  # sr=None preserves the original sample rate
            print(f"Sample rate: {sample_rate} Hz for file {file_name}")


def resample_single(path: str, orig_sr: int = 24000, target_sr: int = TARGET_SAMPLING_RATE):
    file = path.replace(".mp3", ".wav")
    audio, _ = librosa.load(file, sr=orig_sr)  # Load with original SR
    resampled_audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)  # Resample to 22050 Hz
    sf.write(file, resampled_audio, target_sr)


def resample_wav_files(_file_list: list = file_list, orig_sr: int = 24000, target_sr: int = TARGET_SAMPLING_RATE):
    for file in _file_list:
        resample_single(file, orig_sr=orig_sr, target_sr=target_sr)


def move_to_folder():
    df = pd.read_excel("SNF_Test_Sentences.xlsx")
    for index, row in df.iterrows():
        dialect = LANG_MAP_INV[row['dialect_region']]
        shutil.copy2(f"clips__test/{row['path']}", f"../../../speakers/{dialect}/{row['path'].replace('/', '_')}")


def convert_snf_to_wav():
    for dialect in LANG_MAP.keys():
        if dialect == "de":
            continue
        result = os.listdir(f"{HOME}/{dialect}/references")
        for element in result:
            files = os.listdir(f"{HOME}/{dialect}/references/{element}")
            for clip in files:
                if ".mp3" in clip:
                    path = f"{HOME}/{dialect}/references/{element}/{clip}"
                    wav_file = path.replace("mp3", "wav")
                    if os.path.exists(wav_file):
                        continue
                    sound = AudioSegment.from_file(path)
                    sound.export(wav_file, format="wav")
                    resample_single(path, orig_sr=32000, target_sr=24000)

        orig_clips = os.listdir(f"{HOME}/{dialect}")
        for element in orig_clips:
            if os.path.isdir(element):
                continue
            if ".mp3" in element:
                path = f"{HOME}/{dialect}/{element}"
                wav_file = path.replace("mp3", "wav")
                sound = AudioSegment.from_file(path)
                sound.export(wav_file, format="wav")
                resample_single(path, orig_sr=32000, target_sr=24000)
                os.remove(path)

        # for clip in result:
        #     clip_path = f"{HOME}/{dialect}/{clip}"
        #     print(clip_path)
        #     convert_single(clip_path)
        #     resample_single(clip_path, orig_sr=32000, target_sr=24000)


def check_sample_rate_snf():
    for dialect in LANG_MAP.keys():
        result = os.listdir(f"{HOME}/{dialect}")
        for clip in result:
            if ".wav" in clip:
                _, sample_rate = librosa.load(HOME + f"/{dialect}" + "/" + clip,
                                              sr=None)  # sr=None preserves the original sample rate
                print(f"Sample rate: {sample_rate} Hz for file {clip}")


if __name__ == '__main__':
    convert_snf_to_wav()
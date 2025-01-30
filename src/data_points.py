import numpy as np


class DatasetDataPoint:
    # use an object instead of a dict since the trainer removes all columns that do not match the signature
    def __init__(
            self,
            sample_name: str,
            duration: float,
            track_start: float,
            track_end: float,
            track_id: int,
            speaker_id: str,
            output_len,
            de_text: str = "",
            phoneme: str = "",
            did: str = "",
            ch_text: str = "",
            mel_spec: np.ndarray = None
    ):
        self.dataset_name = ""
        self.sample_name = sample_name
        self.output_len = output_len
        self.duration = duration
        self.track_start = track_start
        self.track_end = track_end
        self.track_id = track_id
        self.speaker_id = speaker_id
        self.de_text = de_text
        self.phoneme = phoneme
        self.did = did
        self.ch_text = ch_text
        self.mel_spec = mel_spec

        self.orig_episode_name = self.sample_name.split("_")[0]

    @staticmethod
    def load_single_datapoint(split_properties: list):
        sample_name = split_properties[0]
        track_id = int(split_properties[1])
        duration = float(split_properties[2])
        track_start = float(split_properties[3])
        track_end = float(split_properties[4])
        speaker = split_properties[5]
        n_outputs = int(split_properties[6])
        return DatasetDataPoint(
            sample_name=sample_name,
            duration=duration,
            track_start=track_start,
            track_end=track_end,
            track_id=track_id,
            speaker_id=speaker,
            output_len=n_outputs,
            de_text=split_properties[7] if len(split_properties) > 7 else "",
            phoneme=split_properties[8] if len(split_properties) > 8 else "",
            did=split_properties[9] if len(split_properties) > 9 else "",
            ch_text=split_properties[10] if len(split_properties) > 10 else "",
            mel_spec=split_properties[11] if len(split_properties) > 11 else None
        )

    def __len__(self):
        return self.output_len

    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, dataset_name):
        self._dataset_name = dataset_name

    def to_string(self):
        to_string = f"{self.sample_name}\t{self.track_id}\t{self.duration}\t{self.track_start}\t{self.track_end}\t{self.speaker_id}\t{self.output_len}"

        if self.de_text:
            to_string += f"\t{self.de_text}"
        if self.phoneme:
            to_string += f"\t{self.phoneme}"
        if self.did:
            to_string += f"\t{self.did}"
        if self.ch_text:
            to_string += f"\t{self.ch_text}"
        # if self.mel_spec:  # array to text does not make sense.
        #     to_string += f"\t{self.mel_spec}"

        to_string += "\n"
        return to_string

    def convert_to_dialect_datapoint(self):
        return DialectDataPoint(
            self.dataset_name,
            self.sample_name,
            self.duration,
            self.speaker_id,
            self.de_text
        )


class DialectDataPoint:
    def __init__(
            self,
            dataset_name: str,
            sample_name: str,
            duration: float,
            speaker_id: str,
            de_text: str,
    ):
        self.dataset_name = dataset_name
        self.sample_name = sample_name
        self.duration = duration
        self.speaker_id = speaker_id
        self.de_text = de_text

        self.orig_episode_name = self.sample_name.split("_")[0]

    @staticmethod
    def number_of_properties():
        return 5

    @staticmethod
    def load_single_datapoint(split_properties: list):
        return DialectDataPoint(
            dataset_name=split_properties[0],
            sample_name=split_properties[1],
            duration=split_properties[2],
            speaker_id=split_properties[3],
            de_text=split_properties[4]
        )

    def to_string(self):
        return f"{self.dataset_name}\t{self.sample_name}\t{self.duration}\t{self.speaker_id}\t{self.de_text}\n"

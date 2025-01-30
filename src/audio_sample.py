class AudioSample:  #TODO delete and replace with single boi
    """
    Storage for cut audio sample information. Currently used in tandem with json, but could be moved to csv if storage
    starts to be an issue...
    """

    def __init__(self, segment: dict, dialect: str = None, text: str = None, phon: list = None,
                 melspec: list = None) -> None:
        self.start = round(segment["start"], 4)
        self.end = round(segment["end"], 4)
        self.dur = round(segment["dur"], 4)
        self.speaker: str = segment["speaker"]
        self.track: str = segment["track"]
        self.dialect: str = dialect
        self.text: str = text
        self.phon: list = phon
        self.melspec: list = melspec

    @staticmethod
    def create_audio_sample_from_dict(content: dict):
        segment = {
            "start": content["start"],
            "end": content["end"],
            "dur": content["dur"],
            "speaker": content["speaker"],
            "track": content["track"]
        }
        dialect = content.get("dialect", None)
        text = content.get("text", None)
        phon = content.get("phon", None)
        melspec = content.get("melspec", None)
        return AudioSample(segment=segment, dialect=dialect, text=text, phon=phon, melspec=melspec)

    def to_dict(self):
        data = {
            "start": self.start,
            "end": self.end,
            "dur": self.dur,
            "track": self.track,
            "speaker": self.speaker
        }
        if self.dialect is not None:
            data["dialect"] = self.dialect
        if self.text is not None:
            data["text"] = self.text
        if self.phon is not None:
            data["phon"] = self.phon
        if self.melspec is not None:
            data["melspec"] = self.melspec
        return data

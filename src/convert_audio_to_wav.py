import json
import os

from pydub import AudioSegment

from src.download_yt_audio import podcasts_collected, PODCAST_YT_METADATA_FOLDER


def convert_audio(podcast):
    print(f"Running podcast: {podcast['title']}")
    with open(f"{PODCAST_YT_METADATA_FOLDER}/{podcast['title']}.json", "r") as f:
        episodes = json.loads(f.read())

    for e_id, _ in episodes.items():
        print(f"Converting episode: {e_id}")
        e = f"{PODCAST_YT_METADATA_FOLDER}/{podcast['title']}/{e_id}.mp4"
        if os.path.exists(e):
            if os.path.exists(e.replace(".mp4", ".mp3")):
                print("mp3 already exists...")
                continue
            sound = AudioSegment.from_file(f"{PODCAST_YT_METADATA_FOLDER}/{podcast['title']}/{e_id}.mp4")
            sound.export(f"{PODCAST_YT_METADATA_FOLDER}/{podcast['title']}/{e_id}.mp3", format="mp3")
        else:
            print(f"{e_id} doesn't exist for podcast {podcast['title']}")


if __name__ == "__main__":
    # sound = AudioSegment.from_file(f"{PODCAST_YT_METADATA_FOLDER}/gNRN7olwYCo.mp4")
    # sound.export(f"{PODCAST_YT_METADATA_FOLDER}/gNRN7olwYCo.mp3", format="mp3")
    for pod in podcasts_collected:
        convert_audio(pod)

import json
import os
from os import listdir
from os.path import isfile, join

from pytubefix import YouTube, Playlist
from pytubefix.exceptions import VideoUnavailable, UnknownVideoError

PODCAST_YT_METADATA_FOLDER = "youtube_audio_crawl"

podcasts_collected = [
    {"title": "Sexologie - Wissen macht Lust",
     "url": "https://www.youtube.com/playlist?list=PL3D2QP2F5r9VDSj6YQb6Ihr_63Gxtm4L5", "lang": "mixed"},
    {"title": "Ein Buch Ein Tee", "url": "https://www.youtube.com/playlist?list=PLCospSPttrrVSk0N5Mqj1dveKZtDZNOAl",
     "lang": "ch"},
    {"title": "Ungerwegs Daheim", "url": "https://www.youtube.com/playlist?list=PLM4IdPP-Tx3W84w1GB8cn33GnuIGcqaeP",
     "lang": "ch"},
    {"title": "expectations - geplant und ungeplant kinderfrei",
     "url": "https://www.youtube.com/playlist?list=PL5ZbqYujTUkVmNCGMP4e0yFVhY8P5EC73", "lang": "ch"},
    {"title": "Wir müssen reden - Public Eye spricht Klartext",
     "url": "https://www.youtube.com/playlist?list=PLtTxFB6b5Pljl4RU6vimwfQpV490K6SQe", "lang": "de"},
    {"title": "FinanzFabio", "url": "https://www.youtube.com/playlist?list=PLGJjtm2tSyhQXU-_N2YkfqCffXhY6UHNe",
     "lang": "ch"},
    {"title": "Feel Good Podcast", "url": "https://www.youtube.com/playlist?list=PLf-k85Nq3_j-glR2im1SZv_BxqzdYdENk",
     "lang": "ch"},
    {"title": "Berner Jugendtreff", "url": "https://www.youtube.com/playlist?list=PLyWje_91744G6UAsfHjTLWDtejJdHmuYv",
     "lang": "ch"},
    {"title": "fadegrad", "url": "https://www.youtube.com/playlist?list=PL356t1Y2d_AXycvLzBF1n8ee0uM4pw9JX",
     "lang": "ch"},
    {"title": "Scho ghört", "url": "https://www.youtube.com/playlist?list=PLKaFe_fDMhQNbWvnJGC6HArb285ZUdGbz",
     "lang": "ch"},
    {"title": "Über den Bücherrand", "url": "https://www.youtube.com/playlist?list=PLPtjJ0sjI3yzhNtZUBY0_e462_gKtr90V",
     "lang": "mixed"},
    {"title": "Auf Bewährung: Leben mit Gefängnis",
     "url": "https://www.youtube.com/playlist?list=PLAD8a6PKLsRhHc-uS6fA6HTDijwE5Uwju", "lang": "mixed"},
    # {"title": "", "url": "", "lang": ""},
    # {"title": "", "url": "", "lang": ""} # https://www.podcastclub.ch/schweizer-podcasts/ https://www.podcastschmiede.ch/podcasts/
]


def _get_duration(video: YouTube):
    try:
        return video.streaming_data["formats"][0]["approxDurationMs"]
    except KeyError:
        try:
            return video.streaming_data["adaptiveFormats"][0]["approxDurationMs"]
        except KeyError:
            return "NO_DURATION"


def save_podcast_pl_metadata(podcasts: list = None, skip: bool = True) -> None:
    if podcasts is None:
        podcasts = podcasts_collected

    saved_podcasts = [f for f in listdir(PODCAST_YT_METADATA_FOLDER) if isfile(join(PODCAST_YT_METADATA_FOLDER, f))]

    for podcast in podcasts:
        if skip and f"{podcast['title']}.json" in saved_podcasts:
            continue

        episodes = {}
        pl = Playlist(podcast["url"])
        for i, video in enumerate(pl.videos):
            try:
                duration = _get_duration(video)
            except UnknownVideoError:
                print(f"Unknown video error occurred for video '{video.video_id}' with title")
                continue

            episodes[video.video_id] = {
                "title": video.title,
                "description": video.description if video.description else "NO_DESCRIPTION",
                "date_published": str(video.publish_date),
                "duration_ms": duration,
                "url": video.watch_url
            }
        with open(f"{PODCAST_YT_METADATA_FOLDER}/{podcast['title']}.json", "w") as f:
            json.dump(episodes, f)


def download_podcast(podcast: str):
    with open(f"{PODCAST_YT_METADATA_FOLDER}/{podcast}.json", "r") as f:
        episodes = json.loads(f.read())

    podcast_path = f"{PODCAST_YT_METADATA_FOLDER}/{podcast}"

    if not os.path.exists(podcast_path):
        os.mkdir(f"{podcast_path}")
        print(f"Created {podcast_path}")

    for episode, metadata in episodes.items():
        if os.path.exists(f"{podcast_path}/{episode}.mp4"):
            continue

        y = YouTube(metadata["url"])
        try:
            y.streams.get_audio_only().download(output_path=podcast_path, filename=f"{episode}.mp3",
                                                max_retries=4)
        except VideoUnavailable:
            print(f"Did not download: {episode} for {podcast} because video was not available")
            continue

        print(f"downloaded {episode} for {podcast}")


def get_duration_podcasts():
    dur = 0.0
    skipped = 0
    for p in podcasts_collected:
        e_dur = 0.0
        with open(f"{PODCAST_YT_METADATA_FOLDER}/{p['title']}.json", "r") as f:
            episodes = json.loads(f.read())
        for _id, e in episodes.items():
            e_dur2 = e["duration_ms"]
            if e_dur2 == "NO_DURATION":
                skipped += 1
                continue
            e_dur += int(e["duration_ms"]) / 1000
        print(f"Duration podcast {p['title']}: {e_dur}")
        dur += e_dur

    print(f"Duration all: {dur}")
    print(f"Skipped {skipped} episodes")


if __name__ == "__main__":
    # save_podcast_pl_metadata(podcasts=podcasts_collected, skip=True)
    for p in podcasts_collected:
        download_podcast(p['title'])
    # get_duration_podcasts()

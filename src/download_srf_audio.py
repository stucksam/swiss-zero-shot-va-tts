import json
import os
import time
from os import listdir
from os.path import isfile, join

import requests

CONSUMER_KEY = "YOUR:KEY"
CONSUMER_SECRET = "YOUR:SECRET"
AUTH_TOKEN = "YOUR:AUTH_TOKEN"

URL_BASE = "https://api.srgssr.ch"
URL_CLIENT_CREDENTIALS = f"{URL_BASE}/oauth/v1/accesstoken?grant_type=client_credentials"
URL_AUDIOS = f"{URL_BASE}/audiometadata/v2"
URL_VIDEOS = f"{URL_BASE}/videometadata/v2"

PODCAST_METADATA_FOLDER = "srf_audio_crawl"

list_of_podcasts_ch_de = [  # 17278872.0 seconds
    "Debriefing 404",
    "Digital Podcast",  # mixed, sometimes background noise
    "Dini Mundart",
    # "Dini Mundart Schnabelweid",  # sometimes background noise  <- no downloads available
    "Gast am Mittag",
    "Geek-Sofa",
    # "Giigets Die SRF 3-Alltagsphilosophie",  # small samples of background noise  <- no downloads available
    # "Morgengast",  # <- no downloads available
    "Pipifax",
    "Podcast am Pistenrand",  # can contain background noise
    "Samstagsrundschau",  # <-- big
    # "Schwiiz und dütlich",  # <-- no downloads available
    "#SRFglobal",
    "Sykora Gisler",  # mixed
    "Tagesgespräch",  # <-- big, skipped
    "Ufwärmrundi",
    "Vetters Töne",  # can contain background noise
    "Wetterfrage",
    "Zivadiliring",
    "Zytlupe"
]

list_of_podcasts_de = [
    "100 Sekunden Wissen",  # can contain background noise
    # "Kontext",  # can contain background noise
    "Kultur-Talk",
    "Kopf voran",  # mixed, can contain background noise
    "Literaturclub: Zwei mit Buch",  # mixed, can contain background noise
    "Medientalk",  # can contain background noise
    "Sternstunde Philosophie",
    "Sternstunde Religion",
    "Wirtschaftswoche",
    "Wissenschaftsmagazin"  # mixed (EN, DE, CHDE), can contain background noise
]

list_of_podcasts = list_of_podcasts_ch_de + list_of_podcasts_de


def get_duration_podcasts():
    dur = 0.0
    skipped = 0
    for p in list_of_podcasts_ch_de:
        e_dur = 0.0
        with open(f"{PODCAST_METADATA_FOLDER}/{p}.json", "r") as f:
            episodes = json.loads(f.read())
        for _id, e in episodes.items():
            e_dur2 = e["duration_ms"]
            if e_dur2 == "NO_DURATION":
                skipped += 1
                continue
            e_dur += int(e["duration_ms"]) / 1000
        print(f"{e_dur} for podcast {p}: ")
        dur += e_dur

    print(f"Duration all: {dur}")
    print(f"Skipped {skipped} episodes")


def _check_and_load_response(response: requests.Response) -> dict:
    if response.status_code in [200, 203]:
        return json.loads(response.text)
    else:
        raise RuntimeError(f"Failed to get response. Response code {response.status_code} with message {response.text}")


def get_access_token() -> dict:
    headers = {
        "Authorization": "Basic " + AUTH_TOKEN,
        "Cache-Control": "no-cache",
        "Content-Length": "0",
    }
    response = requests.post(URL_CLIENT_CREDENTIALS, headers=headers)
    return _check_and_load_response(response)


def _collect_metadata(media: list, current_podcast) -> dict:
    episodes = {}
    for episode in media:
        if episode["show"]["title"].lower() == current_podcast.lower():
            episodes[episode["id"]] = {
                "title": episode["title"],
                "description": episode.get("description", "NO_DESCRIPTION"),
                "date_published": episode.get("date", "NO_DATE"),
                "duration_ms": episode["duration"],
                "download_available": episode["downloadAvailable"],
                "subtitles_available": episode["subtitlesAvailable"],
                "url": episode.get("podcastHdUrl", "NO_URL"),
            }
    return episodes


def save_podcast_metadata(podcasts: list = None, skip: bool = True) -> None:
    if podcasts is None:
        podcasts = list_of_podcasts

    access_token = get_access_token()
    headers = {
        "Authorization": f"{access_token['token_type']} {access_token['access_token']}",
        "Cache-Control": "no-cache"
    }

    saved_podcasts = [f for f in listdir(PODCAST_METADATA_FOLDER) if isfile(join(PODCAST_METADATA_FOLDER, f))]

    for podcast in podcasts:
        if skip and f"{podcast}.json" in saved_podcasts:
            continue
        params = {
            "bu": "srf",
            "q": podcast,
            "pageSize": 100
        }
        response = requests.get(URL_AUDIOS + "/audios/search", headers=headers, params=params)
        json_response = _check_and_load_response(response)

        episodes = {}
        total_episodes = json_response["total"]
        print(f"Getting podcast {podcast} with total number of episodes: {json_response['total']}")
        episodes = episodes | _collect_metadata(json_response["searchResultListMedia"], podcast)

        while "next" in json_response:
            params = {
                "bu": "srf",
                "q": podcast,
                "next": json_response["next"].split("?")[1].replace("next=", "").split("&")[0]
            }
            try:
                response = requests.get(URL_AUDIOS + "/audios/search", headers=headers, params=params)
                json_response = _check_and_load_response(response)
            except Exception as e:
                print(str(e))
                break
            episodes = episodes | _collect_metadata(json_response["searchResultListMedia"], podcast)

        print(f"Expected number of podcasts: {total_episodes}, saved {len(episodes)}")
        with open(f"{PODCAST_METADATA_FOLDER}/{podcast}.json", "w") as f:
            json.dump(episodes, f)


def download_podcasts(podcast: str) -> None:
    with open(f"{PODCAST_METADATA_FOLDER}/{podcast}.json", "r") as f:
        episodes = json.loads(f.read())

    podcast_path = f"{PODCAST_METADATA_FOLDER}/{podcast}"
    if not os.path.exists(podcast_path):
        os.mkdir(f"{podcast_path}")
        print(f"Created {podcast_path}")

    for episode, metadata in episodes.items():
        if not metadata["download_available"] or os.path.exists(f"{podcast_path}/{episode}.mp3"):
            continue

        response = requests.get(metadata["url"], allow_redirects=True)
        with open(f"{PODCAST_METADATA_FOLDER}/{podcast}/{episode}.mp3", 'wb') as f:
            f.write(response.content)

        print(f"downloaded {episode} for {podcast}")
        time.sleep(1)


def save_video_metadata():
    access_token = get_access_token()
    headers = {
        "Authorization": f"{access_token['token_type']} {access_token['access_token']}",
        "Cache-Control": "no-cache"
    }

    params = {
        "bu": "srf",
        "q": "Arena"
    }
    response = requests.get(URL_VIDEOS + "/tv_shows", headers=headers, params=params)
    json_response = _check_and_load_response(response)
    arena_id = json_response["searchResultShowList"][0]["id"]
    video_count = json_response["searchResultShowList"][0]["videoCount"]

    params = {
        "bu": "srf"
    }
    response = requests.get(URL_VIDEOS + f"/tv_shows/{arena_id}", headers=headers, params=params)
    arena_show_json = _check_and_load_response(response)

    x = 1
    b = x + 1


if __name__ == "__main__":
    # save_podcast_metadata()
    save_podcast_metadata(["Pipifax"])
    # for p in ["Pipifax"]:
    #     download_podcasts(p)

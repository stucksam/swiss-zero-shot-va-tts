import json
import logging
import os
from collections import Counter

import pandas as pd
import seaborn as sns
import spacy
from matplotlib import pyplot as plt
from transformers import AutoTokenizer

from src.analytics.analyze_text_generation import ANALYTICS_PATH
from src.util import DIALECT_DATA_PATH, load_meta_data, DATASETS_PATH

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")

logger = logging.getLogger(__name__)
nlp = spacy.load('de_core_news_sm')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", use_auth_token=HF_ACCESS_TOKEN)


def load_snf_sentences() -> list[str]:
    sentences = []
    txt_files = [f for f in os.listdir(DIALECT_DATA_PATH) if f.endswith('.txt') and not f.startswith("ch_") and not f.startswith("more_than") and not f.startswith("token_counted")]
    for file in txt_files:
        meta_data, _ = load_meta_data(f"{DIALECT_DATA_PATH}/{file}")
        sentences.extend([sample.de_text for sample in meta_data if
                          sample.dataset_name in ["SwissDial", "SNF"] and sample.de_text != "NO_TEXT"])
    return sentences


def load_srf_sentences() -> list[str]:
    sentences = []
    txt_files = [f for f in os.listdir(DIALECT_DATA_PATH) if f.endswith('.txt')and not f.startswith("ch_") and not f.startswith("more_than") and not f.startswith("token_counted")]
    for file in txt_files:
        meta_data, _ = load_meta_data(f"{DIALECT_DATA_PATH}/{file}")
        sentences.extend([sample.de_text for sample in meta_data if
                          sample.dataset_name not in ["SwissDial", "SNF"] and sample.de_text != "NO_TEXT"])
    return sentences


def load_srf_tokens() -> list[int]:
    tokens = []
    txt_files = [f for f in os.listdir(DIALECT_DATA_PATH) if f.endswith('.txt') and f.startswith("token_counted")]
    for file in txt_files:
        with open(f"{DIALECT_DATA_PATH}/{file}", "rt", encoding="utf-8") as meta_file:
            for line in meta_file:
                line_split = line.split("\t")
                if line_split[0] not in ["SwissDial", "SNF"]:
                    tokens.append(int(line_split[-1]))

    return tokens


def load_srf_durations() -> list[int]:
    tokens = []
    txt_files = [f for f in os.listdir(DIALECT_DATA_PATH) if f.endswith('.txt') and f.startswith("token_counted")]
    for file in txt_files:
        with open(f"{DIALECT_DATA_PATH}/{file}", "rt", encoding="utf-8") as meta_file:
            for line in meta_file:
                line_split = line.split("\t")
                if line_split[0] not in ["SwissDial", "SNF"]:
                    tokens.append(int(float(line_split[2])))

    return tokens

def load_snf_durations() -> list[int]:
    tokens = []
    txt_files = [f for f in os.listdir(DIALECT_DATA_PATH) if f.endswith('.txt') and f.startswith("token_counted")]
    for file in txt_files:
        with open(f"{DIALECT_DATA_PATH}/{file}", "rt", encoding="utf-8") as meta_file:
            for line in meta_file:
                line_split = line.split("\t")
                if line_split[0] in ["SNF"]:
                    tokens.append(int(float(line_split[2])))

    return tokens


def plot_token_distro(x, y, t_type: str = "spaCy", p_type: str = "SRF") -> None:
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x, y=y)
    plt.xlabel('Number of Tokens', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.title(f"{p_type} Token Distribution with {t_type}", fontsize=16)
    plt.xticks(ticks=range(0, 391, 10), rotation=45)  # Adjust tick step if necessary
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  # Avoid label overlapping
    plt.savefig(f"{ANALYTICS_PATH}/token_distribution_{p_type}_{t_type}.png")
    plt.show()


def run_token_analysis(sentences: list[str], p_type: str) -> None:
    # Tokenize with spaCy
    token_counts = [len(nlp(sentence)) for sentence in sentences]

    # Count occurrences of each token length
    token_distribution = Counter(token_counts)

    # Prepare data for plotting
    x = sorted(token_distribution.keys())  # Unique token counts
    y = [token_distribution[t] for t in x]  # Corresponding frequencies
    plot_token_distro(x, y, p_type=p_type)

    # Tokenize with llama
    token_counts = [len(tokenizer.tokenize(sentence)) for sentence in sentences]
    token_distribution = Counter(token_counts)
    x = sorted(token_distribution.keys())  # Unique token counts
    y = [token_distribution[t] for t in x]  # Corresponding frequencies
    plot_token_distro(x, y, "llama", p_type)


def snf_token_distribution():
    sentences = load_snf_sentences()
    run_token_analysis(sentences, p_type="SNF")


def srf_token_distribution():
    sentences = load_srf_sentences()
    run_token_analysis(sentences, p_type="SRF")


def show_token_distribution():
    snf_token_distribution()
    srf_token_distribution()

def get_small_sentences():
    sentences = load_srf_sentences()
    token_counts = [len(nlp(sentence)) for sentence in sentences]

    iter_count = 0
    for idx, cnt in enumerate(token_counts):
        if iter_count % 100 == 0:
            break
        if cnt < 7 :
            logger.info()


def run_token_srf_analysis():
    tokens = load_srf_tokens()
    # Count occurrences of each token length
    token_distribution = Counter(tokens)

    # Prepare data for plotting
    x = sorted(token_distribution.keys())  # Unique token counts
    x = [t for t in x if t <= 6]
    print(len(x))
    y = [token_distribution[t] for t in x]  # Corresponding frequencies
    print(sum(y))
    print(y)
    df = pd.DataFrame({'Tokens': x, 'Samples': y})
    t_type = "spaCy"
    p_type= "SNF"
    plt.figure(figsize=(10, 6))
    sns.barplot(df, x="Tokens", y="Samples")
    plt.xlabel('Number of Tokens', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.title(f"{p_type} Token Distribution with {t_type}", fontsize=16)
    # plt.xticks(ticks=range(0, len(x), 3), rotation=45)  # Adjust tick step if necessary
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  # Avoid label overlapping
    plt.savefig(f"{ANALYTICS_PATH}/token_distribution_{p_type}_{t_type}.png")
    plt.show()

def run_time_analysis():
    time = load_snf_durations()
    # Count occurrences of each token length
    time_distribution = Counter(time)

    # Prepare data for plotting
    x = sorted(time_distribution.keys())  # Unique token counts
    y = [time_distribution[t] for t in x]  # Corresponding frequencies

    df = pd.DataFrame({'Time': x, 'Samples': y})
    t_type = "spaCy"
    p_type= "STT4SG-350"
    plt.figure(figsize=(10, 6))
    sns.barplot(df, x="Time", y="Samples")
    plt.xlabel('Duration of Samples in seconds', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.title(f"{p_type}-corpus Time Distribution", fontsize=16)
    plt.xticks(rotation=45)  # Adjust tick step if necessary
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  # Avoid label overlapping
    plt.savefig(f"{ANALYTICS_PATH}/time_distribution_{p_type}.png")
    plt.show()

def count_no_text_and_phoneme():
    thrown_out_de = 0
    thrown_out_phon = 0
    both_missing = 0
    only_de_missing = 0
    only_phon_missing = 0
    total = 0
    collect = {}
    txt_files = [f for f in os.listdir(DATASETS_PATH) if f.endswith('_enriched.txt')]
    for file in txt_files:
        meta_data, _ = load_meta_data(f"{DATASETS_PATH}/{file}")
        srf_meta = [sample for sample in meta_data if sample.dataset_name not in ["SwissDial", "SNF"]]
        de = len([sample for sample in srf_meta if sample.de_text == "NO_TEXT"])
        phon = len([sample for sample in srf_meta if sample.phoneme == "NO_PHONEME"])
        both = len([sample for sample in srf_meta if sample.de_text == "NO_TEXT" and sample.phoneme == "NO_PHONEME"])
        only_de = len([sample for sample in srf_meta if sample.de_text == "NO_TEXT" and not sample.phoneme == "NO_PHONEME"])
        only_phon = len([sample for sample in srf_meta if not sample.de_text == "NO_TEXT" and sample.phoneme == "NO_PHONEME"])
        total += len(srf_meta)
        collect[file.replace("_enriched.txt", "")] = {
            "no_de": de,
            "no_phon": phon,
            "no_both": both,
            "no_only_de": only_de,
            "no_only_phon": only_phon
        }
        thrown_out_de += de
        thrown_out_phon += phon
        both_missing += both
        only_de_missing += only_de
        only_phon_missing+= only_phon

    print(json.dumps(collect, indent=4))
    print(f"No DE Text: {thrown_out_de}")
    print(f"No Phoneme Text: {thrown_out_phon}")
    print(f"Both missing: {both_missing}")
    print(f"Only de missing: {only_de_missing}")
    print(f"Only Phonn missing: {only_phon_missing}")
    print(f"Total samples: {total}")

if __name__ == '__main__':
    run_time_analysis()

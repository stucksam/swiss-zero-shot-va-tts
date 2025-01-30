import logging
import os
from collections import Counter

import numpy as np
import seaborn as sns
import spacy
from matplotlib import pyplot as plt

from src.data_points import DialectDataPoint
from src.util import ANALYTICS_PATH, DIALECT_DATA_PATH

logger = logging.getLogger(__name__)
nlp = spacy.load('de_core_news_sm')


def plot_property(distro: Counter, title: str, y_axis: str, x_axis: str):
    x = sorted(distro.keys())  # Unique token counts
    y = [distro[t] for t in x]  # Corresponding frequencies
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x, y=y, palette="magma")
    plt.xlabel(x_axis, fontsize=14)
    plt.ylabel(y_axis, fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.savefig(f"{ANALYTICS_PATH}/token_distribution_{p_type}_{t_type}.png")
    plt.show()


def load_dialect_data(path: str) -> tuple[list, int]:
    sample_list = []
    with open(path, "rt", encoding="utf-8") as meta_file:
        for line in meta_file:
            split_line = line.replace('\n', '').split('\t')
            sample_list.append(DialectDataPoint.load_single_datapoint(split_line))
    return sample_list, len(sample_list)


def show_distributions_for_dialects():
    txt_files = [f for f in os.listdir(ANALYTICS_PATH + "/did_distribution") if
                 f.endswith('.txt') and not f.startswith("ch_")]  # training de text

    for txt in txt_files:
        dialect = txt.replace(".txt", "")
        meta_data, num_samples = load_dialect_data(ANALYTICS_PATH + "/did_distribution/" + txt)
        logger.info(f"{dialect}: {num_samples}")

        duration = [int(float(sample.duration)) for sample in meta_data]
        dur_distro = Counter(duration)
        title = f"{dialect} Duration distribution"
        plot_property(dur_distro, title, y_axis="Number of Samples", x_axis='Number of seconds (rounded)')

        tokens = [len(nlp(sample.de_text)) for sample in meta_data]
        token_distro = Counter(tokens)
        title = f"{dialect} Token distribution"
        plot_property(token_distro, title, y_axis="Number of Samples", x_axis='Number of Tokens')


def show_did_in_single_plot():
    txt_files = [f for f in os.listdir(ANALYTICS_PATH + "/did_distribution") if
                 f.endswith('.txt') and not f.startswith("ch_")]  # training de text

    # Generate example data for 8 plots
    x = np.linspace(0, 10, 100)
    data = [np.sin(x + i) for i in range(8)]  # Example data for 8 plots

    # Create the figure and axes for 8 subplots (4 rows, 2 per row)
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))  # 4 rows, 2 columns
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing
    deutschland = False
    # Plot each dataset in its respective subplot
    for i, txt in enumerate(txt_files):
        dialect = txt.replace(".txt", "")
        # if dialect == "Deutschland":
        #     deutschland = True
        #     continue
        meta_data, num_samples = load_dialect_data(ANALYTICS_PATH + "/did_distribution/" + txt)
        meta_data = [sample for sample in meta_data if sample.dataset_name not in ["SwissDial", "SNF"] and sample.de_text != "NO_TEXT"]
        # logger.info(f"{dialect}: {num_samples}")
        duration_sum = sum([float(sample.duration) for sample in meta_data])
        logger.info(f"{dialect}: {duration_sum}")
        duration = [int(float(sample.duration)) for sample in meta_data]
        dur_distro = Counter(duration)

        x = sorted(dur_distro.keys())  # Unique token counts
        y = [dur_distro[t] for t in x]  # Corresponding frequencies
        # if deutschland:
        #     i = i - 1
        axes[i].bar(x, y)
        axes[i].set_title(f"{dialect} Duration distribution")
        # axes[i].set_ylim(0, 12000)
        # axes[i].set_xlim(0, 15)
        axes[i].set_xticks(range(2, 16, 1))
        axes[i].legend()
        axes[i].grid()

    # Hide the last (empty) subplot
    axes[-1].axis('off')
    # axes[-2].axis('off')


    plt.tight_layout()
    plt.savefig(ANALYTICS_PATH + "/did_duration.png")
    plt.show()


def show_did_in_single_token_distribution():
    txt_files = [f for f in os.listdir(DIALECT_DATA_PATH) if f.endswith('.txt') and f.startswith("token_counted")]

    # Generate example data for 8 plots
    x = np.linspace(0, 10, 100)
    data = [np.sin(x + i) for i in range(8)]  # Example data for 8 plots

    # Create the figure and axes for 8 subplots (4 rows, 2 per row)
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))  # 4 rows, 2 columns
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing
    deutschland = False
    # Plot each dataset in its respective subplot
    for i, txt in enumerate(txt_files):
        dialect = txt.replace(".txt", "").replace("token_counted_", "")
        if dialect == "Deutschland":
            deutschland = True
            continue
        tokens = []
        with open(DIALECT_DATA_PATH + "/" + txt, "rt", encoding="utf-8") as meta_file:
            for line in meta_file:
                line_split = line.split("\t")
                if line_split[0] in ["SwissDial", "SNF"] and line_split[4] != "NO_TEXT":
                    tokens.append(int(line_split[-1]))

        logger.info(f"{dialect}: {sum(tokens)}")
        # logger.info(f"{dialect}: {len(tokens)}")

        token_distro = Counter(tokens)

        x = sorted(token_distro.keys())  # Unique token counts
        x = [t for t in x if t <= 200]
        y = [token_distro[t] for t in x]  # Corresponding frequencies
        if deutschland:
            i = i - 1
        axes[i].bar(x, y)
        axes[i].set_title(f"{dialect} Token Distribution")
        axes[i].set_xticks(range(0, len(x), 5))
        axes[i].set_ylim(0, 7000)
        axes[i].set_xlim(0, 35)
        axes[i].set_xticks(range(0, 35, 5))
        axes[i].legend()
        axes[i].grid()

    # Hide the last (empty) subplot
    axes[-1].axis('off')
    axes[-2].axis('off')


    plt.tight_layout()
    plt.savefig(ANALYTICS_PATH + "/did_duration.png")
    plt.show()




def write_out_more_than_six_tokens():
    txt_files = [f for f in os.listdir(ANALYTICS_PATH + "/did_distribution") if
                 f.endswith('.txt') and not f.startswith("ch_")]  # training de text
    for txt in txt_files:
        dialect = txt.replace(".txt", "")
        meta_data, num_samples = load_dialect_data(ANALYTICS_PATH + "/did_distribution/" + txt)
        logger.info(f"{dialect}: {num_samples}")

        with open(ANALYTICS_PATH + "/did_distribution/" + "more_than_6_" + txt, "wt", encoding="utf-8") as meta_file:
            for sample in meta_data:
                tokens = len(nlp(sample.de_text))
                if tokens > 6:
                    meta_file.write(sample.to_string())
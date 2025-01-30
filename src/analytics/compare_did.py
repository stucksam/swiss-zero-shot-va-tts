import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

file_naive = "Zivadiliring_enriched_DID_DE_included.txt"
file_majority = "Zivadiliring_enriched_DID_majority.txt"
file_ch = "Zivadiliring_enriched_DID_CH_only.txt"
file_whisper = "Zivadiliring_enriched_DID_whisper.txt"


def read_file_content(file_name: str) -> list:
    with open(file_name, "r", encoding="utf-8") as f:
        content = f.read().strip().split("\n")
    return content


def compare_old():
    content_naive = read_file_content(file_ch)
    content_whisper = read_file_content(file_whisper)

    compare_did = 0
    total = len(content_naive)
    for i, line in enumerate(content_naive):
        line = line.split("\t")
        line_whisper = content_whisper[i]
        line_whisper_name = line_whisper.split("NAME: ")[1].split(",")[0]
        line_whisper_did = line_whisper.split("DID: ")[1].strip()
        assert line[0] == line_whisper_name
        print(f"{line[9]} <-> {line_whisper_did}")
        print(f"{line[0]} <-> {line_whisper_did}")
        if line[9] == line_whisper_did:
            compare_did += 1

    print(f"{compare_did} of {total} samples are identical.")


def compare():
    content_naive = read_file_content(file_ch)
    content_whisper = read_file_content(file_whisper)

    saved = {}
    for i, line in enumerate(content_naive):
        line = line.split("\t")
        print(line[0])
        saved[line[0]] = {"did_n": line[9], "did_w": ""}

    for i, line in enumerate(content_whisper):
        line_whisper_name = line.split("NAME: ")[1].split(",")[0]
        line_whisper_did = line.split("DID: ")[1].strip()

        saved[line_whisper_name]["did_w"] = line_whisper_did

    compare_did = 0
    total = len(content_naive)
    for cut, entry in saved.items():
        print(entry)
        if entry["did_n"] == entry["did_w"]:
            compare_did += 1

    print(f"{compare_did} of {total} samples are identical.")


def compare_did_direct_and_majority():
    def get_did(content: list):
        did_col = {
            "Zürich": 0,
            "Innerschweiz": 0,
            "Wallis": 0,
            "Graubünden": 0,
            "Ostschweiz": 0,
            "Basel": 0,
            "Bern": 0,
            "Deutschland": 0
        }
        for line in content:
            did = line.split("\t")[9]
            did_col[did] += 1

        return did_col

    def plot_distribution(dialects: dict, model_type: str):
        plt.figure(figsize=(10, 6))
        plt.bar(dialects.keys(), dialects.values(), color='lightcoral')
        plt.xlabel("Dialects")
        plt.ylabel("Number of Occurrences")
        plt.title(f"Distribution of Dialect Occurrences {model_type}")
        plt.xticks(rotation=45)
        plt.show()

    def compare_two_datasets(content_1: list, content_2: list, name1_vs_name2: str):
        diff = 0
        for i, entry in enumerate(content_1):
            entry = entry.split("\t")
            compare_entry = content_2[i].split("\t")

            if entry[0] == compare_entry[0] and entry[9] != compare_entry[9]:
                print(f"{entry[0]} different ({entry[5]}), DID: {entry[9]} <-> {compare_entry[9]}")
                diff += 1
        print(f"Have {diff} different entries for {name1_vs_name2}\n")

    title_naive = "Naive + DE"
    title_majority = "Majority + DE"
    title_ch_only = "Naive CH only"
    title_whisper = "Whisper CH"

    color_naive = "#4C72B0"
    color_majority = "#DD8452"
    color_ch_only = "#55A868"
    color_whisper = "#C44E52"

    content_naive = read_file_content(file_naive)
    content_majority = read_file_content(file_majority)
    content_ch_only = read_file_content(file_ch)
    content_whisper = read_file_content(file_whisper)

    naive_did = get_did(content_naive)
    majority_did = get_did(content_majority)
    ch_only_did = get_did(content_ch_only)
    ch_only_did["Deutschland"] = 0  # no issues with class sizes
    whisper_did = get_did(content_whisper)
    whisper_did["Deutschland"] = 0  # no issues with class sizes

    plot_distribution(naive_did, title_naive)
    plot_distribution(majority_did, title_majority)
    plot_distribution(ch_only_did, title_ch_only)
    plot_distribution(whisper_did, title_whisper)

    # Plotting each dataset with adjusted positions
    fig, ax = plt.subplots(figsize=(12, 7))
    num_values = len(naive_did.values())
    bar_width = 0.2
    dialects = list(naive_did.keys())
    x = np.arange(len(naive_did.keys()))
    offset = (num_values - 1) * bar_width / 4.5  # Calculate the offset for centering

    ax.bar(x - offset + 0 * bar_width, naive_did.values(), width=bar_width, color=color_naive, label=title_naive)
    ax.bar(x - offset + 1 * bar_width, majority_did.values(), width=bar_width, color=color_majority, label=title_majority)
    ax.bar(x - offset + 2 * bar_width, ch_only_did.values(), width=bar_width, color=color_ch_only, label=title_ch_only)
    ax.bar(x - offset + 3 * bar_width, whisper_did.values(), width=bar_width, color=color_whisper, label=title_whisper)

    ax.set_xticks(x + bar_width * (num_values - 1) / 2)
    ax.set_xticklabels(dialects)

    # Adding labels and title
    plt.xlabel("Dialects")
    plt.ylabel("Number of Occurrences")
    plt.title("Distribution of Dialect Occurrences Across Four Models")
    plt.xticks(x, dialects)
    plt.legend(title="Datasets")
    plt.show()

    # Calculating total occurrences for each dataset to compute percentages
    fig, ax = plt.subplots(figsize=(12, 7))
    total_data_1 = sum(naive_did.values())
    total_data_2 = sum(majority_did.values())
    total_data_3 = sum(ch_only_did.values())
    total_data_4 = sum(whisper_did.values())

    # Converting occurrences to percentages
    percent_data_1 = {dialect: (count / total_data_1) * 100 for dialect, count in naive_did.items()}
    percent_data_2 = {dialect: (count / total_data_2) * 100 for dialect, count in majority_did.items()}
    percent_data_3 = {dialect: (count / total_data_3) * 100 for dialect, count in ch_only_did.items()}
    percent_data_4 = {dialect: (count / total_data_4) * 100 for dialect, count in whisper_did.items()}

    # Plotting each dataset with adjusted positions
    ax.bar(x - offset + 0 * bar_width, percent_data_1.values(), width=bar_width, color=color_naive, label=title_naive)
    ax.bar(x - offset + 1 * bar_width, percent_data_2.values(), width=bar_width, color=color_majority, label=title_majority)
    ax.bar(x - offset + 2 * bar_width, percent_data_3.values(), width=bar_width, color=color_ch_only, label=title_ch_only)
    ax.bar(x - offset + 3 * bar_width, percent_data_4.values(), width=bar_width, color=color_whisper, label=title_whisper)

    # Adding labels, title, and y-axis percentage formatting
    plt.xlabel("Dialects")
    plt.ylabel("Percentage of Occurrences (%)")
    plt.title("Percentage Distribution of Dialect Occurrences Across Four Models")
    plt.xticks(x, dialects)
    plt.legend(title="Datasets")
    plt.show()

    df = pd.DataFrame({
        "Dialect": list(naive_did.keys()),
        title_naive: list(naive_did.values()),
        title_majority: list(majority_did.values()),
        title_ch_only: list(ch_only_did.values()),
        title_whisper: list(whisper_did.values())
    })
    print(df)

    df_percentage = pd.DataFrame({
        "Dialect": list(percent_data_1.keys()),
        title_naive: list(percent_data_1.values()),
        title_majority: list(percent_data_2.values()),
        title_ch_only: list(percent_data_3.values()),
        title_whisper: list(percent_data_4.values())
    })
    print(df_percentage)

    compare_two_datasets(content_naive, content_majority, "Naive vs. Majority")
    # only 131, mostly where naive used Innerschwiiz, majority had Zurich, while majority introduced a bit more wallis
    compare_two_datasets(content_naive, content_ch_only, "Naive vs. CH-Only")
    # Have 179 different entries for Naive vs. CH-Only, innerschweiz -> zuirich and Wallis -> ostschweiz
    compare_two_datasets(content_majority, content_ch_only, "Majority vs. CH-Only")
    # 288 different, mostly switch outs between zurich and innerschwiiz and Wallis and Ostschweiz


def analyze_ostschweiz():
    content_majority = read_file_content(file_majority)
    ostschweiz = []
    for entry in content_majority:
        did = entry.split("\t")[9]
        if did == "Ostschweiz":
            ostschweiz.append(entry)

    for entry in ostschweiz:
        split_entry = entry.split("\t")

        print(f"{split_entry[0]} as {split_entry[5]}")


if __name__ == '__main__':
    compare_did_direct_and_majority()

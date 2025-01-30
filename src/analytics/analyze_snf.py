import logging
import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

datapath = "src/config"
logger = logging.getLogger(__name__)

def get_distribution_of_property(df: pd.DataFrame, property: str) -> pd.DataFrame:
    # Count each unique age value
    counts = df[property].value_counts().reset_index()
    counts.columns = [property, 'count']  # Rename columns for clarity

    # Add a 'percentage' column
    total_count = counts["count"].sum()
    counts["percentage"] = round((counts["count"] / total_count) * 100, 4)
    return counts


def print_distribution(df, col, name: str = "") -> None:
    unique_values_count = df[col].nunique()
    col_description = df[col].describe()

    # Count each unique age value
    counts = get_distribution_of_property(df, col)
    logger.info(counts)
    logger.info(col_description)

    # Plotting the distribution of 'age'
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], bins=unique_values_count, kde=False, color='skyblue')  # kde=True adds a smooth curve

    # Customizing plot
    plt.title(f"{col} Distribution {name}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

    return counts


def analyze_properties():
    files = ["valid.tsv", "train_all.tsv", "test.tsv"]
    for split in files:
        df = pd.read_csv(os.path.join(datapath, split), sep="\t")
        print_distribution(df, "age", name=split.replace(".tsv", ""))
        print_distribution(df, "gender", name=split.replace(".tsv", ""))

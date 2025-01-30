import logging
import os
import random
import shutil

import jiwer
import pandas as pd
import seaborn as sns
from bert_score import score as bert_score
from matplotlib import pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.transcribe_speech import MODEL_T5_TOKENIZER, MODEL_PATH_DE_CH
from src.util import setup_gpu_device, load_meta_data, DATASETS_PATH, DIALECT_TO_TAG, get_metadata_path, ANALYTICS_PATH

logger = logging.getLogger(__name__)
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.expand_frame_repr", False)  # Prevent line breaks in large DataFrames

BATCH_SIZE = 8


def select_entries():
    meta_data, _ = load_meta_data(get_metadata_path("Zivadiliring", enriched=True))

    samples = random.sample(meta_data, 100)

    data = {
        "sample_id": [s.sample_name for s in samples],
        "speaker": [s.speaker_id for s in samples],
        "duration": [s.duration for s in samples],
        "de_text_whisper": [s.de_text for s in samples],
        "de_text_fhnw": ["" for _ in samples],
        "de_text_manual": ["" for _ in samples],
        "did": [s.did for s in samples],
        "did_manual": ["" for _ in samples],
        "ch_text_t5": [s.ch_text for s in samples],
        "ch_text_manual": ["" for _ in samples]
    }

    df = pd.DataFrame(data)

    for sample in samples:
        # Copy file example.txt into directory test/
        shutil.copy(f"{DATASETS_PATH}/Zivadiliring/{sample.orig_episode_name}/{sample.sample_name}.mp3",
                    f"{ANALYTICS_PATH}/samples")

    df.to_csv(f"{ANALYTICS_PATH}/select_entries.csv", index=False, encoding="utf-8")


def add_ch_generated_sentences_from_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    num_samples = len(df)
    device, _ = setup_gpu_device()

    model = T5ForConditionalGeneration.from_pretrained(os.path.join(MODEL_PATH_DE_CH, "best-model"))
    tokenizer = T5Tokenizer.from_pretrained(MODEL_T5_TOKENIZER)
    tokenizer.add_tokens(["Ä", "Ö", "Ü"])
    model.to(device)

    for start_idx in range(0, num_samples, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, num_samples)
        batch = df.iloc[start_idx:end_idx]
        batch_texts = [f"[{DIALECT_TO_TAG[row['did']]}]: {row['de_text_manual']}" for index, row in batch.iterrows()]

        # Tokenize the batch of sentences
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=300)

        # Move input tensors to the device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Generate translations
        output_ids = model.generate(input_ids=input_ids, max_length=300, attention_mask=attention_mask, num_beams=5,
                                    num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

        batch_translations = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        batch_translations = [x.replace("Ä ", "Ä").replace("Ü ", "Ü").replace("Ö ", "Ö") for x in
                              batch_translations]

        # Save results
        for idx, result in enumerate(batch_translations):
            df.at[start_idx + idx, "ch_text_t5_manual"] = batch_translations[idx]

    return df


def calculate_scores(reference: list, hypothesis: list) -> pd.DataFrame:
    reference = [sentence.strip().lower() for sentence in reference]
    hypothesis = [sentence.strip().lower() for sentence in hypothesis]
    single_results = {
        "wer": [],
        "mer": [],
        "wil": [],
        "cer": [],
        "bert_score": []
    }
    for i, ref in enumerate(reference):
        hypo = hypothesis[i]
        output = jiwer.process_words(ref, hypo)
        single_results["wer"].append(output.wer)
        single_results["mer"].append(output.mer)
        single_results["wil"].append(output.wil)
        single_results["cer"].append(jiwer.process_characters(ref, hypo).cer)

        # Calculate BERTScore
        P, R, F1 = bert_score([hypo], [ref], lang="de")
        _bert_score = F1.mean().item()
        single_results["bert_score"].append(_bert_score)

    # Calculate BLEU Score
    reference_split = [sentence.split(" ") for sentence in reference]
    hypothesis_split = [sentence.split(" ") for sentence in hypothesis]
    bleu_scores = [sentence_bleu([ref], hyp) for ref, hyp in zip(reference_split, hypothesis_split)]
    result = {
        "reference": reference,
        "hypothesis": hypothesis,
        "bleu_score": bleu_scores
    }
    result.update(single_results)
    return pd.DataFrame(result)


def calc_scores_lower(comparison: pd.DataFrame) -> pd.DataFrame:
    comparison["wer"] = 0.0
    comparison["wer_lower"] = 0.0
    comparison["mer"] = 0.0
    comparison["mer_lower"] = 0.0
    comparison["wil"] = 0.0
    comparison["wil_lower"] = 0.0
    comparison["cer"] = 0.0
    comparison["cer_lower"] = 0.0
    comparison["bert_score"] = 0.0

    for idx, row in comparison.iterrows():
        hypo = row["hypothesis"].strip()
        ref = row["reference"].strip()
        hypo_low = hypo.lower()
        ref_low = ref.lower()
        output = jiwer.process_words(ref, hypo)
        output_low = jiwer.process_words(ref_low, hypo_low)
        comparison.at[idx, "wer"] = output.wer
        comparison.at[idx, "wer_lower"] = output_low.wer
        comparison.at[idx, "mer"] = output.mer
        comparison.at[idx, "mer_lower"] = output_low.mer
        comparison.at[idx, "wil"] = output.wil
        comparison.at[idx, "wil_lower"] = output_low.wil
        comparison.at[idx, "cer"] = jiwer.process_characters(ref, hypo).cer
        comparison.at[idx, "cer_lower"] = jiwer.process_characters(ref_low, hypo_low).cer

        # Calculate BERTScore
        P, R, F1 = bert_score([hypo], [ref], lang="de")
        _bert_score = F1.mean().item()
        comparison.at[idx, "bert_score"] = _bert_score

    # Calculate BLEU Score
    reference_split = [ref.strip().split(" ") for ref in comparison["reference"]]
    hypothesis_split = [hyp.strip().split(" ") for hyp in comparison["hypothesis"]]
    bleu_scores = [sentence_bleu([ref], hyp) for ref, hyp in zip(reference_split, hypothesis_split)]
    comparison["bleu_score"] = bleu_scores

    reference_split = [ref.strip().lower().split(" ") for ref in comparison["reference"]]
    hypothesis_split = [hyp.strip().lower().split(" ") for hyp in comparison["hypothesis"]]
    bleu_scores = [sentence_bleu([ref], hyp) for ref, hyp in zip(reference_split, hypothesis_split)]
    comparison["bleu_score_lower"] = bleu_scores
    return pd.DataFrame(comparison)


def run_comparison(df, col_reference: str, col_hypothesis: str):
    num_samples = len(df)
    df_calc = pd.DataFrame()
    for start_idx in range(0, num_samples, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, num_samples)
        batch = df.iloc[start_idx:end_idx]
        reference = batch[col_reference].to_list()
        hypothesis = batch[col_hypothesis].to_list()

        # df_batch = calculate_scores(reference, hypothesis)
        df_batch = pd.DataFrame({"hypothesis": batch[col_hypothesis].to_list(), "reference": batch[col_reference].to_list()})
        df_batch = calc_scores_lower(df_batch)
        df_calc = pd.concat([df_calc, df_batch], ignore_index=True)

    logger.info(df_calc)
    return df_calc


def visualize_scores(df: pd.DataFrame, name: str) -> None:
    def plot_histogram(col: str) -> None:
        sns.histplot(df[col], bins=10, kde=True)  # Specify 'bins' to control the number of bars
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {col}")
        plt.savefig(f"{ANALYTICS_PATH}/{name}_{col}.png")
        plt.show()

    scores = ["wer", "cer", "bert_score", "bleu_score", "mer", "wil"]
    logger.info(name)
    for score in scores:
        logger.info(f"Avg {score}: {df[score].mean()}")
        plot_histogram(score)


def analyze_text_generation(generate_manual_t5: bool = False):
    df = pd.read_csv(f"{ANALYTICS_PATH}/select_entries.csv", sep=";", encoding="utf-8")
    if generate_manual_t5:
        df["ch_text_t5_manual"] = None
        df = add_ch_generated_sentences_from_ground_truth(df)
        df.to_csv(f"{ANALYTICS_PATH}/select_entries.csv", sep=";", index=False, encoding="utf-8")

    # compare german sentences
    # df_ger = run_comparison(df, "de_text_manual", "de_text_whisper")
    # df_ger.to_csv(f"{ANALYTICS_PATH}/german_text_comparison.csv", index=False, encoding="utf-8")
    # visualize_scores(df_ger, "German vs. German")

    # compare Swiss German sentences (not inferred from manual)
    df_ch = run_comparison(df, "ch_text_manual", "ch_text_t5")
    df_ch.to_csv(f"{ANALYTICS_PATH}/swiss_text_comparison.csv", index=False, encoding="utf-8")
    # visualize_scores(df_ch, "Swiss vs. Swiss")

    # compare Swiss German sentences (inferred from manual)
    df_ch_man = run_comparison(df, "ch_text_manual", "ch_text_t5_manual")
    df_ch_man.to_csv(f"{ANALYTICS_PATH}/swiss_text_comparison_manual.csv", index=False, encoding="utf-8")
    # visualize_scores(df_ch_man, "Swiss vs. Swiss (Manual inference)")


def analyze_xtts_generation():
    df_xtts = pd.read_csv(f"{ANALYTICS_PATH}/generated_xtts.txt", sep="\t", encoding="utf-8")
    df_xtts.column = ["sample_name", "dialect", "orig_de_text", "gen_de_text", "gen_dialect"]

    df_ger = run_comparison(df_xtts, "orig_de_text", "gen_de_text")
    visualize_scores(df_ger, "Swiss vs. Swiss")

    confusion_matrix = pd.crosstab(df_xtts["dialect"], df_xtts["gen_dialect"])

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)

    # Add labels and title
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Hypothesis (Predicted)')
    plt.ylabel('Reference (Actual)')
    plt.show()


if __name__ == '__main__':
    analyze_text_generation(False)

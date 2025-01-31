# swiss-german-0-shot-va-tts
Repository for my master thesis about TTS for Swiss German



# Model Names
The names of the models in the Thesis were given later than when the evaluation was generally performend. As such there
may unknown model names in the code. Generally follow these:

| Original Name | Evaluation Name  | Training regiment                                             |
|---------------|------------------|---------------------------------------------------------------|
| 7_5           | SRF+STT4SG-Mixed | Mixed dataset consisting of STT4SG-350 and SRF-Corpus.        |
| 7_6_SNF       | Mixed+STT4SG-FT  | Finetuning of 7_5 model checkpoint on STT4SG-350 exclusively. |
| 9_2 / 9_2_SNF | STT4SG-Only      | Baseline model, only trained on STT4SG-350                    |


# File structure
## Evaluation
Evaluation folder contains all evaluation results. Table below provides overview.

| Folder                                                    | File                              | Description                                                                                                                             |
|-----------------------------------------------------------|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| [Automated Evaluation](evaluation/Automated%20Evaluation) | \*\_enriched\_*_long.csv          | Contain the results of the Long experiments for each of the models including scores.                                                    |
| [Automated Evaluation](evaluation/Automated%20Evaluation) | gpt_long_sentences.csv            | Contains the sentences used for the GPT-Long evaluation                                                                                 |
| [Automated Evaluation](evaluation/Automated%20Evaluation) | gpt_random_sentences.csv          | Contains the sentences used for the GPT-Random evaluation                                                                               |
| [Automated Evaluation](evaluation/Automated%20Evaluation) | gpt_random_sentences_complete.csv | Contains the original 900 sentences sourced from ChatGPT                                                                                |
| [Automated Evaluation](evaluation/Automated%20Evaluation) | selected_condition_samples.csv    | Lists the used conditioning samples for inference during zero-shot voice adaptation                                                     |
| [Automated Evaluation](evaluation/Automated%20Evaluation) | snf_long_sentences.csv            | Contains the sentences used for the SNF-Long evaluation                                                                                 |
| [Automated Evaluation](evaluation/Automated%20Evaluation) | SNF_Test_Sentences.xlsx           | Contains the sentences used for the SNF-Short evaluation                                                                                |
| [Automated Evaluation](evaluation/Automated%20Evaluation) | speaker_quality_distro.csv        | Distribution of sample quality calculated using dataspeech of the 43 speakers used in evaluation in the complete STT4SG-350 test split. |
| [Automated Evaluation](evaluation/Automated%20Evaluation) | XTTS_Evaluation.xlsx              | Contains all calculated scores of all models in all evaluations of this thesis. Can be considered as single source of truth.            |
| [Data Pipeline](evaluation/Data%20Pipeline)               | Data Pipeline Evaluation.xlsx     | Contains output of Python analytics script from [analytics folder](src/analytics)                                                       |
| [Data Pipeline](evaluation/Data%20Pipeline)               | Text-Generation                   | Contains the manually transcribed sentences in German -> German and Swiss German -> Swiss German                                        | 
| [Human Evaluation](evaluation/Human%20Evaluation)         | HumanEvaluation.xlsx              | Contains all evaluations in a single file, can be considered single source of truth                                                     | 
| [Human Evaluation](evaluation/Human%20Evaluation)         | human_eval_\*.csv                 | Contains reference sentences and speakers used for each evaluation                                                                      | 
| [Human Evaluation](evaluation/Human%20Evaluation)         | save_\*\_group_\*_date.csv        | Contains evaluations of each human evaluator in each subset                                                                             | 
| [Shuffle-Experiment](evaluation/Shuffle-Experiment)       | shuffle_evaluation.csv            | Contains scores of the conditioning shuffle experiment                                                                                  | 

# Audio

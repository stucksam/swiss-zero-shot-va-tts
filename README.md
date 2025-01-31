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


# Audio
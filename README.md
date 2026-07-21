# A Low-Resource Machine Translation: English To Dholuo

This project builds and evaluates a machine translation pipeline between English and Dholuo, with a focus on adapting the system for medical/telemedicine use.
   
## Problem Statement

Dholuo-speaking patients — particularly rural and elderly patients managing chronic conditions like diabetes or arthritis — often struggle to communicate clearly with doctors on telemedicine platforms. Kenya's doctor-to-population ratio is below WHO guidance, making language-inclusive remote care valuable. No existing Kenyan telemedicine platform (e.g. BYON8) currently supports Dholuo, despite the language having millions of speakers.

## Objective: 
To build a multilingual English–Dholuo translation model, adapt it toward clinical phrasing, and produce a pipeline that could plug into a telemedicine chat workflow.
    
## Target audience:
Dholuo-speaking patients, Kenyan telemedicine providers, and healthcare workers needing a communication bridge during remote consultations.

## Success criteria:
BLEU and chrF score improvement over an untuned pretrained baseline, evaluated on a held-out general-domain test split and a separate medical-phrase test set.

## Dataset

Source: 
* `luo_eng.csv` (English–Dholuo parallel corpus, originally from a Kaggle dataset)
* 136,625 sentence pairs, 2 columns (`eng`, `luo`), no missing values, 11 duplicate rows (dropped)
* English sentences average ~18.3 words (~90 characters); Dholuo sentences average ~19.1 words (~89 characters) — the two languages are closely length-aligned

## Project Structure / Workflow

1. **Business Understanding** — problem framing, target audience, success 

2. **Data Understanding** — loading and inspecting the parallel corpus

3. **Exploratory Data Analysis** 
* Missing values and duplicates check
* Sentence-length outlier detection (IQR, boxplots)
* Class/length balance checks
* Keyword frequency bar charts and word clouds (English vs. Dholuo)
* Zipf's law plot (validates the corpus as natural, well-aligned language data)
* Sentence-length correlation scatter plot (English vs. Dholuo word counts)
* Vocabulary growth / type-token ratio curve (shows Dholuo's greater morphological richness)
* Punctuation and special-character frequency comparison (notably the Dholuo apostrophe, which is phonemic rather than auxiliary)
   
4. **Data Preprocessing**
* Custom tokenization/cleaning function (markdown/bracket stripping, punctuation-aware)
* Deliberately **no stemming, lemmatization, or stopword removal** — grammatical words (e.g. \"not\") and verb tense endings are preserved because removing them can invert the meaning of a clinical instruction (e.g. \"Do not take this pill\" → \"Take pill\")
    
5. **Modelling** — train/test split, TF-IDF vectorization, and four progressively more sophisticated translation approaches:

1. **Baseline:** TF-IDF + Nearest Neighbors (retrieval-based)

2. **Seq2Seq (LSTM encoder–decoder)**

3. **Seq2Seq + Attention**

4. **Transformer** (trained from scratch)

5. **NLLB-200 + LoRA** (fine-tuned pretrained multilingual model, hyperparameter-tuned)

6. **Model Evaluation** — BLEU, chrF, validation loss, training time, and parameter count compared across all models

7. **Model Explainability**

* Cross-attention matrix visualization
* Saliency / gradient-based feature attribution (Inseq)
* Integrated Gradients feature attribution
* Side-by-side comparison scorecard of the explainability methods

8. **Final Insights, Recommendations, and Conclusion**

## Key Results

| Model | BLEU | chrF | Val Loss | Training Time |
|---|---|---|---|---|
| TF-IDF + 1-NN (baseline) | 24.50 | – | – | – |
| Seq2Seq (LSTM) | – | – | ~5.9–6.0 | – |
| Seq2Seq + Attention | – | 33.40 | ~5.56–5.74 | 133.5 min |
| Transformer (from scratch) | 9.38 | – | 3.83 | 185.0 min |
| NLLB-200 + LoRA | 34.49 | 50.76 | 1.17 | 62.83 min |

* **NLLB-200 + LoRA is the clear winner** — highest BLEU/chrF, lowest validation loss, and nearly 3x faster to train than the from-scratch Transformer, demonstrating that transfer learning from a pretrained multilingual model is far more effective than training low-resource NMT architectures from scratch.
    
## Main Takeaways

* Dataset quality (size and diversity), not model architecture, is the primary bottleneck for translation performance.
* The corpus is clean and well-aligned but linguistically low-resource: limited vocabulary diversity and modest lexical growth are visible in the Zipf and type-token analyses.
* Neural approaches substantially outperform the TF-IDF retrieval baseline; attention and self-attention (Transformer) architectures further improve contextual alignment.
* Fine-tuning a pretrained multilingual model (NLLB-200) with LoRA is the most efficient and effective strategy for this low-resource language pair.
* Explainability analyses (cross-attention, saliency, integrated gradients) confirm the fine-tuned model learns genuine linguistic alignments rather than memorizing examples.
* Preprocessing intentionally skips stemming/stopword removal to avoid corrupting clinically important meaning.

## Recommendations

1. Expand and diversify the Dholuo–English parallel dataset (books, news, healthcare materials, native speakers).
2. Build a dedicated medical/telemedicine-domain corpus (symptoms, diagnoses, prescriptions, patient-doctor dialogue).
3. Continue leveraging transfer learning from multilingual models (e.g. NLLB-200) rather than training from scratch.
4. Add complementary evaluation metrics (ROUGE, METEOR, BERTScore) and human evaluation.
5. Collect real-world test data from native Dholuo speakers.
6. Extend the pipeline with speech-to-text / text-to-speech for a full voice-based telemedicine workflow.
7. Develop Dholuo-specific tokenizers and normalization rules.
8. Periodically retrain as more data becomes available, and open-source datasets/pipelines to support other low-resource African languages.

## Requirements

The notebook was authored to run on **Kaggle with a T4 x2 GPU accelerator** (internet enabled for package installs). Key libraries used:

1. torch
2. transformers
3. peft
4. sentencepiece
5. sacrebleu
6. evaluate
7. inseq
8. pandas
9. numpy
10. scikit-learn
11. nltk
12. matplotlib
13. seaborn
14. wordcloud

## How to Run

1. Open the notebook on Kaggle (or another GPU-enabled environment) with the `luo_eng.csv` dataset attached.

2. Ensure the GPU accelerator (T4 x2 or equivalent) is selected and internet access is enabled for `pip install` cells.

3. Run cells sequentially from top to bottom — later sections (Transformer, NLLB-200 + LoRA) depend on artifacts (vocabularies, model checkpoints) produced earlier.

4. Note: one NLLB-200 + LoRA fine-tuning cell is an earlier draft and is intentionally left commented out to avoid duplicate training runs — only the final \"Cell 1 – Install Required Packages\" NLLB section should be executed.

5. To access translations from deployed model, access via link: https://huggingface.co/spaces/roy123njuguna/Eng_Luo 

## Disclaimer
This repository contains a proof-of-concept pipeline, It is **not** a validated clinical tool.

Given the dataset limitations discussed above, translations — especially of medical instructions — should not be relied upon for real patient care without further validation, a domain-specific dataset, and human review."


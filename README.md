# Layer-Wise Truthfulness Probing in Sub-4B Language Models

Code repository for the paper: *Layer-Wise Probing for Truthfulness Discrimination in Sub-4B Language Models*.

---

## Overview

This repository contains the standalone pipeline notebook for the study:

- **`layerwise-probing_pipeline_end_to_end.ipynb`**: Single self-contained notebook covering dataset pairing, hidden-state extraction, layer-wise linear probing, statistical evaluation, and visualization across five sub-4B models (`Llama-3.2-1B`, `Llama-3.2-3B`, `Qwen2.5-3B`, `Phi-3.5-mini-instruct`, `gemma-2-2b`).

The notebook is pre-rendered with full outputs showing and explaining each stage of the experiment without requiring re-computation.

---

## Notebook Walkthrough & Results

`layerwise-probing_pipeline_end_to_end.ipynb` view step-by-step implementation and reported results:

- **Surface Baselines (Part 2):** Shows length, negation token, and TF-IDF lexical floors for TruthfulQA and HaluEval to establish non-parametric baselines.
- **Probe Training (Part 4 & 5):** Details layer-by-layer sweeps for Logistic Regression and Linear SVM ($C$ cross-validation) along with constant-probe diagnostics.
- **Results & Visualizations (Part 6):** Displays layer-wise ROC-AUC curves, 95% paired cluster bootstrap intervals, inter-layer redundancy matrices, multi-layer ensemble tests, out-of-distribution transfer to HaluEval, and embedded evaluation figures.
- **Paired TF-IDF Comparison (Part 7):** Explains question-by-question agreement tables, exact binomial tests on disagreements, and score rank correlations.

---

## Full Execution

To re-run the pipeline from scratch:

```bash
pip install transformers accelerate sentencepiece huggingface_hub datasets scikit-learn pandas numpy scipy matplotlib
export HF_TOKEN="your_hf_token" 
```

Run `layerwise-probing_pipeline_end_to_end.ipynb` top-to-bottom.

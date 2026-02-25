# Adaptive Phishing Email Detection Using Generative AI and Robust NLP Classifiers

This repository implements the pipeline described in your COMP1682 Final Year Project proposal:

- Phase 1: Classical + Transformer baselines on email text.
- Phase 2: Generative data augmentation using LLMs to create synthetic phishing and hard-negative samples.
- Phase 3: Robustness & interpretability analysis (stylometry, calibration, SHAP).

## Project structure

```
FYP-Phishing-Detector/
├── data/
│   ├── raw/          # Place original datasets here (Nazario, Kaggle, Enron, etc.)
│   ├── processed/    # Cleaned / merged datasets
│   └── synthetic/    # Generated phishing & hard-negative emails
├── src/
│   ├── config.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── classical_models.py
│   ├── transformer_models.py
│   ├── augmentation.py
│   ├── evaluation.py
│   ├── stylometry.py
│   └── utils.py
├── scripts/
│   ├── run_phase1_baselines.py
│   ├── run_phase2_augmentation.py
│   └── run_phase3_robustness.py
├── outputs/
│   ├── metrics/
│   ├── figures/
│   └── logs/
├── notebooks/
├── requirements.txt
└── README.md
```

## Quick start

1. Create and activate a virtual environment (Windows):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place at least one CSV phishing dataset into `data/raw/`, e.g.:

   - `data/raw/kaggle_phishing.csv`
   - `data/raw/enron_labeled.csv`

   Required columns for the loaders in `src/data_loading.py` (you can adapt if needed):

   - `text`  – the email body (or subject + body merged)
   - `label` – 1 for phishing/spam, 0 for legitimate

4. Run Phase 1 baselines (classical + transformers):

   ```bash
   python scripts/run_phase1_baselines.py
   ```

5. Run Phase 2 (synthetic data generation + augmentation experiments):

   ```bash
   python scripts/run_phase2_augmentation.py
   ```

6. Run Phase 3 (robustness + stylometry + interpretability):

   ```bash
   python scripts/run_phase3_robustness.py
   ```

## Notes

- The code provides a **template** consistent with your proposal. You will still need to:
  - Point to your actual dataset file names/columns in `config.py` or `data_loading.py`.
  - Provide an API key / local LLM for generative augmentation (or manually paste synthetic emails).
  - Optionally tune hyperparameters and add more experiments.

- All heavy training (e.g. DistilBERT/RoBERTa) may take time on CPU; using a GPU is recommended if available.

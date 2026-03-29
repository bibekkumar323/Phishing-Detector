# Adaptive Phishing Email Detection Using Generative AI and Robust NLP Classifiers

**Student:** Bibek Kumar (001331932)  
**University:** University of Greenwich  
**Module:** COMP1682 — Final Year Individual Project  
**Supervisor:** Yasmine / Barzinji  

---

## Project Overview

This project investigates the robustness of classical machine learning classifiers for phishing email detection when trained on datasets augmented with AI-generated phishing emails.

**Phase 1** establishes baseline performance of four classifiers (Logistic Regression, Linear SVM, Random Forest, XGBoost) trained on real-world email datasets using TF-IDF feature extraction.

**Phase 2** evaluates whether exposing these models to AI-generated phishing emails during training improves robustness, by testing four augmentation ratios: 0%, 10%, 25%, and 50%.

---

## Project Structure

```
FYP-Phishing-Detector/
│
├── data/
│   ├── raw/
│   │   ├── enron.csv                        # Download from Kaggle (see below)
│   │   └── nazario.csv                      # Download from Kaggle (see below)
│   └── processed/
│       ├── clean_emails.csv                 # Download from Google Drive (see below)
│       └── synthetic_phishing_emails.csv    # Included in repo — 300 AI-generated phishing emails
│
├── notebooks/
│   ├── 01_merge_clean.ipynb                 # Dataset preparation and cleaning
│   ├── 02_phase_1_baseline_model.ipynb      # Phase 1 — baseline model training and evaluation
│   └── 03_phase_2_augmentation.ipynb        # Phase 2 — AI-augmented training experiments
│
├── outputs/
│   ├── models/                              # Saved .pkl model and vectorizer files
│   │   ├── tfidf_vectorizer.pkl
│   │   ├── logistic_regression.pkl
│   │   ├── linear_svm.pkl
│   │   ├── random_forest.pkl
│   │   └── xgboost.pkl
│   ├── metrics/                             # CSV results tables
│   │   ├── phase2_results.csv
│   │   └── phase2_vs_phase1_comparison.csv
│   └── figures/                             # Generated charts and plots
│       ├── phase2_performance_trends.png
│       ├── phase2_heatmaps.png
│       ├── phase2_fp_fn_trends.png
│       └── phase2_vs_phase1.png
│
├── .gitignore                               # Excludes large data files from Git
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

---

## Datasets

### Raw Datasets — Kaggle

The two raw datasets are publicly available on Kaggle:

> 📦 **[Phishing Email Dataset — Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data)**

Download `enron.csv` and `nazario.csv` from the link above and place them in `data/raw/`.

| File | Place in | Description |
|------|----------|-------------|
| `enron.csv` | `data/raw/` | Enron legitimate email corpus (Klimt & Yang, 2004) |
| `nazario.csv` | `data/raw/` | Nazario phishing email corpus (Nazario, 2007) |

### Cleaned Dataset — Google Drive

The pre-processed cleaned dataset is available on Google Drive:

> 📁 **[Google Drive — Cleaned Dataset](https://drive.google.com/drive/folders/1cxy1UFvHFOHys8VM-wPJE4zEKZ4kpbQE?usp=sharing)**

Download `clean_emails.csv` and place it in `data/processed/`.

> **Note:** Downloading `clean_emails.csv` from Drive allows you to **skip notebook 01** entirely and go straight to notebook 02. If you prefer to generate it yourself, run notebook 01 after placing the raw files in `data/raw/`.

> **Note:** `synthetic_phishing_emails.csv` is already included in this repository under `data/processed/` and does not need to be downloaded separately.

### Dataset Summary

| Dataset | Emails | Labels |
|---------|--------|--------|
| Enron corpus | 29,767 | 15,791 legitimate / 13,976 phishing |
| Nazario corpus | 1,565 | 1,565 phishing |
| **Final cleaned dataset** | **29,833** | 15,447 phishing / 14,386 legitimate |
| Synthetic phishing emails | 300 | Phishing (1) — 6 attack categories |

**Train/test split:** 80/20 stratified — 23,866 train / 5,967 test

### Synthetic Email Categories
- Account verification and suspicious login alerts
- Bank and payment fraud
- Parcel and delivery scams
- MFA and security reset attacks
- University scholarship and student finance
- Prize and lottery scams

---

## Models

| Model | Type | Key Parameters |
|-------|------|----------------|
| Logistic Regression | Linear | `max_iter=2000`, `random_state=42` |
| Linear SVM | Linear | `max_iter=2000`, `random_state=42` |
| Random Forest | Ensemble | `n_estimators=100`, `random_state=42`, `class_weight=balanced` |
| XGBoost | Gradient Boosting | `n_estimators=100`, `learning_rate=0.1`, `max_depth=6`, `scale_pos_weight` |

**Feature extraction:** TF-IDF — `max_features=5000`, `ngram_range=(1,2)`, `stop_words=english`

---

## Phase 1 Results (Baseline)

| Model | Accuracy | Phishing Recall | Macro F1 |
|-------|----------|-----------------|----------|
| Logistic Regression | 98.39% | 0.99 | 0.98 |
| Linear SVM | **98.49%** | 0.99 | 0.98 |
| Random Forest | 98.06% | 0.98 | 0.98 |
| XGBoost | 96.18% | 0.99 | 0.96 |

---

## Phase 2 Experimental Design

The test set is **fixed** (real emails only) across all experiments. Only the training set is augmented.

| Configuration | Real Phishing | Synthetic Phishing | Legitimate |
|---------------|---------------|--------------------|------------|
| 0% (baseline) | 12,357 | 0 | 11,509 |
| 10% augmentation | 11,122 | 1,235 | 11,509 |
| 25% augmentation | 9,268 | 3,089 | 11,509 |
| 50% augmentation | 6,178 | 6,178 | 11,509 |

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/bibekkumar323/Phishing-Detector.git
cd FYP-Phishing-Detector
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get the datasets

**Option A — Skip notebook 01 (recommended):**
1. Download `clean_emails.csv` from [Google Drive](https://drive.google.com/drive/folders/1cxy1UFvHFOHys8VM-wPJE4zEKZ4kpbQE?usp=sharing)
2. Place it in `data/processed/`
3. Go straight to notebook 02

**Option B — Run from scratch:**
1. Download `enron.csv` and `nazario.csv` from [Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data)
2. Place both in `data/raw/`
3. Run notebook 01 first

---

## Running the Notebooks

Run the notebooks **in order**:

### Step 1 — Data Preparation *(skip if using Drive)*
```
notebooks/01_merge_clean.ipynb
```
Merges Enron and Nazario datasets, cleans text, removes duplicates, saves `clean_emails.csv`.

### Step 2 — Phase 1 Baseline Models
```
notebooks/02_phase_1_baseline_model.ipynb
```
Trains and evaluates all four models. Saves `.pkl` files to `outputs/models/`.

### Step 3 — Phase 2 Augmentation Experiments
```
notebooks/03_phase_2_augmentation.ipynb
```
Runs 16 training configurations (4 models × 4 ratios). Saves figures and metrics to `outputs/`.

> **Always use Kernel → Restart & Run All** to ensure cells run in the correct order.

---

## Key Findings

- Linear SVM achieved the highest Phase 1 accuracy (98.49%) and remained the most stable model across all augmentation levels
- All 16 Phase 2 configurations maintained phishing recall ≥ 0.97, satisfying the Recall ≥ 0.95 project threshold
- XGBoost showed the most notable response to augmentation, improving from 96.18% (Phase 1) to 97.70% (Phase 2 at 50%) — an increase of +1.52 percentage points
- Phase 2 results suggest AI-generated phishing emails can be safely incorporated into training pipelines without degrading detection performance

---

## References

- Klimt, B. and Yang, Y. (2004) 'The Enron corpus: A new dataset for email classification research'
- Nazario, J. (2007) Phishing Corpus
- Joachims, T. (1998) 'Text categorization with Support Vector Machines'
- Chen, T. and Guestrin, C. (2016) 'XGBoost: A scalable tree boosting system'
- Breiman, L. (2001) 'Random Forests', Machine Learning, 45(1), pp. 5–32

---

## License

This project is submitted for academic assessment at the University of Greenwich. All datasets used are publicly available for research purposes.

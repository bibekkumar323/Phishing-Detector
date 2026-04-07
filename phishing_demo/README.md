# PhishGuard — Phishing Email Detection Demo
# Bibek Kumar | 001331932 | University of Greenwich

## Folder Structure
```
phishing_demo/
├── app.py                  # Flask backend
├── templates/
│   └── index.html          # Frontend
├── models/                 # Put your .pkl files here
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_regression.pkl
│   ├── linear_svm.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install flask scikit-learn xgboost joblib numpy
```

### 2. Copy your saved model files
Copy your .pkl files from your notebooks output folder into the `models/` folder.

Make sure the filenames match exactly:
- tfidf_vectorizer.pkl
- logistic_regression.pkl
- linear_svm.pkl        ← your notebook saves as "linear_svm.pkl" or "svm.pkl" — rename if needed
- random_forest.pkl
- xgboost.pkl

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
Go to: http://127.0.0.1:5000

## Features
- Paste any email text and click Analyse Email
- Shows Phishing / Legitimate verdict from all 4 models
- Overall majority vote verdict
- Confidence score per model
- Top trigger words highlighted in the email text
- Try sample phishing and legitimate emails with the sample buttons
- Ctrl+Enter keyboard shortcut to analyse

## Notes for Report
- Screenshots of the demo working go in Chapter 5 (Results)
- Architecture diagram: User → Flask → TF-IDF Vectorizer → 4 Models → JSON Response → Frontend
- The word highlighting uses model coefficients × TF-IDF weights (no SHAP needed)
  - For LR and SVM: coef_ × tfidf weight
  - For RF and XGBoost: feature_importances_ × tfidf weight

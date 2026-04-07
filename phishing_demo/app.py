from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import re
import os

app = Flask(__name__)

# ── Load models and vectorizer ──────────────────────────────────────────────
# Update these paths to where your .pkl files are saved
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
models = {
    "Logistic Regression": joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl")),
    "Linear SVM":          joblib.load(os.path.join(MODELS_DIR, "linear_svm.pkl")),
    "Random Forest":       joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl")),
    "XGBoost":             joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl")),
}

# ── Text cleaning (same as training pipeline) ────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Get top triggering words ─────────────────────────────────────────────────
def get_top_words(model_name, model, vec, text_vec, top_n=10):
    feature_names = np.array(vec.get_feature_names_out())

    try:
        if model_name in ("Logistic Regression", "Linear SVM"):
            # Use model coefficients × TF-IDF weight
            coef = model.coef_[0] if hasattr(model, "coef_") else None
            if coef is None:
                return []
            scores = np.array(text_vec.todense()).flatten() * coef
            top_idx = np.argsort(scores)[-top_n:][::-1]
            top_idx = [i for i in top_idx if scores[i] > 0]
            return [{"word": feature_names[i], "score": float(scores[i])} for i in top_idx]

        elif model_name == "Random Forest":
            importances = model.feature_importances_
            tfidf_vals  = np.array(text_vec.todense()).flatten()
            scores      = tfidf_vals * importances
            top_idx     = np.argsort(scores)[-top_n:][::-1]
            top_idx     = [i for i in top_idx if scores[i] > 0]
            return [{"word": feature_names[i], "score": float(scores[i])} for i in top_idx]

        elif model_name == "XGBoost":
            importances = model.feature_importances_
            tfidf_vals  = np.array(text_vec.todense()).flatten()
            scores      = tfidf_vals * importances
            top_idx     = np.argsort(scores)[-top_n:][::-1]
            top_idx     = [i for i in top_idx if scores[i] > 0]
            return [{"word": feature_names[i], "score": float(scores[i])} for i in top_idx]

    except Exception as e:
        print(f"Word extraction error for {model_name}: {e}")

    return []

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data     = request.get_json()
    raw_text = data.get("text", "").strip()

    if not raw_text:
        return jsonify({"error": "No text provided"}), 400

    cleaned  = clean_text(raw_text)
    vec_text = vectorizer.transform([cleaned])

    results = []
    for name, model in models.items():
        prediction = int(model.predict(vec_text)[0])  # 1=phishing, 0=legit

        # Confidence / probability
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec_text)[0]
            confidence = float(proba[prediction]) * 100
        elif hasattr(model, "decision_function"):
            score = model.decision_function(vec_text)[0]
            confidence = min(99.9, max(50.1, 50 + abs(float(score)) * 10))
        else:
            confidence = None

        top_words = get_top_words(name, model, vectorizer, vec_text)

        results.append({
            "model":      name,
            "prediction": "Phishing" if prediction == 1 else "Legitimate",
            "is_phishing": bool(prediction == 1),
            "confidence": round(confidence, 1) if confidence else None,
            "top_words":  top_words,
        })

    # Majority vote
    phishing_votes = sum(1 for r in results if r["is_phishing"])
    overall = "Phishing" if phishing_votes >= 2 else "Legitimate"

    return jsonify({"results": results, "overall": overall, "votes": phishing_votes})

if __name__ == "__main__":
    app.run(debug=True)

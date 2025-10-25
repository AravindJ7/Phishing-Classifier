# app.py
from flask import Flask, render_template, request, jsonify
import os
import re
import joblib
from pathlib import Path

app = Flask(__name__)

MODEL_PATH = Path("phishing_model.joblib")
VECT_PATH = Path("tfidf_vectorizer.joblib")

model = None
vectorizer = None

# Try loading model/vectorizer if present
if MODEL_PATH.exists() and VECT_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECT_PATH)
        print("Loaded model and vectorizer.")
    except Exception as e:
        print("Failed to load model/vectorizer:", e)
        model = None
        vectorizer = None
else:
    print("Model/vectorizer not found — running in rule-based fallback mode.")

# --------- In-memory history storage ---------
recent_history = []
MAX_HISTORY = 20  # maximum number of recent emails to keep

# --------- Heuristics / rule-based fallback ---------
URGENCY_WORDS = [
    "urgent", "immediate", "right now", "asap", "eod", "end of day",
    "expires", "payment failed", "process now", "wire transfer", "transfer",
    "verify your account", "update your payment", "password reset", "reset your password"
]

SUSPICIOUS_DOMAINS = [
    ".ru", ".cn", ".tk", ".xyz", ".top", ".test"
]

PHONE_RE = re.compile(r"\+?\d[\d\s\-\(\)]{6,}\d")
URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")


def extract_features_textual(email_text: str) -> dict:
    text = email_text.lower()
    urls = URL_RE.findall(email_text)
    emails = EMAIL_RE.findall(email_text)
    phones = PHONE_RE.findall(email_text)
    urgency_count = sum(1 for w in URGENCY_WORDS if w in text)
    has_attachment_phrase = bool(re.search(r"\battach(ed|ment)|attachment:|\bpdf\b|\bdocx?\b|\bxlsx?\b", text))
    replyto_mismatch = False
    from_match = re.search(r"^from:\s*(.+)$", email_text, re.IGNORECASE | re.MULTILINE)
    reply_match = re.search(r"^reply-?to:\s*(.+)$", email_text, re.IGNORECASE | re.MULTILINE)
    if from_match and reply_match:
        from_addr = EMAIL_RE.search(from_match.group(1) or "")
        reply_addr = EMAIL_RE.search(reply_match.group(1) or "")
        if from_addr and reply_addr and from_addr.group(0).split("@")[-1] != reply_addr.group(0).split("@")[-1]:
            replyto_mismatch = True

    suspicious_tld = any(domain in email_text.lower() for domain in SUSPICIOUS_DOMAINS)

    return {
        "num_urls": len(urls),
        "num_emails": len(emails),
        "num_phones": len(phones),
        "urgency_count": urgency_count,
        "has_attachment_phrase": has_attachment_phrase,
        "replyto_mismatch": replyto_mismatch,
        "suspicious_tld": suspicious_tld,
        "raw": email_text
    }


def rule_based_predict(text: str):
    f = extract_features_textual(text)
    score = 0.0

    # heuristics weights (simple)
    score += f["num_urls"] * 0.20
    score += min(f["urgency_count"], 3) * 0.15
    score += 0.3 if f["replyto_mismatch"] else 0.0
    score += 0.2 if f["has_attachment_phrase"] else 0.0
    score += 0.25 if f["num_phones"] > 0 else 0.0
    score += 0.25 if f["suspicious_tld"] else 0.0
    score += 0.1 if f["num_emails"] > 1 else 0.0

    # clamp
    score = max(0.0, min(1.0, score))

    label = "PHISHING" if score >= 0.4 else "SAFE"
    return {
        "label": label,
        "score": int(score * 100)
    }


@app.route("/")
def index():
    # Pass recent history (latest first)
    return render_template("index.html", history=recent_history[::-1])


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"ok": False, "error": "No text provided."}), 400

    # Use model if available
    if model is not None and vectorizer is not None:
        try:
            X = vectorizer.transform([text])
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                classes = model.classes_.tolist() if hasattr(model, "classes_") else None
                phishing_index = None
                if classes:
                    try:
                        phishing_index = classes.index("phish")
                    except ValueError:
                        try:
                            phishing_index = classes.index("PHISHING")
                        except ValueError:
                            phishing_index = 1 if len(proba) > 1 else 0
                else:
                    phishing_index = 1 if len(proba) > 1 else 0
                phishing_score = int(round(proba[phishing_index] * 100))
                label = "PHISHING" if phishing_score >= 50 else "SAFE"
            else:
                pred = model.predict(X)[0]
                label = "PHISHING" if str(pred).lower() in ("phish", "phishing", "1") else "SAFE"
                phishing_score = None
        except Exception as e:
            print("Model prediction error:", e)
            r = rule_based_predict(text)
            label = r["label"]
            phishing_score = r["score"]
    else:
        r = rule_based_predict(text)
        label = r["label"]
        phishing_score = r["score"]

    # Add to in-memory history
    recent_history.append({
        "label": label,
        "score": phishing_score,
        "text": text
    })
    # Trim to max length
    if len(recent_history) > MAX_HISTORY:
        recent_history.pop(0)

    return jsonify({"ok": True, "label": label, "score": phishing_score})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

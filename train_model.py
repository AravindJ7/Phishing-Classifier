# train_model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

SYNTHETIC = [
    # phishing (label 'phish')
    ("From: billing@vendor-payments.example.test\nSubject: Invoice #INV-9012 — Payment Failed\nDear Accounts Payable, update your payment details here: https://vendor-payments.example.test/update?ref=INV-9012", "phish"),
    ("From: \"Rahul Mehta (CFO)\" <rahul.mehta@company-payments.example.test>\nReply-To: rahul.mehta@gmail-example.test\nSubject: Immediate: Wire transfer needed — today\nPlease process an urgent wire transfer of INR 50,00,000 RIGHT NOW", "phish"),
    ("From: security@bank.example.test\nSubject: Account verification required\nClick https://example.test/verify to avoid suspension", "phish"),
    # safe (label 'safe')
    ("From: priya.kumar@evonics.com\nSubject: Team lunch on Friday\nHi team, reply with preferences.", "safe"),
    ("From: hr@evonics.com\nSubject: Payroll processed — payslip attached\nHello Aravind, your payslip is attached.", "safe"),
    ("From: maria@evonics.com\nSubject: Project sync — weekly update\nPlease find attached the weekly report and agenda for tomorrow's meeting.", "safe"),
]

texts = [t for t, _ in SYNTHETIC]
labels = [l for _, l in SYNTHETIC]

vect = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
X = vect.fit_transform(texts)

clf = LogisticRegression(solver="liblinear")
clf.fit(X, labels)

joblib.dump(vect, "tfidf_vectorizer.joblib")
joblib.dump(clf, "phishing_model.joblib")
print("Saved tfidf_vectorizer.joblib and phishing_model.joblib")

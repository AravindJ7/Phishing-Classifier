import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ssl
import sys  # <-- Import the sys module

# --- 1. SETUP: Download NLTK data (stopwords, lemmatizer) ---
# This block handles potential download errors
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# --- 2. DEFINE THE PREPROCESSING FUNCTION (Must be identical to training) ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Cleans and prepares a single email text."""
    if not isinstance(text, str):
        return ""
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# --- 3. LOAD THE SAVED MODEL AND VECTORIZER ---
print("Loading model and vectorizer...")
try:
    model = joblib.load('phishing_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: Model or vectorizer file not found.")
    print("Please make sure 'phishing_model.joblib' and 'tfidf_vectorizer.joblib' are in this directory.")
    exit()

# --- 4. PREDICT ON NEW EMAIL FROM USER INPUT ---

print("\n--- Paste your email text below ---")
print("When you are finished, press Ctrl+D (Linux/Mac) or Ctrl+Z then Enter (Windows).")

# Read multi-line input from the user
new_email_text = sys.stdin.read()
# -------------------------------------------------

print("\n--- Processing Email... ---")

# 1. Clean the new email
cleaned_email = preprocess_text(new_email_text)

# 2. Transform it using the *loaded* vectorizer
vectorized_email = vectorizer.transform([cleaned_email])

# 3. Predict using the *loaded* model
prediction = model.predict(vectorized_email)

# 4. Print the result
print("\n--- PREDICTION ---")
if prediction[0] == 1:
    print("Result: 🔴 PHISHING (1)")
else:
    print("Result: 🟢 SAFE (0)")

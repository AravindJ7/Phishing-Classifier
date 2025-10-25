# 🧠 Phishing Email Detection using Machine Learning

A web-based Machine Learning application built with **Flask** that detects whether an email is **phishing** or **safe** based on its textual content and embedded patterns.

---

## 🚀 Project Overview

Phishing is one of the most common forms of cyber attack, where attackers trick users into revealing sensitive information such as passwords or financial data.  
This project uses **Natural Language Processing (NLP)** and **Machine Learning** techniques to automatically classify emails as **phishing** or **legitimate**.

The model has been trained using a TF-IDF vectorizer and a supervised learning algorithm on a labeled dataset of safe and phishing emails.

---

## 🧩 Features

- 🧾 Classifies emails as **Phishing** or **Safe**
- ⚙️ Machine Learning model trained using **TF-IDF vectorization**
- 🧠 Uses algorithms like **Logistic Regression / SVM / Random Forest**
- 🌐 Flask-based web interface for user interaction
- 📊 Dataset-driven — includes both phishing and safe email samples
- 💾 Model saved using `joblib` for fast loading
- 🧰 Easily extendable for real-world email text scanning

---

## 📁 Folder Structure

```
ML_Project/
│
├── app.py                     # Flask web app for user interface
├── train_model.py             # Script to train the ML model
├── predict.py                 # Email text prediction logic
├── phishing_model.joblib      # Trained model file
├── tfidf_vectorizer.joblib    # Vectorizer for transforming text
├── templates/                 # HTML templates (Flask frontend)
├── static/                    # CSS, JS, and image files
├── Phishing_Email.csv         # Dataset with phishing samples
├── safe_emails.csv            # Dataset with safe samples
├── final_model_dataset.csv    # Combined dataset
├── requirements.txt           # Dependencies list
└── README.md                  # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/ML_Project.git
cd ML_Project
```

### 2️⃣ Setup virtual environment
```bash
python3 -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate     # For Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Flask app
```bash
python app.py
```

Then open your browser and go to:  
👉 **http://127.0.0.1:5000/**

---

## 📚 How It Works

1. The email text is input through a web form.  
2. The text is vectorized using the trained TF-IDF model.  
3. The phishing detection model (`phishing_model.joblib`) predicts whether the email is phishing or safe.  
4. The result is displayed instantly on the web interface.

---

## 🧑‍💻 Author

**Aravind J**  
📧 For any queries: [aravindj27012007@gmail.com]  
🌐 GitHub: [https://github.com/AravindJ7]

---

## 🪪 License

This project is licensed under the **MIT License** — feel free to modify and use it for educational purposes.

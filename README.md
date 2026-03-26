# 🧠 Sentiment Analysis Project

## 📌 Overview
This project is a Machine Learning-based Sentiment Analysis system that predicts whether a given text is **Positive** or **Negative**.

The model is trained using NLP techniques and deployed using Streamlit for real-time predictions.

---

## 🚀 Features
- Predict sentiment of user input text
- Clean and simple Streamlit UI
- Real-time prediction
- End-to-end ML pipeline

---

## 🧠 Machine Learning Workflow
1. Data Collection  
2. Data Preprocessing (Cleaning, Tokenization)  
3. Feature Extraction (TF-IDF / CountVectorizer)  
4. Model Training (Logistic Regression / Naive Bayes)  
5. Model Evaluation  
6. Deployment using Streamlit  

---

## 💻 Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Joblib  

---

## 📂 Project Structure
```
sentiment-analysis/
│
├── sentiment_project.py
├── sentiment_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── venv/
```

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/khatri174/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2️⃣ Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the app
```bash
streamlit run sentiment_project.py
```

---

## 📊 Example

Input:
```
I love this product!
```

Output:
```
Positive 😊
```

---

## 🎯 Future Improvements
- Add deep learning models (LSTM, BERT)
- Improve UI design
- Deploy on cloud (Streamlit Cloud / AWS)

---

## 🙌 Author
Akshay Khatri

---

## ⭐ If you like this project
Give it a ⭐ on GitHub!

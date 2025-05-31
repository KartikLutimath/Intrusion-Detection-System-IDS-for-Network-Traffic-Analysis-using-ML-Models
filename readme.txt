# 🚨 Hybrid Machine Learning-Based Intrusion Detection System (IDS) with Real-Time Visualization

An intelligent, real-time Intrusion Detection System (IDS) leveraging a hybrid machine learning model to detect malicious activities in network traffic. This project combines traditional ML classification with modern data visualization for effective, explainable, and actionable threat detection.

---

## 📌 Features

- 🔐 Hybrid ML Models (e.g., Random Forest + KNN/SVM)
- 📊 Real-Time Visualization Dashboard (using Streamlit)
- 🧠 Multi-class Attack Detection (DoS, Probe, R2L, U2R, etc.)
- 💡 Explainable AI (SHAP or LIME support)
- 📈 Performance Metrics: Accuracy, Precision, Recall, F1-score
- 💾 NSL-KDD / CIC-IDS 2017 Dataset Support
- 🌐 Web-based Monitoring Interface
- 🛠️ Modular and Scalable Architecture

---

## 🖼️ Demo

## 🖼️ Demo

![Intrusion Detection System Demo](assets/img1.png)
![Per-Second Counts (Normal vs Intrusions)](assets/img2.png)
![Random Forest Predictions Over Time](assets/img3.png)
![Isolation Forest Anomaly Scores Over Time](assets/img4.png)

---


## 🧠 Machine Learning Pipeline

- **Preprocessing**:
  - Handling missing values
  - Label/One-hot Encoding
  - Feature Scaling
- **Modeling**:
  - Hybrid approach using ensemble techniques
- **Evaluation**:
  - Cross-validation, confusion matrix, ROC curve, classification report

---

## 🧪 Model Performance

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 98.3%  |
| Precision | 97.9%  |
| Recall    | 98.1%  |
| F1-Score  | 98.0%  |

---
#Running This Project

python -m venv venv

venv\Scripts\activate

pip install streamlit joblib datasets matplotlib pandas numpy scikit-learn xgboost seaborn

python intrusion_detection_app.py train

streamlit run intrusion_detection_app.py dashboard

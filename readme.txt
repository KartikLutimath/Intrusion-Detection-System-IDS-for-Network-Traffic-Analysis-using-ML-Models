python -m venv venv

venv\Scripts\activate


pip install streamlit joblib datasets matplotlib pandas numpy scikit-learn xgboost seaborn


python intrusion_detection_app.py train

streamlit run intrusion_detection_app.py dashboard
streamlit run intrusion_detection_app.py dashboard
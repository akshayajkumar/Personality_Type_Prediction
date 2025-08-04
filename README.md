# 🧠 Personality Type Predictor

This project is a Machine Learning-powered web application that predicts an individual's personality type — **Introvert** or **Extrovert** — based on social and behavioral traits. It uses a pre-trained model and is deployed via an interactive Streamlit interface.

---

## 📌 Features

- 🔮 Predicts personality type: **Introvert** or **Extrovert**
- 🖥️ Clean and responsive UI with **Streamlit**
- 🧠 Powered by a trained ML model (`.pkl`)
- 📊 Takes 7 behavioral input features
- 🌄 Uses a background image and styled layout for better user experience

---

## 🗂️ Project Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit application code |
| `main.ipynb` | Notebook used for data preprocessing, training, and model building |
| `personality_datasert.csv` | Original dataset used for training |
| `personality_prediction.pkl` | Pickled model with scaler for prediction |

---

## 📥 Dataset Source
**Extrovert vs Introvert Behavior Data**
Source: Kaggle dataset by Rakesh Kapilavai
Available at: Kaggle - Extrovert vs Introvert Behavior Data(https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data) 

This dataset captures key behavioral and social indicators associated with extroversion and introversion, making it a valuable resource for ML classification tasks.

---

## 🧪 Input Features

| Feature Name              | Description |
|--------------------------|-------------|
| Time Spent Alone         | Daily hours spent alone |
| Stage Fear               | Do you fear public speaking? (Yes/No) |
| Social Event Attendance  | Monthly number of social events attended |
| Going Outside            | Weekly frequency of going out |
| Drained After Socializing| Do you feel tired after social interaction? (Yes/No) |
| Number of Close Friends  | How many close friends you have |
| Social Media Frequency   | How often you post on social media per week |

---

## 🧰 Tech Stack

- **Frontend/UI**: Streamlit
- **Backend/Model**: Scikit-learn
- **Language**: Python
- **Model Deployment**: Local via Streamlit (can be deployed to Streamlit Cloud)
- **Other Tools**: pandas, numpy, pickle

---


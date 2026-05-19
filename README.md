# 🧠 AutoExplainML

A production-ready machine learning explainability tool that converts complex ML model behavior into simple human-readable insights using SHAP.

---

## 🌐 Live Demo
- Backend API: https://autoexplainml.onrender.com
- Frontend App: https://autoexplainml-ui.onrender.com

---

## 🚀 Features
- Upload trained ML models (.pkl)
- Upload datasets (.csv)
- Get automatic feature importance explanations
- SHAP-based model interpretation
- Clean web UI (Streamlit)
- FastAPI backend
- Cloud deployment (Render)

---
## 📌 Use Cases

### 🧠 Data Scientists
Understand which features influence model predictions.

### 🎓 Students
Learn how ML models behave internally.

### 🏢 Business Analysts
Gain transparency in AI-based decisions.

---

## 🧠 Tech Stack
- Python
- FastAPI
- Streamlit
- SHAP
- Pandas, NumPy
- Scikit-learn
- Render (Deployment)

---

## 📦 Architecture
Frontend (Streamlit UI)
        ↓
Backend API (FastAPI)
        ↓
SHAP Explainer Engine
        ↓
ML Model Output

---

## 📸 Screenshots
![alt text](<Screenshot 2026-05-20 at 00.54.12.png>)
![alt text](<Screenshot 2026-05-20 at 00.54.05.png>)
---

## ⚙️ Run Locally

### Backend
uvicorn backend.api:app --reload

### Frontend
streamlit run frontend/app.py

### Backend


## 👨‍💻 Author
Sidhant Narang
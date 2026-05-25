
# 🧠 AutoExplainML

A production-ready Explainable AI framework that transforms complex ML models into **human-readable insights, reports, and automated project outputs**.

It supports:
- Classical Machine Learning
- Deep Learning (optional)
- Computer Vision (optional)
- Automated reporting (PDF + HTML)
- CLI + API + Web UI

---

## 🌐 Live Demo

- 🔗 Backend API: https://autoexplainml.onrender.com  
- 🔗 Frontend UI: https://autoexplainml-ui.onrender.com  

---

# ⚡ Installation

## 🧩 Core Installation
```bash
pip install autoexplainml
````

---

## 🚀 Optional Feature Packs

### 📊 Machine Learning Stack

```bash
pip install autoexplainml[ml]
```

### 👁 Computer Vision Stack

```bash
pip install autoexplainml[cv]
```

### 🧠 Deep Learning (PyTorch)

```bash
pip install autoexplainml[dl_torch]
```

### 🤖 Deep Learning (TensorFlow)

```bash
pip install autoexplainml[dl_tf]
```

### 🔥 Full Feature Pack (Recommended)

```bash
pip install autoexplainml[full]
```

---

# 🚀 CLI Usage

## 📊 Basic Analysis Mode

```bash
autoexplainml model.pkl data.csv --mode analyze
```

## 📦 Full Project Mode (Auto Reports)

```bash
autoexplainml model.pkl data.csv --mode project
```

---

# 📁 Output Generated

When using `--mode project`:

```
autoexplainml_outputs/
 ├── result.json
 ├── report.html
 ├── report.pdf
```

---

# 🚀 Features

## 🧠 Explainability Engine

* SHAP-based feature importance
* LIME explanations
* Permutation analysis

## 📊 Intelligence Layer

* Data quality checks
* Fairness & bias detection
* Model reasoning insights

## 📦 Automation Layer

* Full ML project generation
* Auto PDF + HTML reports
* Structured JSON outputs

## 🌐 Interfaces

* FastAPI backend
* Streamlit frontend
* CLI tool

---

# 🧠 Architecture

```
Frontend (Streamlit)
        ↓
FastAPI Backend
        ↓
AutoExplainML Engine
        ↓
Explainability + Intelligence Layer
        ↓
Reporting System (PDF/HTML)
```

---

# 📌 Use Cases

## 🎓 Students

* Auto-generate ML projects
* Submit ready-made reports
* Learn explainability easily

## 🧠 Data Scientists

* Understand model decisions
* Debug feature impact

## 🏢 Industry

* Model transparency
* AI auditability

---

# ⚙️ Run Locally

## Backend

```bash
uvicorn backend.api:app --reload
```

## Frontend

```bash
streamlit run frontend/app.py
```

---

# 📸 Screenshots

![UI Screenshot](Screenshot%202026-05-20%20at%2000.54.12.png)
![Report Screenshot](Screenshot%202026-05-20%20at%2000.54.05.png)

---

# 🧪 Example Workflow

```python
from autoexplainml.core.pipeline import run_pipeline

result = run_pipeline(model, X)
print(result)
```

---

# 👨‍💻 Author

Sidhant Narang

---

# 🔥 Why This Project Matters

AutoExplainML bridges the gap between:

* Machine Learning models
* Human understanding
* Automated reporting systems

Making AI **transparent, explainable, and usable for everyone**.



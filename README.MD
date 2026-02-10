# 📧 Email Listing (CMP) — MLOps Project

An end‑to‑end Machine Learning Operations (MLOps) system that automatically classifies and organizes emails using a reproducible pipeline built with ZenML and MLflow. The project supports training, tracking, deployment validation, and CI/CD automation via GitHub Actions.

---

## 🚀 Project Goals

This project demonstrates a production‑style ML workflow:

* Data ingestion and preprocessing
* Model training and evaluation
* Experiment tracking (MLflow)
* Pipeline orchestration (ZenML)
* Application interface (Streamlit)
* Continuous Integration & Continuous Deployment (GitHub Actions)

The system enables automated email categorization such as spam filtering, priority sorting, and label grouping.

---

## 🧠 Tech Stack

| Category            | Tools              |
| ------------------- | ------------------ |
| Language            | Python             |
| ML                  | Scikit‑learn / NLP |
| Pipeline            | ZenML              |
| Experiment Tracking | MLflow             |
| API                 | FastAPI            |
| UI                  | Streamlit          |
| CI/CD               | GitHub Actions     |
| Packaging           | Joblib / Pickle    |

---

## 📂 Project Structure

```
Email_Listing (CMP)
│
├── config/                 # Configuration management
├── entity/                 # Data & model schemas
├── pipelines/              # ZenML training & deployment pipelines
├── src/                    # Core ML logic
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   └── model_building.py
│
├── steps/                  # ZenML pipeline steps
├── utils/                  # Helper utilities
│
├── run_pipeline.py         # Training pipeline entry
├── run_deployment.py       # Deployment pipeline entry
├── streamlit_app.py        # User interface
├── agent.py                # Prediction agent
│
└── .github/workflows/      # CI/CD automation
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd Email_Listing
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔧 Initialize MLOps Stack

```bash
zenml init
zenml integration install mlflow -y
```

Start experiment tracking UI:

```bash
mlflow ui
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 🏋️ Run Training Pipeline

```bash
python run_pipeline.py
```

This will:

1. Ingest dataset
2. Clean & preprocess emails
3. Train classifier
4. Evaluate model
5. Log experiment to MLflow
6. Store model artifact

---

## 🤖 Run Prediction App (Streamlit)

```bash
streamlit run streamlit_app.py
```

The UI allows users to paste an email and get predicted category instantly.

---

## 🔁 Deployment Pipeline

```bash
python run_deployment.py
```

Validates saved model and prepares it for serving.

---

## 🔄 CI/CD (GitHub Actions)

On every push:

1. Install dependencies
2. Initialize ZenML
3. Run training pipeline
4. Validate Streamlit app

If any step fails → commit rejected ❌

This ensures model reproducibility and stability.

---

## 🔐 Environment Variables (Best Practice)

Never store credentials in code.

Create `.env` file:

```
API_KEY=your_key_here
```

Access in Python:

```python
import os
api_key = os.getenv("API_KEY")
```

Add to `.gitignore`:

```
.env
credentials.json
```

---

## 🧪 Troubleshooting

**ZenML not initialized**

```
zenml init
```

**MLflow not opening**

```
mlflow ui --port 5000
```

**Pipeline fails in CI**
Make sure dataset path is relative and not local absolute path.

---

## 📈 Future Improvements

* Docker deployment
* Remote artifact store (S3/GCS)
* Scheduled retraining
* Email inbox integration (IMAP API)
* Transformer‑based classifier

---

## 👨‍💻 Author

Arun Raja
B.Tech Information Technology
AI & MLOps Enthusiast

---

## 📜 License

This project is for academic and learning purposes.

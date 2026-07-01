# Agentic Model Risk Validator

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![OpenAI](https://img.shields.io/badge/Agents-OpenAI-black)




<img width="971" height="514" alt="image" src="https://github.com/user-attachments/assets/68738595-972c-47ae-8b22-9a46a70504e7" />









An end-to-end AI model validation app that behaves like a lightweight Model Risk team.

Upload model evidence, run deterministic Python validation tools, and let structured LLM agents turn the results into a clear validation report with findings, challenge questions, risk rating, and final decision.

## Why It Is Different

The LLM never calculates metrics.

Python computes profiling, leakage checks, model metrics, plots, explainability, and fairness summaries. OpenAI agents only interpret the evidence and write structured reviews using Pydantic schemas.

## Features

- FastAPI backend with SQLite job tracking
- Streamlit validation UI
- Dataset profiling, missing values, duplicates, outliers, imbalance, and leakage checks
- Baseline model training with scikit-learn
- Classification and regression metrics
- Feature importance and SHAP support
- Fairness checks when group attributes are available
- Markdown and PDF validation reports
- Docker, tests, sample data, and offline fallback agents

## Quick Start

```bash
cp .env.example .env
docker compose up --build
```

Open:

- App: http://localhost:8501
- API docs: http://localhost:8000/docs

No API key? The project still runs locally with deterministic fallback agent outputs.

## Local Dev

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/generate_sample_data.py
uvicorn app.main:app --reload
```

In another terminal:

```bash
streamlit run frontend/streamlit_app.py
```

## API Example

```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{"target_column":"defaulted","problem_type":"classification"}'
```

## Tests

```bash
pytest
```

## Stack

Python, FastAPI, Streamlit, OpenAI Responses API, Pydantic, pandas, NumPy, scikit-learn, SHAP, matplotlib, SQLite, pytest, Docker.

## Output

The final report includes:

Executive summary, files reviewed, validation scope, data quality review, methodology review, performance review, explainability review, fairness review, governance review, key findings, challenge questions, LOW/MEDIUM/HIGH risk rating, recommendations, and final validation decision.


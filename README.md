# Agentic Model Risk Validator

A full-stack Python project for validating machine learning models like a model risk or model validation team. Python tools calculate metrics and analysis artifacts; OpenAI-backed agents interpret those results and write the validation report.

## What It Does

- Upload a dataset, optional trained model, training code, metrics file, and documentation.
- Run deterministic profiling, leakage checks, baseline training, performance metrics, explainability plots, and fairness summaries.
- Use Pydantic schemas for every agent output.
- Generate a Markdown report and PDF export.
- Track jobs in SQLite under `app/storage`.

The LLM is never asked to calculate metrics. It only receives Python-calculated evidence and writes structured interpretations.

## Run With Docker

```bash
cp .env.example .env
# Add OPENAI_API_KEY to .env if you want live OpenAI agent calls.
docker compose up --build
```

Open:

- FastAPI: http://localhost:8000/docs
- Streamlit: http://localhost:8501

Without an API key, the app still runs with deterministic local fallback agent outputs so the project works end-to-end for development and tests.

## Run Locally

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python scripts/generate_sample_data.py
uvicorn app.main:app --reload
```

In another terminal:

```bash
.\.venv\Scripts\Activate.ps1
streamlit run frontend/streamlit_app.py
```

## API Usage

Create a job and upload files:

```bash
curl -X POST http://localhost:8000/upload ^
  -F "dataset=@sample_data/credit_risk_sample.csv"
```

Run validation:

```bash
curl -X POST http://localhost:8000/validate ^
  -H "Content-Type: application/json" ^
  -d "{\"job_id\":\"JOB_ID\",\"target_column\":\"defaulted\",\"problem_type\":\"classification\"}"
```

Download report:

```bash
curl http://localhost:8000/reports/JOB_ID/download -o validation_report.md
curl "http://localhost:8000/reports/JOB_ID/download?format=pdf" -o validation_report.pdf
```

## Tests

```bash
pytest
```

## Project Structure

```text
app/
  main.py
  api/
  agents/
  schemas/
  services/
  storage/
  tools/
frontend/
  streamlit_app.py
sample_data/
tests/
```

## OpenAI Configuration

Set these in `.env`:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
USE_LLM=true
```

The OpenAI integration uses structured outputs with Pydantic models through the Responses API. `.env` is ignored by git.


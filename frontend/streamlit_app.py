from __future__ import annotations

import os
from io import BytesIO

import pandas as pd
import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


st.set_page_config(page_title="Model Risk Validator", page_icon="MR", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem; max-width: 1280px; }
    div[data-testid="stMetric"] { border: 1px solid #d8dee4; padding: 0.75rem; border-radius: 6px; }
    .stButton button { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Agentic Model Risk Validator")


def upload_payload(uploaded_files: dict[str, object]) -> dict[str, tuple[str, bytes, str]]:
    payload = {}
    for field, uploaded in uploaded_files.items():
        if uploaded is None:
            continue
        payload[field] = (
            uploaded.name,
            uploaded.getvalue(),
            uploaded.type or "application/octet-stream",
        )
    return payload


def post_upload(files: dict[str, tuple[str, bytes, str]]) -> str | None:
    if not files:
        return None
    response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
    response.raise_for_status()
    return response.json()["job_id"]


def run_validation(job_id: str | None, target_column: str | None, problem_type: str) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/validate",
        json={
            "job_id": job_id,
            "target_column": target_column,
            "problem_type": problem_type,
        },
        timeout=900,
    )
    response.raise_for_status()
    return response.json()


left, right = st.columns([0.38, 0.62], gap="large")

with left:
    st.subheader("Inputs")
    dataset = st.file_uploader("Dataset", type=["csv", "xlsx", "xls", "json", "parquet"])
    model = st.file_uploader("Model artifact", type=["pkl", "pickle", "joblib"])
    training_code = st.file_uploader("Training code", type=["py", "ipynb", "txt", "md"])
    metrics_file = st.file_uploader("Metrics file", type=["json", "csv", "txt", "md"])
    documentation = st.file_uploader("Documentation", type=["txt", "md", "pdf", "docx"])

    target_options: list[str] = []
    if dataset is not None:
        try:
            if dataset.name.lower().endswith(".csv"):
                preview = pd.read_csv(BytesIO(dataset.getvalue()), nrows=50)
            elif dataset.name.lower().endswith((".xlsx", ".xls")):
                preview = pd.read_excel(BytesIO(dataset.getvalue()), nrows=50)
            elif dataset.name.lower().endswith(".json"):
                preview = pd.read_json(BytesIO(dataset.getvalue()))
            else:
                preview = pd.DataFrame()
            target_options = list(preview.columns)
            if not preview.empty:
                st.dataframe(preview.head(8), use_container_width=True, height=240)
        except Exception as exc:
            st.warning(f"Preview failed: {exc}")

    target_column = st.selectbox(
        "Target column",
        options=[""] + target_options,
        index=(target_options.index("defaulted") + 1 if "defaulted" in target_options else 0),
    )
    problem_type = st.selectbox("Problem type", options=["auto", "classification", "regression"], index=0)
    run_button = st.button("Run Validation", type="primary")

with right:
    status_box = st.container()
    results_box = st.container()

if run_button:
    with status_box:
        st.subheader("Status")
        with st.spinner("Running validation pipeline and agents..."):
            try:
                files = upload_payload(
                    {
                        "dataset": dataset,
                        "model": model,
                        "training_code": training_code,
                        "metrics_file": metrics_file,
                        "documentation": documentation,
                    }
                )
                job_id = post_upload(files)
                result = run_validation(job_id, target_column or None, problem_type)
                st.session_state["last_result"] = result
                st.session_state["last_job_id"] = result["job_id"]
                st.success(f"Validation complete for job {result['job_id']}")
            except requests.HTTPError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(detail)
            except Exception as exc:
                st.error(str(exc))

result = st.session_state.get("last_result")
job_id = st.session_state.get("last_job_id")

if result:
    with results_box:
        st.subheader("Decision")
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk", result["risk_rating"])
        c2.metric("Decision", result["final_decision"])
        c3.metric("Findings", len(result.get("findings", [])))

        findings = result.get("findings", [])
        if findings:
            st.subheader("Findings")
            st.dataframe(pd.DataFrame(findings), use_container_width=True, hide_index=True)

            risk_counts = (
                pd.DataFrame(findings)
                .groupby(["category", "severity"])
                .size()
                .reset_index(name="count")
                .pivot(index="category", columns="severity", values="count")
                .fillna(0)
                .astype(int)
            )
            st.subheader("Risk Matrix")
            st.dataframe(risk_counts, use_container_width=True)

        if job_id:
            try:
                job_response = requests.get(f"{BACKEND_URL}/jobs/{job_id}", timeout=60)
                job_response.raise_for_status()
                timeline = job_response.json().get("agent_timeline", [])
                if timeline:
                    st.subheader("Agent Timeline")
                    st.dataframe(pd.DataFrame(timeline), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.warning(f"Timeline unavailable: {exc}")

        st.subheader("Report")
        st.download_button(
            "Download Markdown",
            data=result["report_markdown"],
            file_name="validation_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        if job_id:
            pdf_response = requests.get(f"{BACKEND_URL}/reports/{job_id}/download?format=pdf", timeout=120)
            if pdf_response.ok and pdf_response.headers.get("content-type", "").startswith("application/pdf"):
                st.download_button(
                    "Download PDF",
                    data=pdf_response.content,
                    file_name="validation_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        st.markdown(result["report_markdown"])

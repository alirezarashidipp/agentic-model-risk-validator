from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.tools.data_tools import generate_sample_credit_data


if __name__ == "__main__":
    generate_sample_credit_data("sample_data/credit_risk_sample.csv", rows=500)
    print("Generated sample_data/credit_risk_sample.csv")

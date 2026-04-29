from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    from src.config.config import (
        CLEANED_RAW_OUTPUT_FILE,
        INPUT_FILE,
        PROCESSED_DATA_DIR,
    )
except ModuleNotFoundError:
    # Support direct execution: `python src/data/preprocess.py`
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config.config import (
        CLEANED_RAW_OUTPUT_FILE,
        INPUT_FILE,
        PROCESSED_DATA_DIR,
    )


COLUMN_MAP = {
    "Problem (formerly Complaint Type)": "Problem",
    "Problem Detail (formerly Descriptor)": "Problem Detail",
}

KEEP_COLUMNS = [
    "Created Date",
    "Problem",
    "Problem Detail",
    "Incident Zip",
    "Borough",
    "Latitude",
    "Longitude",
]

RAW_INPUT_COLUMNS = set(KEEP_COLUMNS) | set(COLUMN_MAP.keys())


def preprocess_311_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()

    print(f"Original shape: {prepared.shape}")

    prepared.columns = prepared.columns.str.strip()
    prepared = prepared.rename(columns=COLUMN_MAP)

    existing_cols = [col for col in KEEP_COLUMNS if col in prepared.columns]
    prepared = prepared[existing_cols]

    prepared = prepared.dropna(subset=["Problem", "Problem Detail", "Incident Zip", "Latitude", "Longitude"])
    prepared["Incident Zip"] = prepared["Incident Zip"].astype(str).str.extract(r"(\d{5})")
    prepared["Created Date"] = pd.to_datetime(prepared["Created Date"], errors="coerce")
    prepared["Latitude"] = pd.to_numeric(prepared["Latitude"], errors="coerce")
    prepared["Longitude"] = pd.to_numeric(prepared["Longitude"], errors="coerce")
    prepared = prepared.dropna(subset=["Created Date", "Incident Zip", "Latitude", "Longitude"])

    # FILTER DATE RANGE (Mar 2025 -> Jun 2026)
    start_date = pd.to_datetime("2025-03-01")
    end_date = pd.to_datetime("2026-06-30")
    prepared = prepared[
        (prepared["Created Date"] >= start_date) & (prepared["Created Date"] <= end_date)
    ].copy()

    latest_date = prepared["Created Date"].max()
    lambda_decay = 0.01
    prepared["recency_weight"] = np.exp(-lambda_decay * (latest_date - prepared["Created Date"]).dt.days)

    print(f"Prepared shape: {prepared.shape}")
    return prepared


def build_cleaned_raw_output(
    input_file: Path = INPUT_FILE,
    output_file: Path = CLEANED_RAW_OUTPUT_FILE,
) -> Path:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(
        input_file,
        low_memory=False,
        usecols=lambda col: col.strip() in RAW_INPUT_COLUMNS,
    )
    cleaned_df = preprocess_311_dataframe(df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_file, index=False)

    print("Cleaning complete!")
    print(f"Cleaned raw file saved to: {output_file}")
    return output_file


def ensure_cleaned_raw_output_file(
    input_file: Path = INPUT_FILE,
    output_file: Path = CLEANED_RAW_OUTPUT_FILE,
) -> Path:
    if output_file.exists():
        return output_file

    return build_cleaned_raw_output(input_file=input_file, output_file=output_file)


def main() -> None:
    build_cleaned_raw_output()


if __name__ == "__main__":
    main()

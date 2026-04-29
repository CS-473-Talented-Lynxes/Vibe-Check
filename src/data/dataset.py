from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.config.config import (
        CLEANED_RAW_OUTPUT_FILE,
        DATA_END_DATE,
        DATA_START_DATE,
        INPUT_FILE,
        RAW_DATA_FILE_PATTERNS,
    )
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    import sys
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config.config import (
        CLEANED_RAW_OUTPUT_FILE,
        DATA_END_DATE,
        DATA_START_DATE,
        INPUT_FILE,
        RAW_DATA_FILE_PATTERNS,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RAW_FILE = PROJECT_ROOT / "data" / "raw" / "311_example_2.csv"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_TO_CLEAN_COLUMNS = {
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


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared.columns = prepared.columns.str.strip()
    prepared = prepared.rename(columns=RAW_TO_CLEAN_COLUMNS)

    existing_cols = [col for col in KEEP_COLUMNS if col in prepared.columns]
    prepared = prepared[existing_cols].copy()

    prepared = prepared.dropna(subset=["Problem", "Problem Detail", "Incident Zip", "Latitude", "Longitude"])
    prepared["Incident Zip"] = prepared["Incident Zip"].astype(str).str.extract(r"(\d{5})")
    prepared["Created Date"] = pd.to_datetime(prepared["Created Date"], errors="coerce")
    prepared["Latitude"] = pd.to_numeric(prepared["Latitude"], errors="coerce")
    prepared["Longitude"] = pd.to_numeric(prepared["Longitude"], errors="coerce")

    prepared = prepared.dropna(subset=["Created Date", "Incident Zip", "Latitude", "Longitude"])

    start_date = pd.to_datetime(DATA_START_DATE)
    end_date = pd.to_datetime(DATA_END_DATE)
    prepared = prepared[(prepared["Created Date"] >= start_date) & (prepared["Created Date"] <= end_date)].copy()

    latest_date = prepared["Created Date"].max()
    lambda_decay = 0.01
    prepared["recency_weight"] = np.exp(-lambda_decay * (latest_date - prepared["Created Date"]).dt.days)
    prepared["Borough"] = prepared["Borough"].fillna("Unknown")

    return prepared.reset_index(drop=True)


def resolve_raw_data_path() -> Path:
    if INPUT_FILE.exists():
        return INPUT_FILE

    matching_raw_files = []
    seen_paths = set()
    for pattern in RAW_DATA_FILE_PATTERNS:
        for path in RAW_DATA_DIR.glob(pattern):
            if path not in seen_paths:
                matching_raw_files.append(path)
                seen_paths.add(path)
    matching_raw_files = sorted(matching_raw_files, key=lambda path: path.stat().st_mtime, reverse=True)

    if matching_raw_files:
        return matching_raw_files[0]
    if SAMPLE_RAW_FILE.exists():
        return SAMPLE_RAW_FILE
    raise FileNotFoundError(
        "No compatible raw 311 dataset was found. Download the full dataset into data/raw/ or keep the sample file."
    )


def resolve_default_data_path() -> Path:
    if CLEANED_RAW_OUTPUT_FILE.exists():
        return CLEANED_RAW_OUTPUT_FILE
    try:
        return resolve_raw_data_path()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "No compatible 311 dataset was found. Expected a processed file, raw file, or sample raw file."
        ) from exc


def load_prepared_311_data(data_path=None) -> pd.DataFrame:
    path = Path(data_path) if data_path else resolve_default_data_path()
    df = pd.read_csv(path, low_memory=False)

    if {"Problem", "Problem Detail", "Incident Zip", "Latitude", "Longitude", "recency_weight"}.issubset(df.columns):
        prepared = df.copy()
        prepared["Created Date"] = pd.to_datetime(prepared.get("Created Date"), errors="coerce")
        prepared["Latitude"] = pd.to_numeric(prepared["Latitude"], errors="coerce")
        prepared["Longitude"] = pd.to_numeric(prepared["Longitude"], errors="coerce")
        prepared["recency_weight"] = pd.to_numeric(prepared["recency_weight"], errors="coerce").fillna(1.0)
        prepared["Incident Zip"] = prepared["Incident Zip"].astype(str).str.extract(r"(\d{5})")
        prepared["Borough"] = prepared["Borough"].fillna("Unknown")
        prepared = prepared.dropna(subset=["Problem", "Problem Detail", "Incident Zip", "Latitude", "Longitude"])
        return prepared.reset_index(drop=True)

    return _prepare_dataframe(df)

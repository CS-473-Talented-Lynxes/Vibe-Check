from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

INPUT_FILE = RAW_DATA_DIR / "311_Service_Requests_from_20250427_to_20260427.csv"
RAW_DATA_FILE_PATTERNS = [
    "311_Service_Requests_from_20250427_to_20260427.csv",
    "311_Service_Requests_from_2020_to_Present_*.csv",
    "311_Service_Requests*.csv",
]
DATA_START_DATE = "2025-04-27"
DATA_END_DATE = "2026-04-27"
CLEANED_RAW_OUTPUT_FILE = PROCESSED_DATA_DIR / "cleaned_raw_311.csv"
AGGREGATED_OUTPUT_FILE = PROCESSED_DATA_DIR / "cleaned_311.csv"
EMBEDDINGS_OUTPUT_FILE = PROCESSED_DATA_DIR / "complaint_category_embeddings.npz"

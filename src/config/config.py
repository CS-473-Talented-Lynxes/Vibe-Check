from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

INPUT_FILE = RAW_DATA_DIR / "311_Service_Requests_from_2020_to_Present_20260321.csv" # 311_Service_Requests_from_2020_to_Present_20260321.csv
CLEANED_RAW_OUTPUT_FILE = PROCESSED_DATA_DIR / "cleaned_raw_311.csv"
AGGREGATED_OUTPUT_FILE = PROCESSED_DATA_DIR / "cleaned_311.csv"
EMBEDDINGS_OUTPUT_FILE = PROCESSED_DATA_DIR / "complaint_category_embeddings.npz"

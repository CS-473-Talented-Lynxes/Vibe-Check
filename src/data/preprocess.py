import pandas as pd
import numpy as np
from pathlib import Path
import sys

try:
    from src.config.config import (
        AGGREGATED_OUTPUT_FILE,
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
        AGGREGATED_OUTPUT_FILE,
        CLEANED_RAW_OUTPUT_FILE,
        INPUT_FILE,
        PROCESSED_DATA_DIR,
    )

# LOAD DATA

df = pd.read_csv(INPUT_FILE, low_memory=False)


print(f"Original shape: {df.shape}")


# STANDARDIZE COLUMN NAMES

df.columns = df.columns.str.strip()

# Rename columns
column_map = {
    "Problem (formerly Complaint Type)": "Problem",
    "Problem Detail (formerly Descriptor)": "Problem Detail"
}

df = df.rename(columns=column_map)


# FILTER COLUMNS

KEEP_COLUMNS = [
    "Created Date",
    "Problem",
    "Problem Detail",
    "Incident Zip",
    "Borough",
    "Latitude",
    "Longitude"
]

existing_cols = [col for col in KEEP_COLUMNS if col in df.columns]
df = df[existing_cols]


# CLEAN DATA


df = df.dropna(subset=["Problem", "Incident Zip", "Latitude", "Longitude"])

# Clean ZIP codes
df["Incident Zip"] = df["Incident Zip"].astype(str).str.extract(r"(\d{5})")

# Convert date
df["Created Date"] = pd.to_datetime(df["Created Date"], errors="coerce")
df = df.dropna(subset=["Created Date"])


# FILTER DATE RANGE (Mar 2024 → Mar 2026)


start_date = pd.to_datetime("2024-03-01")
end_date = pd.to_datetime("2026-03-31")

df = df[(df["Created Date"] >= start_date) & (df["Created Date"] <= end_date)]


# RECENCY SCORE (IMPORTANT)


# Use most recent date in dataset as reference
latest_date = df["Created Date"].max()

# Exponential decay (you can tune lambda)
lambda_decay = 0.01

df["recency_weight"] = np.exp(-lambda_decay * (latest_date - df["Created Date"]).dt.days)


# CATEGORY GROUPING


# CATEGORY_MAP = {
#     "Rodent": "rats",
#     "Noise": "noise",
#     "Street Condition": "potholes",
#     "Dirty": "cleanliness",
#     "Sanitation": "cleanliness", #more to be added
# }

# def map_category(x):
#     for key in CATEGORY_MAP:
#         if key.lower() in str(x).lower():
#             return CATEGORY_MAP[key]
#     return "other"


# df["Category"] = df["Problem"].apply(map_category)

# 
# # AGGREGATION (WEIGHTED)
# 

# agg_df = (
#     df.groupby(["Incident Zip", "Category"])["recency_weight"]
#     .sum()
#     .reset_index(name="weighted_count")
# )

# pivot_df = agg_df.pivot(index="Incident Zip", columns="Category", values="weighted_count").fillna(0)

# # Normalize
# pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)


# SAVE OUTPUTS


PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(CLEANED_RAW_OUTPUT_FILE, index=False)
# pivot_df.to_csv(AGGREGATED_OUTPUT_FILE)

print("Cleaning complete!")
print(f"Cleaned raw file saved to: {CLEANED_RAW_OUTPUT_FILE}")
# print(f"Aggregated file saved to: {AGGREGATED_OUTPUT_FILE}")

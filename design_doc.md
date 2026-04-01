# Directory Structure

```text
Vibe-Check/
├── data/
│   └── raw/                # Original 311 service request datasets (e.g., 311_example_2.csv)
├── src/                    # Core source code
│   ├── config/             # Configuration files (API keys, project hyperparams)
│   ├── data/               # Data-specific utility and preprocessing scripts
│   ├── clustering.py       # FROM-SCRATCH implementation of k-means/MoG (No sklearn wrappers)
│   ├── embeddings.py       # Logic for Transformers and vector conversion
│   └── main.py             # CLI entry point for backend logic
├── .gitignore              # Specifies files for Git to ignore (e.g., venv, local data)
├── app.py                  # Streamlit web application for the live project demo
├── design_doc.md           # This document: includes tree and labor allocation
├── LICENSE                 # GPL-3.0 open-source license
├── README.md               # Project proposal and environment setup guide
├── requirements.txt        # Working environment dependencies (torch, streamlit, etc.)
└── test_output.txt         # Sample output logs from initial testing
```

# Division of Labor (alphabetical by last name)
- **Zeyue Xu**: Clustering, neighborhood ranking, and evaluation
- **Joseph Zhang**: Frontend / backend integration, app structure
- **Isabella Liu**: Repository organization, README/design document, requirements/setup, and checkpoint submission polishing
- **Yujia Guo**: Data collection and preprocessing of NYC 311 records
- **Jinyu Zheng**: Embedding-based complaint matching and similarity pipeline

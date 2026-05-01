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
- **Zeyue Xu**: Algorithm implementation, clustering logic, neighborhood ranking, and model evaluation
- **Joseph Zhang**: Frontend UI design, Streamlit app structure, and frontend/backend integration
- **Isabella Liu**: Cross-platform setup and testing, README/setup documentation, repository organization, requirements validation, and end-to-end demo verification
- **Yujia Guo**: NYC 311 data collection, data preprocessing, and presentation outline
- **Jinyu Zheng**: Embedding-based complaint matching, similarity pipeline, environment configuration support, and presentation design

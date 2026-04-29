# Vibe Check

Vibe Check is a Streamlit app for NYC apartment hunters who want to evaluate neighborhood conditions before committing to a lease. Instead of asking only about rent, amenities, or commute time, the app answers a user-facing question a renter could plausibly ask:

> Given the urban problems I personally want to avoid, which NYC areas show the strongest or weakest 311 complaint patterns for those concerns?

The app uses NYC 311 Service Requests, semantic matching over complaint categories, and a from-scratch k-means clustering implementation to produce a map plus ranked lists of lower-concern areas and hotspots.

## Data

The project uses the NYC 311 Service Requests dataset. The preprocessing pipeline keeps the columns needed for the recommendation workflow:

```text
Created Date, Problem, Problem Detail, Incident Zip, Borough, Latitude, Longitude, recency_weight
```

Records are filtered to March 2024 through March 2026. The `recency_weight` column gives more weight to recent complaints:

```text
recency_weight = exp(-lambda * delta_days)
lambda = 0.01
```

where `delta_days` is the number of days between the most recent complaint date in the loaded dataset and the complaint's `Created Date`.

## Method

First, the app maps the user's free-form concern text to relevant 311 complaint categories. It embeds complaint category labels with `nomic-ai/nomic-embed-text-v1.5` and ranks categories by a numpy cosine-similarity calculation against the user's query.

Second, the app clusters the latitude/longitude points for the matched complaints. This is the course algorithm implemented from scratch in [src/clustering.py](src/clustering.py). It does not call `sklearn.cluster.KMeans`.

The k-means implementation:

1. Initializes up to `k` unique latitude/longitude centroids with a fixed random seed.
2. Assigns each matched complaint point to the nearest centroid using squared Euclidean distance.
3. Recomputes each centroid as the mean latitude/longitude of the points assigned to that cluster.
4. Repeats assignment and centroid updates until centroid movement is below `1e-5`, assignments stop changing, or 50 iterations have run.
5. Assigns all 311 complaints to the final centroids to compute a local baseline for each cluster.

Third, each cluster is ranked by a normalized severity score:

```text
severity_score = sum(recency_weight_i * similarity_i)
concern_share = severity_score / baseline_score
reliability_factor = log(1 + matched_complaint_count)
normalized_severity = concern_share * reliability_factor
```

`concern_share` measures how much of the local complaint volume matches the user's selected concerns. The `log(1 + matched_complaint_count)` factor keeps absolute complaint count in the score without letting very large neighborhoods dominate linearly. It also reduces the chance that a tiny cluster with only one or two matching complaints ranks too highly just because its share is large.

## Setup

```bash
git clone https://github.com/CS-473-Talented-Lynxes/Vibe-Check.git
cd Vibe-Check
python -m venv venv
```

Activate the environment:

```bash
# macOS/Linux
source venv/bin/activate

# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Run the command-line demo:

```bash
python .\src\main.py
```

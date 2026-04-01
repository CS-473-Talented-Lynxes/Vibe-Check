# Proposal
## Description
The apartment search experience in New York City is already saturated with platforms that optimize for price, amenities, and proximity, constantly telling users which listing offers the best value. But in many cases, the real issue is not the apartment itself, but the neighborhood around it. A unit may look ideal on paper, yet still be a poor fit if the surrounding area is consistently affected by noise, pests, sanitation complaints, or infrastructure problems that a renter personally finds intolerable. This project introduces “Gotham Gauge”,  a neighborhood recommender system that utilizes the NYC 311 Service Requests dataset (2020–Present) to quantify the localized environment of various city sectors. Our application addresses a critical user-facing question: "Given a resident's specific sensitivity to urban stressors—such as noise, pests, or infrastructure failures—which NYC neighborhoods provide the best statistical match for their profile?" This problem is compelling because it replaces broad neighborhood generalizations with a granular "Red Flag" report. Instead of relying on hearsay, users can weight their own tolerances to generate a dual-mode output: a geospatial heat map visualizing recurring local issues—such as specific patterns of heating failures or commercial noise—alongside a ranked "Top Matches" list of neighborhoods. This allows for a data-driven assessment of a street-level "vibe" before a user commits to a lease.

## Methods
Our method is designed to match the structure of this problem: user preferences are text, while neighborhood conditions are large-scale complaint records with location and time. First, we use a pre-trained sentence-transformer embedding model and cosine similarity to map free-form user concerns (for example, noise or pests) to the most relevant NYC 311 complaint categories; this is appropriate because users do not use exactly the same wording as the dataset, so semantic matching is more reliable than keyword matching. Second, we run k-means clustering on the latitude/longitude points of matched complaints to identify concentrated issue areas and produce ranked neighborhood candidates; this is appropriate because the output needs to be geographic and interpretable for decision-making. Third, we apply recency weighting so newer complaints contribute more than older ones; this is appropriate because neighborhood conditions change over time, and users care most about current liveability rather than historical patterns. Together, these steps create a practical pipeline that turns subjective preferences into explainable, data-driven area recommendations.


# Setup Instructions
To run this project locally, follow these steps:
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/CS-473-Talented-Lynxes/Vibe-Check.git](https://github.com/CS-473-Talented-Lynxes/Vibe-Check.git)
   cd Vibe-Check
    ```
2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
    ```
4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
    ```
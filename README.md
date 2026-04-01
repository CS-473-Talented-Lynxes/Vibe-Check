# Proposal
## Description
The apartment search experience in New York City is already saturated with platforms that optimize for price, amenities, and proximity, constantly telling users which listing offers the best value. But in many cases, the real issue is not the apartment itself, but the neighborhood around it. A unit may look ideal on paper, yet still be a poor fit if the surrounding area is consistently affected by noise, pests, sanitation complaints, or infrastructure problems that a renter personally finds intolerable. 

This project introduces “Vibe Check”, a neighborhood recommender system that utilizes the NYC 311 Service Requests dataset to quantify the localized environment of various city sectors. Our application addresses a critical user-facing question: "Given a resident's specific sensitivity to urban stressors—such as noise, pests, or infrastructure failures—which NYC neighborhoods provide the best statistical match for their profile?" This problem is compelling because it replaces broad neighborhood generalizations with a granular "Red Flag" report. 

Instead of relying on hearsay, users can weight their own tolerances to generate a dual-mode output: a geospatial heat map visualizing recurring local issues—such as specific patterns of heating failures or commercial noise—alongside a ranked "Top Matches" list of neighborhoods. This allows for a data-driven assessment of a street-level "vibe" before a user commits to a lease.

## Methods
Our method is designed to match the structure of this problem: user preferences are text, while neighborhood conditions are large-scale complaint records with location and time. 

First, we process the 311 data to clean it down to 8 columns and within a stricter time frame(2024/3 - 2026/3):
```
Created Date,Problem,Problem Detail,Incident Zip,Borough,Latitude,Longitude,recency_weight
```
The first 7 columns is directly outputed from the source dataset, the last column, recency_weight is calculated as follows:

$$
\text{recency\_weight} = e^{-\lambda \cdot \Delta t}\\
$$

$$
\lambda = 0.01\\
\Delta t = t_{\text{Current Date}} - t_{\text{Created Date}}
$$

Second, we use a pre-trained sentence-transformer embedding model (all-MiniLM-L6-v2) and cosine similarity to map free-form user concerns from embedded Problem and Problem Detail (for example, Unsanitary condition - Pests) among the most relevant NYC 311 complaint categories; We believe semantic matching is more reliable than keyword matching. 

Third, we run k-means clustering on the latitude/longitude points of matched complaints to identify concentrated issue areas and produce ranked neighborhood candidates with similarity scores, then we calculate severity scores for each clusters:

$$
\text{Severity} = \sum_{i=1}^{n} (\text{recency\_weight}_i \times \text{similarity}_i)\\
\text{n} = \text{number of complaints in the category}
$$

Afterwards, the output can be further normalized, geographically analyzed and interpreted for decision-making. 

Fourth, we output the result into json format to the front-end webapp. Together, these steps create a practical pipeline that turns subjective preferences into explainable, data-driven area recommendations.


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
4. **Run the app in CLI:**
    ```bash
    python .\src\main.py 
    ```
5. **Run the Streamlit app (in development):**
    ```bash
    streamlit run app.py
    ```
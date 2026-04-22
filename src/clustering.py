import os
import pandas as pd
from sklearn.cluster import KMeans
import warnings

try:
    from src.data.dataset import load_prepared_311_data
except ModuleNotFoundError:
    from data.dataset import load_prepared_311_data

# Suppress kmeans warning about memory leak
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.cluster')

class LocationClusterer:
    def __init__(self, data_path=None):
        self.data_path = data_path
        print("Loading 311 dataset for clustering...")
        self.df = load_prepared_311_data(self.data_path)
        # Ensure lat/lon and recency_weight are numeric
        self.df['Latitude'] = pd.to_numeric(self.df['Latitude'], errors='coerce')
        self.df['Longitude'] = pd.to_numeric(self.df['Longitude'], errors='coerce')
        if 'recency_weight' in self.df.columns:
            self.df['recency_weight'] = pd.to_numeric(self.df['recency_weight'], errors='coerce').fillna(1.0)
        else:
            self.df['recency_weight'] = 1.0
            
        self.df = self.df.dropna(subset=['Latitude', 'Longitude'])

    def _matched_dataframe(self, matched_categories):
        if not matched_categories:
            return pd.DataFrame()

        matches_df = pd.DataFrame(matched_categories).copy()
        if not {'problem', 'detail'}.issubset(matches_df.columns):
            return pd.DataFrame()
        if 'similarity' not in matches_df.columns:
            matches_df['similarity'] = 1.0
        matches_df = matches_df[['problem', 'detail', 'similarity']].copy()
        matches_df = matches_df.rename(columns={
            'problem': 'Problem',
            'detail': 'Problem Detail'
        })
        matches_df['similarity'] = pd.to_numeric(matches_df['similarity'], errors='coerce').fillna(0.0)
        matches_df = (
            matches_df
            .groupby(['Problem', 'Problem Detail'], as_index=False)['similarity']
            .max()
        )

        filtered_df = self.df.merge(matches_df, on=['Problem', 'Problem Detail'], how='inner')
        if len(filtered_df) == 0:
            return filtered_df

        filtered_df['severity_contribution'] = filtered_df['recency_weight'] * filtered_df['similarity']
        return filtered_df

    def cluster_locations(self, matched_categories, k_clusters=15):
        """
        Filter dataset by matching problem categories and cluster locations.
        """
        filtered_df = self._matched_dataframe(matched_categories)
        
        if len(filtered_df) == 0:
            return []
             
        # Adjust K if we have very few points
        actual_k = min(k_clusters, len(filtered_df))
        
        coords = filtered_df[['Latitude', 'Longitude']].values
        
        # Apply K-Means
        kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init='auto')
        filtered_df['Cluster'] = kmeans.fit_predict(coords)
        
        # Aggregate cluster stats
        cluster_stats = []
        for i in range(actual_k):
            cluster_data = filtered_df[filtered_df['Cluster'] == i]
            
            centroid = kmeans.cluster_centers_[i]
            count = len(cluster_data)
            total_weight = cluster_data['severity_contribution'].sum()
            normalized_severity = total_weight / count if count else 0.0
            
            # Get the most common zip code and borough for context
            most_common_zip = cluster_data['Incident Zip'].mode().iloc[0] if not cluster_data['Incident Zip'].mode().empty else "Unknown"
            most_common_borough = cluster_data['Borough'].mode().iloc[0] if not cluster_data['Borough'].mode().empty else "Unknown"
            
            cluster_stats.append({
                'cluster_id': i,
                'center_lat': float(centroid[0]),
                'center_lon': float(centroid[1]),
                'complaint_count': int(count),
                'severity_score': float(total_weight),
                'normalized_severity': float(normalized_severity),
                'primary_zip': most_common_zip,
                'primary_borough': most_common_borough
            })
            
        # Rank clusters by normalized severity so large clusters do not automatically dominate.
        ranked_clusters = sorted(
            cluster_stats,
            key=lambda x: (x['normalized_severity'], x['severity_score']),
            reverse=True,
        )
        return ranked_clusters

    def cluster_extremes(self, matched_categories, k_clusters=15, top_n=5):
        ranked_clusters = self.cluster_locations(matched_categories, k_clusters=k_clusters)
        if not ranked_clusters:
            return {"worst": [], "best": []}

        worst_clusters = ranked_clusters[:top_n]
        best_clusters = list(reversed(ranked_clusters[-top_n:]))
        return {
            "worst": worst_clusters,
            "best": best_clusters,
        }

if __name__ == "__main__":
    # Test script slightly
    clusterer = LocationClusterer()
    # Dummy mock test data
    dummy_categories = [{"problem": "Noise - Residential", "detail": "Loud Music/Party"}]
    print("Testing clustering with dummy categories...")
    results = clusterer.cluster_locations(dummy_categories, k_clusters=5)
    for res in results:
        print(f"Rank. {res['primary_borough']} Zip: {res['primary_zip']} - Score: {res['severity_score']:.2f} (Count: {res['complaint_count']})")

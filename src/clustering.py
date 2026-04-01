import os
import pandas as pd
from sklearn.cluster import KMeans
import warnings

# Suppress kmeans warning about memory leak
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.cluster')

class LocationClusterer:
    def __init__(self, data_path=None):
        self.data_path = data_path if data_path else os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'cleaned_raw_311.csv')
        print("Loading 311 dataset for clustering...")
        self.df = pd.read_csv(self.data_path)
        # Ensure lat/lon and recency_weight are numeric
        self.df['Latitude'] = pd.to_numeric(self.df['Latitude'], errors='coerce')
        self.df['Longitude'] = pd.to_numeric(self.df['Longitude'], errors='coerce')
        if 'recency_weight' in self.df.columns:
            self.df['recency_weight'] = pd.to_numeric(self.df['recency_weight'], errors='coerce').fillna(1.0)
        else:
            self.df['recency_weight'] = 1.0
            
        self.df = self.df.dropna(subset=['Latitude', 'Longitude'])

    def cluster_locations(self, matched_categories, k_clusters=15):
        """
        Filter dataset by matching problem categories and cluster locations.
        """
        if not matched_categories:
            return []

        matches_df = pd.DataFrame(matched_categories).copy()
        if not {'problem', 'detail'}.issubset(matches_df.columns):
            return []
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
            return []

        filtered_df['severity_contribution'] = filtered_df['recency_weight'] * filtered_df['similarity']
             
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
            
            # Get the most common zip code and borough for context
            most_common_zip = cluster_data['Incident Zip'].mode().iloc[0] if not cluster_data['Incident Zip'].mode().empty else "Unknown"
            most_common_borough = cluster_data['Borough'].mode().iloc[0] if not cluster_data['Borough'].mode().empty else "Unknown"
            
            cluster_stats.append({
                'cluster_id': i,
                'center_lat': centroid[0],
                'center_lon': centroid[1],
                'complaint_count': count,
                'severity_score': total_weight,
                'primary_zip': most_common_zip,
                'primary_borough': most_common_borough
            })
            
        # Rank clusters by total severity score
        ranked_clusters = sorted(cluster_stats, key=lambda x: x['severity_score'], reverse=True)
        return ranked_clusters

if __name__ == "__main__":
    # Test script slightly
    clusterer = LocationClusterer()
    # Dummy mock test data
    dummy_categories = [{"problem": "Noise - Residential", "detail": "Loud Music/Party"}]
    print("Testing clustering with dummy categories...")
    results = clusterer.cluster_locations(dummy_categories, k_clusters=5)
    for res in results:
        print(f"Rank. {res['primary_borough']} Zip: {res['primary_zip']} - Score: {res['severity_score']:.2f} (Count: {res['complaint_count']})")

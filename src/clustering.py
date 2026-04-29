import numpy as np
import pandas as pd

try:
    from src.data.dataset import load_prepared_311_data
except ModuleNotFoundError:
    from data.dataset import load_prepared_311_data

RANDOM_SEED = 42
BATCH_SIZE = 10_000
MAX_KMEANS_ITERATIONS = 50
KMEANS_TOLERANCE = 1e-5

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

    def _initialize_centroids(self, coords, k_clusters):
        unique_coords = np.unique(coords, axis=0)
        actual_k = min(max(int(k_clusters), 0), len(unique_coords))
        if actual_k == 0:
            return np.empty((0, coords.shape[1]), dtype=float)

        rng = np.random.default_rng(RANDOM_SEED)
        chosen_positions = rng.choice(len(unique_coords), size=actual_k, replace=False)
        return unique_coords[chosen_positions].astype(float)

    def _assign_to_closest_centroid(self, coords, centroids):
        assignments = np.empty(len(coords), dtype=int)

        for start in range(0, len(coords), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(coords))
            batch = coords[start:stop]
            deltas = batch[:, None, :] - centroids[None, :, :]
            distances = np.sum(deltas * deltas, axis=2)
            assignments[start:stop] = np.argmin(distances, axis=1)

        return assignments

    def _recompute_centroids(self, coords, assignments, previous_centroids):
        centroids = previous_centroids.copy()

        for cluster_id in range(len(previous_centroids)):
            cluster_points = coords[assignments == cluster_id]
            if len(cluster_points) > 0:
                centroids[cluster_id] = cluster_points.mean(axis=0)

        return centroids

    def _calculate_inertia(self, coords, centroids, assignments):
        inertia = 0.0

        for start in range(0, len(coords), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(coords))
            batch = coords[start:stop]
            batch_centroids = centroids[assignments[start:stop]]
            deltas = batch - batch_centroids
            inertia += float(np.sum(deltas * deltas))

        return inertia

    def _fit_kmeans(self, coords, k_clusters, max_iterations=MAX_KMEANS_ITERATIONS, tolerance=KMEANS_TOLERANCE):
        centroids = self._initialize_centroids(coords, k_clusters)
        if len(centroids) == 0:
            return centroids, np.array([], dtype=int), {
                'iterations': 0,
                'converged': True,
                'inertia': 0.0,
            }

        assignments = np.full(len(coords), -1, dtype=int)
        converged = False
        iterations = 0

        for iteration in range(1, max_iterations + 1):
            new_assignments = self._assign_to_closest_centroid(coords, centroids)
            new_centroids = self._recompute_centroids(coords, new_assignments, centroids)
            centroid_shift = np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1)).max()

            assignments_changed = not np.array_equal(new_assignments, assignments)
            centroids = new_centroids
            assignments = new_assignments
            iterations = iteration

            if centroid_shift <= tolerance or not assignments_changed:
                converged = True
                break

        return centroids, assignments, {
            'iterations': iterations,
            'converged': converged,
            'inertia': self._calculate_inertia(coords, centroids, assignments),
        }

    def cluster_locations(self, matched_categories, k_clusters=300):
        """
        Filter dataset by matching problem categories, run k-means from scratch
        on matched latitude/longitude points, then rank each cluster by a
        normalized concern score.
        """
        filtered_df = self._matched_dataframe(matched_categories)
        
        if len(filtered_df) == 0:
            return []

        coords = filtered_df[['Latitude', 'Longitude']].to_numpy(dtype=float)
        centroids, assignments, kmeans_info = self._fit_kmeans(coords, k_clusters)
        if len(centroids) == 0:
            return []

        filtered_df = filtered_df.copy()
        filtered_df['Cluster'] = assignments

        baseline_df = self.df[['Latitude', 'Longitude', 'recency_weight']].copy()
        baseline_coords = baseline_df[['Latitude', 'Longitude']].to_numpy(dtype=float)
        baseline_df['Cluster'] = self._assign_to_closest_centroid(baseline_coords, centroids)
        
        cluster_stats = []
        for i, centroid in enumerate(centroids):
            cluster_data = filtered_df[filtered_df['Cluster'] == i]
            baseline_cluster_data = baseline_df[baseline_df['Cluster'] == i]
            count = len(cluster_data)
            if count == 0:
                continue

            concern_weight = cluster_data['severity_contribution'].sum()
            baseline_weight = baseline_cluster_data['recency_weight'].sum()
            baseline_count = len(baseline_cluster_data)

            concern_share = concern_weight / baseline_weight if baseline_weight else 0.0
            reliability_factor = np.log1p(count)
            normalized_severity = concern_share * reliability_factor

            most_common_zip = cluster_data['Incident Zip'].mode().iloc[0] if not cluster_data['Incident Zip'].mode().empty else "Unknown"
            most_common_borough = cluster_data['Borough'].mode().iloc[0] if not cluster_data['Borough'].mode().empty else "Unknown"

            cluster_stats.append({
                'cluster_id': i,
                'center_lat': float(centroid[0]),
                'center_lon': float(centroid[1]),
                'complaint_count': int(count),
                'baseline_complaint_count': int(baseline_count),
                'baseline_score': float(baseline_weight),
                'severity_score': float(concern_weight),
                'concern_share': float(concern_share),
                'reliability_factor': float(reliability_factor),
                'normalized_severity': float(normalized_severity),
                'primary_zip': most_common_zip,
                'primary_borough': most_common_borough,
                'kmeans_iterations': int(kmeans_info['iterations']),
                'kmeans_converged': bool(kmeans_info['converged']),
                'kmeans_inertia': float(kmeans_info['inertia']),
            })

        ranked_clusters = sorted(
            cluster_stats,
            key=lambda x: (x['normalized_severity'], x['severity_score']),
            reverse=True,
        )
        return ranked_clusters

    def cluster_extremes(self, matched_categories, k_clusters=300, top_n=50):
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
    results = clusterer.cluster_locations(dummy_categories, k_clusters=300)
    for res in results:
        print(f"Rank. {res['primary_borough']} Zip: {res['primary_zip']} - Score: {res['severity_score']:.2f} (Count: {res['complaint_count']})")

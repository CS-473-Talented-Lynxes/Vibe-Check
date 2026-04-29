import numpy as np
import pandas as pd

try:
    from src.data.dataset import load_prepared_311_data
except ModuleNotFoundError:
    from data.dataset import load_prepared_311_data

RANDOM_SEED = 42
BATCH_SIZE = 10_000
KMEANS_DISTANCE_TOLERANCE = 1e-4
KMEANS_MAX_ITERATIONS = 100
SUPPORTED_CLUSTER_METHODS = {"seed", "kmeans"}

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

    def _select_seed_indices(self, coords, k_clusters):
        unique_coords, unique_indices = np.unique(coords, axis=0, return_index=True)
        actual_k = min(k_clusters, len(unique_coords))
        if actual_k == 0:
            return np.array([], dtype=int)

        rng = np.random.default_rng(RANDOM_SEED)
        chosen_positions = rng.choice(len(unique_indices), size=actual_k, replace=False)
        return unique_indices[chosen_positions]

    def _assign_to_closest_seed(self, coords, seed_coords):
        assignments = np.empty(len(coords), dtype=int)
        min_distances = np.empty(len(coords), dtype=float)

        for start in range(0, len(coords), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(coords))
            batch = coords[start:stop]
            deltas = batch[:, None, :] - seed_coords[None, :, :]
            distances = np.sum(deltas * deltas, axis=2)
            closest = np.argmin(distances, axis=1)
            assignments[start:stop] = closest
            min_distances[start:stop] = distances[np.arange(len(batch)), closest]

        return assignments, min_distances

    def _validate_cluster_method(self, method):
        if method not in SUPPORTED_CLUSTER_METHODS:
            supported = ", ".join(sorted(SUPPORTED_CLUSTER_METHODS))
            raise ValueError(f"Unsupported clustering method '{method}'. Expected one of: {supported}.")
        return method

    def _initialize_centers(self, coords, k_clusters):
        seed_indices = self._select_seed_indices(coords, k_clusters)
        if len(seed_indices) == 0:
            return np.empty((0, 2), dtype=float)
        return coords[seed_indices].copy()

    def _cluster_with_seed_assignment(self, coords, baseline_coords, k_clusters):
        centers = self._initialize_centers(coords, k_clusters)
        if len(centers) == 0:
            return np.array([], dtype=int), np.array([], dtype=int), np.empty((0, 2), dtype=float)

        labels, _ = self._assign_to_closest_seed(coords, centers)
        baseline_labels, _ = self._assign_to_closest_seed(baseline_coords, centers)
        return labels, baseline_labels, centers

    def _update_kmeans_centers(self, coords, labels, distances, centers):
        updated_centers = centers.copy()
        cluster_count = len(centers)

        counts = np.bincount(labels, minlength=cluster_count)
        lat_sums = np.bincount(labels, weights=coords[:, 0], minlength=cluster_count)
        lon_sums = np.bincount(labels, weights=coords[:, 1], minlength=cluster_count)

        non_empty = counts > 0
        updated_centers[non_empty, 0] = lat_sums[non_empty] / counts[non_empty]
        updated_centers[non_empty, 1] = lon_sums[non_empty] / counts[non_empty]

        empty_clusters = np.flatnonzero(~non_empty)
        if len(empty_clusters) > 0:
            farthest_indices = np.argsort(distances)[::-1]
            replacement_cursor = 0
            used_indices = set()

            for cluster_id in empty_clusters:
                while replacement_cursor < len(farthest_indices) and farthest_indices[replacement_cursor] in used_indices:
                    replacement_cursor += 1
                if replacement_cursor >= len(farthest_indices):
                    break
                point_index = int(farthest_indices[replacement_cursor])
                used_indices.add(point_index)
                updated_centers[cluster_id] = coords[point_index]
                replacement_cursor += 1

        return updated_centers

    def _cluster_with_kmeans(
        self,
        coords,
        baseline_coords,
        k_clusters,
        tolerance=KMEANS_DISTANCE_TOLERANCE,
        max_iterations=KMEANS_MAX_ITERATIONS,
    ):
        centers = self._initialize_centers(coords, k_clusters)
        if len(centers) == 0:
            return np.array([], dtype=int), np.array([], dtype=int), np.empty((0, 2), dtype=float)

        for _ in range(max_iterations):
            labels, min_squared_distances = self._assign_to_closest_seed(coords, centers)
            if np.sqrt(min_squared_distances.max(initial=0.0)) < tolerance:
                break

            updated_centers = self._update_kmeans_centers(coords, labels, min_squared_distances, centers)
            if np.allclose(updated_centers, centers, atol=tolerance, rtol=0.0):
                centers = updated_centers
                break
            centers = updated_centers

        final_labels, _ = self._assign_to_closest_seed(coords, centers)
        baseline_labels, _ = self._assign_to_closest_seed(baseline_coords, centers)
        return final_labels, baseline_labels, centers

    def _cluster_coordinates(self, coords, baseline_coords, k_clusters, method):
        method = self._validate_cluster_method(method)
        if method == "kmeans":
            return self._cluster_with_kmeans(coords, baseline_coords, k_clusters)
        return self._cluster_with_seed_assignment(coords, baseline_coords, k_clusters)

    def cluster_locations(self, matched_categories, k_clusters=300, method="seed"):
        """
        Filter dataset by matching problem categories, assign each point to its
        cluster, then normalize concern severity by all 311 complaint weight
        assigned to the same cluster.
        """
        filtered_df = self._matched_dataframe(matched_categories)
        
        if len(filtered_df) == 0:
            return []

        coords = filtered_df[['Latitude', 'Longitude']].to_numpy(dtype=float)
        baseline_df = self.df[['Latitude', 'Longitude', 'recency_weight']].copy()
        baseline_coords = baseline_df[['Latitude', 'Longitude']].to_numpy(dtype=float)
        labels, baseline_labels, cluster_centers = self._cluster_coordinates(
            coords,
            baseline_coords,
            k_clusters,
            method,
        )
        if len(cluster_centers) == 0:
            return []

        filtered_df = filtered_df.copy()
        filtered_df['Cluster'] = labels
        baseline_df['Cluster'] = baseline_labels
        
        cluster_stats = []
        for i, cluster_center in enumerate(cluster_centers):
            cluster_data = filtered_df[filtered_df['Cluster'] == i]
            baseline_cluster_data = baseline_df[baseline_df['Cluster'] == i]
            count = len(cluster_data)
            if count == 0:
                continue

            concern_weight = cluster_data['severity_contribution'].sum()
            baseline_weight = baseline_cluster_data['recency_weight'].sum()
            baseline_count = len(baseline_cluster_data)

            # normalized_severity = concern_weight / baseline_weight if baseline_weight else 0.0
            complaint_share = concern_weight / baseline_weight if baseline_weight else 0.0
            normalized_severity = complaint_share * np.log1p(count)

            most_common_zip = cluster_data['Incident Zip'].mode().iloc[0] if not cluster_data['Incident Zip'].mode().empty else "Unknown"
            most_common_borough = cluster_data['Borough'].mode().iloc[0] if not cluster_data['Borough'].mode().empty else "Unknown"

            cluster_stats.append({
                'cluster_id': i,
                'center_lat': float(cluster_center[0]),
                'center_lon': float(cluster_center[1]),
                'complaint_count': int(count),
                'baseline_complaint_count': int(baseline_count),
                'baseline_score': float(baseline_weight),
                'severity_score': float(concern_weight),
                'normalized_severity': float(normalized_severity),
                'primary_zip': most_common_zip,
                'primary_borough': most_common_borough
            })

        ranked_clusters = sorted(
            cluster_stats,
            key=lambda x: (x['normalized_severity'], x['severity_score']),
            reverse=True,
        )
        return ranked_clusters

    def cluster_extremes(self, matched_categories, k_clusters=300, top_n=50, method="seed"):
        ranked_clusters = self.cluster_locations(
            matched_categories,
            k_clusters=k_clusters,
            method=method,
        )
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
    results = clusterer.cluster_locations(dummy_categories, k_clusters=300, method="kmeans")
    for res in results:
        print(f"Rank. {res['primary_borough']} Zip: {res['primary_zip']} - Score: {res['severity_score']:.2f} (Count: {res['complaint_count']})")

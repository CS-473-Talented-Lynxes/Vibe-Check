import numpy as np
from pathlib import Path
from huggingface_hub.constants import HF_HUB_CACHE
from sentence_transformers import SentenceTransformer

try:
    from src.config.config import EMBEDDINGS_OUTPUT_FILE
    from src.data.dataset import load_prepared_311_data
except ModuleNotFoundError:
    from config.config import EMBEDDINGS_OUTPUT_FILE
    from data.dataset import load_prepared_311_data

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class ComplaintSearcher:
    def __init__(self, data_path=None, embeddings_path=None):
        self.data_path = Path(data_path) if data_path else None
        self.embeddings_path = Path(embeddings_path) if embeddings_path else EMBEDDINGS_OUTPUT_FILE
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.categories = []
        self.category_texts = []
        self.embeddings = None

        self._load_and_embed()

    def _load_and_embed(self):
        print("Loading 311 dataset...")
        df = load_prepared_311_data(self.data_path)[["Problem", "Problem Detail"]]

        unique_categories = (
            df.drop_duplicates()
            .dropna()
            .sort_values(["Problem", "Problem Detail"])
            .reset_index(drop=True)
        )
        self.categories = unique_categories.to_dict("records")
        self.category_texts = [self._format_category_text(row) for row in self.categories]

        print(f"Found {len(self.categories)} unique problem categories.")

        if self._load_cached_embeddings():
            print(f"Loaded cached embeddings from {self.embeddings_path}.")
            return self.embeddings_path

        print(f"Computing embeddings with {MODEL_NAME}...")
        model = self._get_model()
        document_texts = [f"{DOCUMENT_PREFIX}{text}" for text in self.category_texts]
        self.embeddings = model.encode(
            document_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self._save_cached_embeddings()
        print(f"Embeddings computed and saved to {self.embeddings_path}.")
        return self.embeddings_path

    def _format_category_text(self, row):
        return f"{row['Problem']}: {row['Problem Detail']}"

    def _get_model(self):
        if self.model is None:
            self.model = self._load_model()
        return self.model

    def _load_model(self):
        cached_model_path = self._resolve_cached_model_path()
        if cached_model_path is not None:
            print(f"Loading cached model from {cached_model_path}...")
            return SentenceTransformer(
                str(cached_model_path),
                trust_remote_code=True,
                local_files_only=True,
            )

        try:
            return SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        except Exception:
            cached_model_path = self._resolve_cached_model_path()
            if cached_model_path is not None:
                print(f"Falling back to locally cached model files at {cached_model_path}...")
                return SentenceTransformer(
                    str(cached_model_path),
                    trust_remote_code=True,
                    local_files_only=True,
                )
            raise

    def _resolve_cached_model_path(self):
        model_cache_dir = Path(HF_HUB_CACHE) / f"models--{MODEL_NAME.replace('/', '--')}"
        snapshots_dir = model_cache_dir / "snapshots"
        refs_main = model_cache_dir / "refs" / "main"

        if refs_main.exists():
            snapshot_path = snapshots_dir / refs_main.read_text(encoding="utf-8").strip()
            if snapshot_path.exists():
                return snapshot_path

        if not snapshots_dir.exists():
            return None

        snapshot_candidates = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        if not snapshot_candidates:
            return None

        return max(snapshot_candidates, key=lambda path: path.stat().st_mtime)

    def _load_cached_embeddings(self):
        if not self.embeddings_path.exists():
            return False

        try:
            with np.load(self.embeddings_path, allow_pickle=True) as cached:
                cached_model_name = str(cached["model_name"].item())
                cached_category_texts = cached["category_texts"].tolist()
                cached_embeddings = cached["embeddings"]
        except (KeyError, ValueError, OSError):
            return False

        if cached_model_name != MODEL_NAME:
            return False
        if cached_category_texts != self.category_texts:
            return False

        self.embeddings = cached_embeddings
        return True

    def _save_cached_embeddings(self):
        np.savez_compressed(
            self.embeddings_path,
            model_name=np.array(MODEL_NAME),
            category_texts=np.array(self.category_texts, dtype=object),
            embeddings=self.embeddings.astype(np.float32),
        )

    def _cosine_similarities(self, query_embedding):
        query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        embedding_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True).T
        denominator = np.maximum(query_norm * embedding_norms, 1e-12)
        return ((query_embedding @ self.embeddings.T) / denominator)[0]

    def search(self, query: str, top_k: int = 50):
        """Search for top K matching categories based on semantic similarity to query."""
        query_embedding = self._get_model().encode(
            [f"{QUERY_PREFIX}{query}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        similarities = self._cosine_similarities(query_embedding)

        top_k_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_k_indices:
            results.append({
                "problem": self.categories[idx]["Problem"],
                "detail": self.categories[idx]["Problem Detail"],
                "similarity": float(similarities[idx]),
            })

        return results

    def get_category_labels(self):
        return [f"{row['Problem']} - {row['Problem Detail']}" for row in self.categories]

if __name__ == "__main__":
    searcher = ComplaintSearcher()
    query = "noise and loud music"
    print(f"\nTesting query: '{query}'")
    results = searcher.search(query, top_k=50)
    for i, res in enumerate(results):
        print(f"{i+1}. {res['problem']} - {res['detail']} (Sim: {res['similarity']:.4f})")

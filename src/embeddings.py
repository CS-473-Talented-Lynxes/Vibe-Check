import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Constants
MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__abspath__)), 'data', 'processed', 'cleaned_raw_311.csv') if '__abspath__' in globals() else os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_raw_311.csv')

class ComplaintSearcher:
    def __init__(self, data_path=None):
        self.data_path = data_path if data_path else os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'cleaned_raw_311.csv')
        self.model = SentenceTransformer(MODEL_NAME)
        self.categories = []
        self.category_texts = []
        self.embeddings = None
        
        self._load_and_embed()

    def _load_and_embed(self):
        print("Loading 311 dataset...")
        # We only need the problem categories to find unique ones
        df = pd.read_csv(self.data_path, usecols=['Problem', 'Problem Detail'])
        
        # Drop duplicates to find unique Problem + Detail combos
        unique_categories = df.drop_duplicates().dropna()
        self.categories = unique_categories.to_dict('records')
        
        print(f"Found {len(self.categories)} unique problem categories. Computing embeddings...")
        
        # Create text representation for embedding
        self.category_texts = [f"{row['Problem']}: {row['Problem Detail']}" for row in self.categories]
        
        # Compute embeddings
        self.embeddings = self.model.encode(self.category_texts)
        print("Embeddings computed successfully.")

    def search(self, query: str, top_k: int = 5):
        """Search for top K matching categories based on semantic similarity to query."""
        query_embedding = self.model.encode([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'problem': self.categories[idx]['Problem'],
                'detail': self.categories[idx]['Problem Detail'],
                'similarity': similarities[idx]
            })
            
        return results

if __name__ == "__main__":
    searcher = ComplaintSearcher()
    query = "noise and loud music"
    print(f"\nTesting query: '{query}'")
    results = searcher.search(query, top_k=5)
    for i, res in enumerate(results):
        print(f"{i+1}. {res['problem']} - {res['detail']} (Sim: {res['similarity']:.4f})")

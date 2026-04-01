import argparse
import sys
from embeddings import ComplaintSearcher
from clustering import LocationClusterer

def main():
    parser = argparse.ArgumentParser(description="Find NYC locations matching your least favorite urban issues.")
    parser.add_argument('--top_categories', type=int, default=5, help="Number of 311 problem categories to match")
    parser.add_argument('--clusters', type=int, default=15, help="Number of location clusters to find")
    args = parser.parse_args()

    print("=== NYC Vibe Check - Reverse Search ===")
    print("Initializing engine (this may take a few seconds)...")
    
    try:
        searcher = ComplaintSearcher()
        clusterer = LocationClusterer()
    except Exception as e:
        print(f"Failed to initialize engines: {str(e)}")
        sys.exit(1)

    while True:
        print("\n" + "="*50)
        query = input("Describe your least favorite urban issues (e.g., 'noise, rats, heating blockages') or type 'exit' to quit:\n> ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not query.strip():
            continue

        print("\nAnalyzing semantics and matching with 311 categories...")
        matches = searcher.search(query, top_k=args.top_categories)
        
        print(f"\nTop {args.top_categories} matching 311 categories:")
        for i, m in enumerate(matches):
            print(f"  {i+1}. {m['problem']} - {m['detail']} (Similarity: {m['similarity']:.2f})")
            
        print("\nGathering geographical data and finding K-Means clusters...")
        clusters = clusterer.cluster_locations(matches, k_clusters=args.clusters)
        
        if not clusters:
            print("No locations found for these categories.")
            continue
            
        print("\n*** Top Worst Ranked NYC Locations for Your Dislikes ***")
        for i, c in enumerate(clusters):
            print(f"Rank {i+1}:")
            print(f"  Borough/Zip : {c['primary_borough']}, {c['primary_zip']}")
            print(f"  Coordinates : {c['center_lat']:.5f}, {c['center_lon']:.5f}")
            print(f"  Severity    : {c['severity_score']:.2f} (sum of recency_weight * similarity across {c['complaint_count']} complaints)")
            print(f"  Map Link    : https://www.google.com/maps?q={c['center_lat']},{c['center_lon']}")
            print("-" * 40)
            
if __name__ == "__main__":
    main()

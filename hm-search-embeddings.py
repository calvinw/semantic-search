import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
from typing import List, Tuple

class ProductSearcher:
    def __init__(self, embeddings_file: str = "product_embeddings.npz"):
        """Initialize the searcher with embeddings file."""
        print("Loading embeddings and model...")
        # Load the saved embeddings and metadata
        self.data = np.load(embeddings_file, allow_pickle=True)
        self.embeddings = self.data['embeddings']
        self.product_names = self.data['product_names']
        self.embedding_strings = self.data['embedding_strings']
        self.product_codes = self.data['product_codes']
        self.article_ids_str = self.data['article_ids_str']
        
        # Load the same model used for creating embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"Loaded {len(self.embeddings)} products")
        
    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Search for products using a text query.
        Returns list of (index, score) tuples.
        """
        # Create embedding for the query
        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k matches
        top_idx = np.argsort(similarities)[::-1][:top_k]
        
        return [(idx, similarities[idx]) for idx in top_idx]
    
    def print_search_results(self, query: str, top_k: int = 3):
        """Perform search and print results in a readable format."""
        print("\n" + "="*80)
        print(f"Search Query: '{query}'")
        print("="*80)
        
        results = self.search(query, top_k)
        
        for idx, score in results:
            print(f"\nScore: {score:.3f}")
            print(f"Product: {self.product_names[idx]}")
            print(f"Product Code: {self.product_codes[idx]}")
            print(f"Article IDs: {self.article_ids_str[idx]}")
            print(f"Full Description: {self.embedding_strings[idx]}")
            print("-"*80)

def main():
    # Initialize searcher
    searcher = ProductSearcher()
    
    # Example searches to demonstrate different types of queries
    example_queries = [
        "black sports clothing",
        "warm winter pajamas",
        "underwear bra comfortable",
        "colorful socks",
        "formal wear",
        "baby clothes"
    ]
    
    # Perform searches
    for query in example_queries:
        searcher.print_search_results(query)
        print("\n")
    
    # Interactive search mode
    print("\nEnter your own searches (type 'quit' to exit):")
    while True:
        query = input("\nEnter search query: ").strip()
        if query.lower() == 'quit':
            break
        searcher.print_search_results(query)

if __name__ == "__main__":
    main()

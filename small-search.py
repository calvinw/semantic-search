import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

# Read the existing data
df = pd.read_csv('dress_catalog.csv')

# Create search texts
df['search_text'] = df.apply(lambda row: f"{row['prod_name']} in {row['perceived_colour_master_name']} ({row['category']}) {row['detail_desc']}", axis=1)

# Print all search texts
print("Search Texts for All Items:")
for idx, row in df.iterrows():
    print(f"\n{row['article_id']}: {row['search_text']}")

# Initialize model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings once
product_embeddings = model.encode(df['search_text'].tolist())

def semantic_search(query, top_k=3):
    """Perform semantic search and return results"""
    query_embedding = model.encode(query)
    
    # Calculate similarities
    similarities = np.dot(product_embeddings, query_embedding) / (
        np.linalg.norm(product_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get top results
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'article_id': df.iloc[idx]['article_id'],
            'product': df.iloc[idx]['prod_name'],
            'category': df.iloc[idx]['category'],
            'color': df.iloc[idx]['perceived_colour_master_name'],
            'similarity': similarities[idx],
            'description': df.iloc[idx]['detail_desc']
        })
    
    return results

# Example searches
example_searches = [
    "something formal for a wedding",
    "comfortable summer dress with pockets",
    "professional dress for business meetings",
    "sparkly party dress",
    "sustainable eco-friendly dress",
    "black dress that's wrinkle free for travel",
    "elegant silk evening wear",
    "warm dress for winter"
]

print("\nSemantic Search Examples:\n")

for query in example_searches:
    print(f"\nSearch Query: '{query}'")
    print("-" * 80)
    
    results = semantic_search(query)
    
    for i, result in enumerate(results, 1):
        print(f"\nMatch {i} (Similarity: {result['similarity']:.3f}):")
        print(f"Product: {result['product']} ({result['color']})")
        print(f"Category: {result['category']}")
        print(f"Article ID: {result['article_id']}")
        print(f"Description: {result['description']}")
    
    time.sleep(1)  # Pause between searches for readability

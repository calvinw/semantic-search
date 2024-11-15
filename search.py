import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import torch
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

image_path = "https://github.com/calvinw/semantic-search/blob/main/images/"

# Load embeddings from file for semantic search
def load_embeddings():
    with open('combined_embeddings.pkl', 'rb') as f:
        data = pickle.load(f)
    return pd.DataFrame(data)

data = load_embeddings()
print(data.head())

# Perform semantic search
def semantic_search(query, data, model, top_n=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, list(data['combined_embeddings']))[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return data.iloc[top_indices], similarities[top_indices]

# Example queries
queries = [
    "A comfortable dress for casual wear with pockets.",
    "Elegant long black dress for an evening event.",
    "A cozy winter dress with long sleeves and a hood.",
    "A sleeveless summer dress in green color.",
    "A sporty striped dress that is light and easy to move in."
]

# Perform semantic search for each query
for i, query in enumerate(queries):
    print(f"\nTop matches for query {i+1}: '{query}'\n")
    results, similarities = semantic_search(query, data, model, top_n=5)
    for idx, (index, row) in enumerate(results.iterrows()):
        print(f"Rank {idx + 1}:\nArticle ID: {row['article_id']}\nCombined Text: {row['combined_text']}\nSimilarity: {similarities[idx]:.4f}\n")



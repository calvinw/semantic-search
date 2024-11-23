import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

def load_product_data():
    """
    Load product data from GitHub URL
    """
    url = "https://raw.githubusercontent.com/calvinw/semantic-search/refs/heads/main/articles-1000.csv"
    try:
        df = pd.read_csv(url)
        print(f"Successfully loaded {len(df)} products")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_search_text(row):
    """
    Create a rich text representation for semantic embedding
    """
    # Primary product description
    main_desc = f"{row['prod_name']} - {row['product_type_name']} {row['product_group_name']}"
    
    # Color information (simplified)
    color_desc = f"in {row['perceived_colour_master_name']}"
    
    # Category and department context
    category_desc = f"from {row['department_name']} {row['section_name']}"
    
    # Detailed description (if available)
    detailed_desc = row['detail_desc'] if pd.notna(row['detail_desc']) else ""
    
    # Combine all elements, filtering out empty strings
    elements = [main_desc, color_desc, category_desc, detailed_desc]
    return " ".join([e for e in elements if e])

def create_product_embeddings(df):
    """
    Create semantic search embeddings for product catalog
    """
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create search texts
    df['search_text'] = df.apply(create_search_text, axis=1)
    
    # Generate embeddings
    embeddings = model.encode(df['search_text'].tolist(), 
                            show_progress_bar=True,
                            batch_size=32)
    
    return embeddings

def semantic_search(query, df, embeddings, top_k=5):
    """
    Perform semantic search on product catalog
    """
    # Initialize model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate query embedding
    query_embedding = model.encode(query)
    
    # Calculate cosine similarities
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Create a list to store unique results
    results = []
    seen_products = set()  # Track unique combinations of product name and description
    
    # Sort all indices by similarity score
    all_indices = np.argsort(similarities)[::-1]
    
    # Iterate through all indices until we have top_k unique products
    for idx in all_indices:
        product_name = df.iloc[idx]['prod_name']
        description = df.iloc[idx]['detail_desc']
        # Create a unique identifier for the product
        product_key = f"{product_name}_{description}"
        
        if product_key not in seen_products:
            seen_products.add(product_key)
            results.append({
                'article_id': df.iloc[idx]['article_id'],
                'product': product_name,
                'description': description,
                'similarity_score': similarities[idx],
                'department': df.iloc[idx]['department_name'],
                'color': df.iloc[idx]['perceived_colour_master_name']
            })
            
            if len(results) == top_k:
                break
    
    return results

# Example usage:
if __name__ == "__main__":
    # Load the data
    df = load_product_data()
    
    if df is not None:
        # Create embeddings
        print("Creating embeddings...")
        embeddings = create_product_embeddings(df)
        
        # Example search
        query = "red sports clothing"
        print(f"\nSearching for: {query}")
        results = semantic_search(query, df, embeddings)
        
        # Print results
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Article ID: {result['article_id']}")
            print(f"Product: {result['product']}")
            print(f"Color: {result['color']}")
            print(f"Department: {result['department']}")
            print(f"Similarity Score: {result['similarity_score']:.3f}")
            print(f"Description: {result['description']}...")

        # Print just article IDs in a compact format
        print("\nMatching Article IDs:")
        article_ids = [result['article_id'] for result in results]
        print(", ".join(article_ids))

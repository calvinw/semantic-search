import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import time
from collections import Counter
import requests
from io import StringIO

def load_remote_csv(github_raw_url):
    """
    Load CSV file directly from GitHub raw URL.
    """
    print(f"Downloading data from GitHub...")
    response = requests.get(github_raw_url)
    response.raise_for_status()  # Raise exception for bad status codes
    
    print("Parsing CSV data...")
    return pd.read_csv(StringIO(response.text))

def create_enhanced_descriptions(df):
    """
    Create enhanced product descriptions with grouped attributes.
    Returns DataFrame with unique products and their enhanced descriptions.
    """
    print("\nGrouping products and creating enhanced descriptions...")
    
    # Group by product code
    product_groups = df.groupby('product_code').agg({
        'prod_name': 'first',
        'product_type_name': 'first',
        'detail_desc': 'first',
        'colour_group_name': lambda x: sorted(list(set(x))),
        'graphical_appearance_name': lambda x: sorted(list(set(x))),
        'article_id': list  # Keep all article IDs for this product
    }).reset_index()
    
    # Create enhanced descriptions
    results = []
    for _, row in product_groups.iterrows():
        colors = f"[Colors: {', '.join(row['colour_group_name'])}]"
        patterns = f"[Patterns: {', '.join(row['graphical_appearance_name'])}]"
        
        enhanced_desc = (f"{row['prod_name']} | {row['product_type_name']} | "
                        f"{colors} {patterns} | {row['detail_desc']}")
        
        results.append({
            'product_code': row['product_code'],
            'product_name': row['prod_name'],
            'article_ids': row['article_id'],
            'num_variants': len(row['colour_group_name']),
            'colors': row['colour_group_name'],
            'patterns': row['graphical_appearance_name'],
            'embedding_string': enhanced_desc
        })
    
    return pd.DataFrame(results)

def print_dataset_statistics(df, enhanced_df):
    """Print comprehensive statistics about the dataset and embeddings."""
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print("\nBasic Statistics:")
    print(f"Total number of articles: {len(df)}")
    print(f"Unique products: {len(enhanced_df)}")
    print(f"Average variants per product: {enhanced_df['num_variants'].mean():.2f}")
    
    print("\nVariant Distribution:")
    variant_counts = enhanced_df['num_variants'].value_counts().sort_index()
    for variants, count in variant_counts.items():
        print(f"Products with {variants} variants: {count}")
    
    print("\nTop 10 Products with Most Variants:")
    most_variants = enhanced_df.nlargest(10, 'num_variants')
    for _, row in most_variants.iterrows():
        print(f"{row['product_name']}: {row['num_variants']} variants")
    
    # Analyze colors
    all_colors = [color for colors in enhanced_df['colors'] for color in colors]
    print("\nTop 10 Most Common Colors:")
    for color, count in Counter(all_colors).most_common(10):
        print(f"{color}: {count} products")
    
    # Analyze patterns
    all_patterns = [pattern for patterns in enhanced_df['patterns'] for pattern in patterns]
    print("\nPattern Distribution:")
    for pattern, count in Counter(all_patterns).most_common():
        print(f"{pattern}: {count} products")
    
    # Analyze string lengths
    string_lengths = enhanced_df['embedding_string'].str.len()
    print("\nEmbedding String Length Statistics:")
    print(f"Average length: {string_lengths.mean():.1f} characters")
    print(f"Maximum length: {string_lengths.max()} characters")
    print(f"Minimum length: {string_lengths.min()} characters")

def save_embeddings(embeddings, product_data, filepath):
    """
    Save embeddings and related data as a compressed NPZ file.
    Handles lists of different lengths properly.
    """
    if not filepath.endswith('.npz'):
        filepath = f"{filepath}.npz"
    
    # Convert article_ids to a string representation for each product
    article_ids_str = [','.join(map(str, ids)) for ids in product_data['article_ids']]
    
    # Prepare metadata
    metadata = {
        'num_products': len(product_data),
        'embedding_dim': embeddings.shape[1],
        'date_created': np.datetime64('now').astype(str),
        'model_used': 'all-MiniLM-L6-v2',
        'description': 'Product embeddings with grouped color variants'
    }
    
    # Save compressed npz file
    np.savez_compressed(
        filepath,
        embeddings=embeddings,
        product_codes=product_data['product_code'].values,
        article_ids_str=article_ids_str,  # Store as comma-separated strings
        product_names=product_data['product_name'].values,
        embedding_strings=product_data['embedding_string'].values,
        metadata=metadata
    )
    
    # Print file info
    file_size_mb = Path(filepath).stat().st_size / (1024 * 1024)
    print(f"\nSaved embeddings file:")
    print(f"Path: {filepath}")
    print(f"Size: {file_size_mb:.2f} MB")
    print(f"Products: {len(product_data)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    
    # Print first few products as examples
    print("\nFirst few products saved:")
    for i in range(min(3, len(product_data))):
        print(f"\nProduct: {product_data['product_name'].iloc[i]}")
        print(f"Article IDs: {article_ids_str[i]}")
        print(f"Embedding string length: {len(product_data['embedding_string'].iloc[i])}")


def main():
    start_time = time.time()
    
    # GitHub raw URL for your articles.csv
    # Replace this URL with your actual GitHub raw URL
    github_url = "https://raw.githubusercontent.com/calvinw/semantic-search/refs/heads/main/articles.csv"
    
    # Load the dataset from GitHub
    try:
        df = load_remote_csv(github_url)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Rest of main function remains the same
    enhanced_df = create_enhanced_descriptions(df)
    print_dataset_statistics(df, enhanced_df)
    
    print("\nCreating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(
        enhanced_df['embedding_string'].tolist(),
        show_progress_bar=True
    )
    
    save_embeddings(embeddings, enhanced_df, "product_embeddings")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.1f} seconds")

if __name__ == "__main__":
    main()

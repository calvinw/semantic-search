import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def create_dress_embeddings_viz(df):
    """
    Create and visualize embeddings for dresses using UMAP.
    """
    # Filter for dresses
    dress_df = df[
        (df['category'].str.contains('Dress', case=False, na=False)) |
        (df['detail_desc'].str.contains('dress', case=False, na=False))
    ].copy()
    
    if len(dress_df) > 50:
        dress_df = dress_df.head(50)
    
    print(f"Found {len(dress_df)} dresses")
    
    # Create embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    text_to_embed = dress_df.apply(
        lambda x: f"{x['prod_name']} {x['category']} {x['detail_desc']}", 
        axis=1
    )
    
    print("Creating embeddings...")
    embeddings = model.encode(
        text_to_embed.tolist(),
        show_progress_bar=True
    )
    
    print("Reducing dimensionality with UMAP...")
    # Faster UMAP settings
    umap_reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=10,  # reduced from 15
        min_dist=0.1,
        metric='euclidean',
        n_epochs=200,    # reduced from default
        learning_rate=1.0,
        init='random',   # faster than 'spectral'
        verbose=True
    )
    
    umap_embeddings = umap_reducer.fit_transform(embeddings)
    
    print("Creating visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # First subplot - Color by perceived color
    colors = dress_df['perceived_colour_master_name'].astype('category').cat.codes
    scatter1 = ax1.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
                          c=colors, cmap='tab20', alpha=0.6)
    
    # Add legend for colors
    legend1 = ax1.legend(scatter1.legend_elements()[0], 
                        dress_df['perceived_colour_master_name'].unique(),
                        title="Colors",
                        bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.add_artist(legend1)
    ax1.set_title('Dress Embeddings by Color')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    
    # Second subplot - Color by category
    categories = dress_df['category'].astype('category').cat.codes
    scatter2 = ax2.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
                          c=categories, cmap='tab20', alpha=0.6)
    
    # Add legend for categories
    legend2 = ax2.legend(scatter2.legend_elements()[0], 
                        dress_df['category'].unique(),
                        title="Categories",
                        bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.add_artist(legend2)
    ax2.set_title('Dress Embeddings by Category')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    
    plt.tight_layout()
    plt.show()
    
    return embeddings, umap_embeddings, dress_df

def print_closest_pairs(umap_embeddings, dress_df, n_pairs=5):
    """Print the n closest pairs of dresses based on UMAP distance"""
    from scipy.spatial.distance import pdist, squareform
    
    distances = squareform(pdist(umap_embeddings))
    np.fill_diagonal(distances, np.inf)
    closest_pairs = []
    
    for _ in range(n_pairs):
        min_idx = np.unravel_index(distances.argmin(), distances.shape)
        closest_pairs.append(min_idx)
        distances[min_idx] = np.inf
    
    print("\nClosest dress pairs in embedding space:")
    for idx1, idx2 in closest_pairs:
        dress1 = dress_df.iloc[idx1]
        dress2 = dress_df.iloc[idx2]
        print(f"\nPair:")
        print(f"1: {dress1['prod_name']} ({dress1['perceived_colour_master_name']})")
        print(f"   {dress1['detail_desc'][:100]}...")
        print(f"2: {dress2['prod_name']} ({dress2['perceived_colour_master_name']})")
        print(f"   {dress2['detail_desc'][:100]}...")

if __name__ == "__main__":
    df = pd.read_csv('dress_catalog.csv')
    embeddings, umap_embeddings, dress_df = create_dress_embeddings_viz(df)
    print_closest_pairs(umap_embeddings, dress_df)

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

def plot_embedding(phrases, num_clusters):
    # Load the model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings
    embeddings = model.encode(phrases)

    # Reduce to 2 dimensions using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Perform clustering using KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    
    # Ensure we have enough colors by recycling if necessary
    while len(colors) < num_clusters:
        colors.extend(colors)
    
    # Plot points and labels
    for i, (x, y) in enumerate(embeddings_2d):
        ax.scatter(x, y, color=colors[cluster_labels[i]], 
                  label=f"Cluster {cluster_labels[i]}" if f"Cluster {cluster_labels[i]}" not in [l.get_label() for l in ax.get_lines()] else "")
        ax.text(x + 0.02, y, phrases[i], fontsize=9)

    plt.title("2D Visualization of Sentence Embeddings with Clustering")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.legend(loc='best', fontsize=8)
    plt.grid(True)
    plt.axis('equal')
    
    return fig, embeddings, cluster_labels

def plot_heatmap(phrases, embeddings):
    # Calculate cosine similarity matrix
    similarity_matrix = np.inner(embeddings, embeddings)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(similarity_matrix, cmap='YlOrRd')
    
    # Add labels
    ax.set_xticks(np.arange(len(phrases)))
    ax.set_yticks(np.arange(len(phrases)))
    ax.set_xticklabels(phrases, rotation=45, ha='right')
    ax.set_yticklabels(phrases)
    
    plt.title("Cosine Similarity Heatmap")
    plt.tight_layout()
    
    return fig

# Set page title
st.title("Sentence Embeddings Visualizer")

# Add text area for input
text_input = st.text_area(
    "Enter sentences (one per line):",
    height=200,
    help="Enter the sentences you want to analyze, with one sentence per line.",
    placeholder="Example:\nI like to be in my house\nI enjoy staying home\nI like spending time where I live"
)

# Add number input for clusters
num_clusters = st.number_input(
    "Number of clusters:",
    min_value=1,
    max_value=8,
    value=2,
    help="Select the number of clusters for K-means clustering"
)

# Add button to trigger embedding
if st.button("Generate Embeddings"):
    if text_input.strip():
        # Split input into lines and remove empty lines
        phrases = [line.strip() for line in text_input.split('\n') if line.strip()]
        
        if len(phrases) < num_clusters:
            st.error(f"Please enter at least {num_clusters} sentences for {num_clusters} clusters.")
        else:
            # Create tabs for the visualizations
            tab1, tab2 = st.tabs(["Embedding Plot", "Similarity Heatmap"])
            
            # Generate and display the embedding plot
            with tab1:
                st.subheader("2D Visualization of Sentence Embeddings")
                with st.spinner("Generating embeddings plot..."):
                    fig, embeddings, cluster_labels = plot_embedding(phrases, num_clusters)
                    st.pyplot(fig)
            
            # Generate and display the heatmap
            with tab2:
                st.subheader("Similarity Heatmap")
                with st.spinner("Generating heatmap..."):
                    fig = plot_heatmap(phrases, embeddings)
                    st.pyplot(fig)
    else:
        st.error("Please enter some sentences to analyze.")

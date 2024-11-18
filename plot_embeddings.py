from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Expanded sample phrases
phrases = ["I like to be in my house", 
         "I enjoy staying home", 
         "I like spending time where I live.", 
         "I love sleeping all day.",
         "The isotope 238u decays to 206pb"]
#phrases = [
    # "Striped long-sleeve tee",
    # "Leather watch",
    # "Cocktail sequin dress",
    # "Running sneakers",
    # "Hoop earrings",
    # "Floral sundress",
    # "Suede loafers",
    # "Statement necklace",
    # "Graphic print tee",
    # "Maxi wrap dress",
    # "Strappy sandals",
    # "Silk scarf",
    # "Little black dress",
    # "Plain white v-neck",
    # "Leather ankle boots",
    # "Crop top"
#]

# Generate embeddings
embeddings = model.encode(phrases)

# Reduce to 2 dimensions using PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Perform clustering using KMeans
num_clusters = 2 
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

# Plot the 2D embeddings with clusters
plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'green', 'purple']
for i, (x, y) in enumerate(embeddings_2d):
    plt.scatter(x, y, color=colors[cluster_labels[i]], label=f"Cluster {cluster_labels[i]}" if f"Cluster {cluster_labels[i]}" not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.text(x + 0.02, y + 0.02, phrases[i], fontsize=9)

plt.title("2D Visualization of Sentence Embeddings with Clustering")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.legend(loc='best', fontsize=8)
plt.grid(True)
plt.axis('equal')
plt.show()


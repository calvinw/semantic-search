import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV data into a pandas DataFrame from Google Colab
from google.colab import files

# Prompt the user to upload the CSV file
uploaded = files.upload()

# Load the uploaded CSV file into a DataFrame
csv_filename = list(uploaded.keys())[0]
df = pd.read_csv(csv_filename)

# Basic dataset information
print("Dataset Overview:\n")
print(df.info())
print("\nFirst Few Rows of Data:\n")
print(df.head())

# Check if GPU is available and use it if possible
if torch.cuda.is_available():
    device = 'cuda'
    print("Using GPU for inference.")
else:
    device = 'cpu'
    print("Using CPU for inference.")

# Generate embeddings for product descriptions using Sentence Transformers
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Combine relevant fields into a single string for each product
df['combined_text'] = df.apply(lambda row: f"{row['prod_name']} {row['product_type_name']} {row['product_group_name']} {row['graphical_appearance_name']} {row['colour_group_name']} {row['detail_desc']}", axis=1)
combined_descriptions = df['combined_text'].tolist()
combined_embeddings = model.encode(combined_descriptions, batch_size=32)

# Add combined embeddings to DataFrame
df['combined_embeddings'] = list(combined_embeddings)
print("\nEmbeddings for Combined Product Information:\n")
print(df[['prod_name', 'combined_embeddings']])

# Save combined embeddings for later use
with open('combined_embeddings.pkl', 'wb') as f:
    pickle.dump(df[['article_id', 'combined_text', 'combined_embeddings']].to_dict(), f)

# Download the saved embeddings file to local system
files.download('combined_embeddings.pkl')

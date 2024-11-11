import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

def main():
    # Load the CSV data into a pandas DataFrame
    df = pd.read_csv('hm-dresses-full.csv')

    # Basic dataset information
    print("Dataset Overview:\n")
    print(df.info())
    print("\nFirst Few Rows of Data:\n")
    print(df.head())

    # Analyze and display unique product types and their count
    product_type_counts = df['product_type_name'].value_counts()
    print("\nProduct Types and Their Counts:\n")
    print(product_type_counts)

    # Analyze and display color distribution
    color_counts = df['colour_group_name'].value_counts()
    print("\nColor Distribution:\n")
    print(color_counts)

    # Plotting color distribution
    color_counts.plot(kind='bar', title='Color Distribution of Products', xlabel='Color', ylabel='Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Group by department name and count the number of items
    department_counts = df.groupby('department_name').size()
    print("\nProduct Count by Department:\n")
    print(department_counts)

    # Plotting department distribution
    department_counts.plot(kind='pie', title='Product Count by Department', autopct='%1.1f%%', figsize=(8, 8))
    plt.ylabel('')
    plt.show()

    # Generate embeddings for product descriptions using Sentence Transformers
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Combine relevant fields into a single string for each product
    df['combined_text'] = df.apply(lambda row: f"{row['prod_name']} {row['product_type_name']} {row['product_group_name']} {row['graphical_appearance_name']} {row['colour_group_name']} {row['detail_desc']}", axis=1)
    combined_descriptions = df['combined_text'].tolist()
    combined_embeddings = model.encode(combined_descriptions)

    # Add combined embeddings to DataFrame
    df['combined_embeddings'] = list(combined_embeddings)
    print("\nEmbeddings for Combined Product Information:\n")
    print(df[['prod_name', 'combined_embeddings']])

    # Save combined embeddings for later use
    with open('combined_embeddings.pkl', 'wb') as f:
        pickle.dump(df[['article_id', 'combined_embeddings']], f)

if __name__ == "__main__":
    main()


from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create the fake fashion dataset
fashion_data = {
    "Item ID": [1, 2, 3, 4, 5],
    "Title": [
        "Red Leather Jacket",
        "Blue Denim Jeans",
        "Black Running Shoes",
        "White Cotton Shirt",
        "Green Silk Scarf"
    ],
    "Description": [
        "A bold red leather jacket perfect for making a statement.",
        "Classic blue denim jeans that are comfortable and stylish.",
        "Lightweight black running shoes designed for all-day comfort.",
        "A crisp white cotton shirt ideal for both casual and formal wear.",
        "Elegant green silk scarf to add a touch of sophistication to any outfit."
    ]
}

# Convert to a DataFrame
fashion_df = pd.DataFrame(fashion_data)

# Sample phrases to test
sample_phrases = [
    "Relaxed outfit for a casual outing.",
    "Comfortable and stylish everyday clothing.",
    "Elegant clothing for business meetings.",
    "Professional attire suitable for office or formal events.",
    "Shoes for running and jogging.",
    "Lightweight and breathable sportswear.",
    "Warm jacket for winter days.",
    "Cool and airy fabric for summer.",
    "Stylish scarf to complement an outfit.",
    "Fashionable accessories for a modern look.",
    "Clothing made with silk for a luxurious feel.",
    "Cotton shirts that are soft and comfortable.",
    "Bold red outfit for making a statement.",
    "Neutral white clothing for versatile styling.",
    "Fashionable denim jeans for a timeless look.",
    "Jackets that stand out in a crowd."
]

# Semantic-only phrases
semantic_only_phrases = [
    "Outerwear for chilly weather.",
    "A garment suitable for formal dinners.",
    "Lightweight footwear for exercise.",
    "Something soft and comfortable for lounging.",
    "A luxurious accessory to elevate an outfit.",
    "Clothes that keep you cool in hot climates.",
    "An elegant piece for a high-end event.",
    "A casual item for a weekend getaway.",
    "Stylish protection from wind and rain.",
    "Comfortable clothing for traveling long distances."
]

specific_semantic_queries = [
    # Matches "Red Leather Jacket"
    "An eye-catching piece of outerwear designed to stand out in any situation.",  
    # Matches "Blue Denim Jeans"
    "Timeless casual bottoms that blend comfort with practicality for everyday wear.",  
    # Matches "Black Running Shoes"
    "Durable footwear optimized for all-day activity and a sporty lifestyle.",  
    # Matches "White Cotton Shirt"
    "A versatile garment ideal for both relaxed outings and more polished events.",  
    # Matches "Green Silk Scarf"
    "A graceful accessory that adds a refined touch to an ensemble for special occasions."  
]

# Encode item descriptions once
item_embeddings = model.encode(fashion_df['Description'], convert_to_tensor=True)

# Loop through sample phrases
for query in specific_semantic_queries:
    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(query_embedding, item_embeddings).squeeze()
    
    # Print results
    print(f"Query: {query}")
    for idx, score in enumerate(cosine_scores):
        print(f"  Title: {fashion_df.iloc[idx]['Title']}")
        print(f"  Description: {fashion_df.iloc[idx]['Description']}")
        print(f"  Similarity Score: {score:.4f}")
        print(f"  ")
    print("-" * 50)


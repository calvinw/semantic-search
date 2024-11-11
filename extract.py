# Import required libraries
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Load the H&M articles dataset
# This is the correct path for the competition dataset
df = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/articles.csv')

# Create a cleaned subset with 1000 random items
subset = df.sample(n=1000)

# Create a simpler version with relevant columns
simple_df = pd.DataFrame({
    'article_id': subset['article_id'],
    'product_name': subset['prod_name'],
    'description': subset.apply(
        lambda row: f"{row['prod_name']} - {row['product_type_name']} - {row['detail_desc']}".strip(),
        axis=1
    ),
    'price': subset['price'],
    'color': subset['colour_group_name'],
    'category': subset['product_type_name'],
    'department': subset['department_name']
})

# Save to CSV
simple_df.to_csv('hm_articles_subset.csv', index=False)

# Print some information about the saved dataset
print(f"Dataset saved with {len(simple_df)} items")
print("\nColumn descriptions:")
print("- article_id: Unique identifier for each product")
print("- product_name: Short product name")
print("- description: Detailed product description")
print("- price: Product price")
print("- color: Product color")
print("- category: Product type/category")
print("- department: Department name")

# Show the first few rows
print("\nFirst few rows of the dataset:")
print(simple_df.head())

# Show some basic statistics
print("\nBasic statistics:")
print(f"Number of unique categories: {simple_df['category'].nunique()}")
print(f"Number of unique colors: {simple_df['color'].nunique()}")
print(f"Price range: {simple_df['price'].min():.2f} - {simple_df['price'].max():.2f}")
print("\nDepartment distribution:")
print(simple_df['department'].value_counts())

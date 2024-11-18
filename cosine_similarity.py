import numpy as np
import matplotlib.pyplot as plt

# Define our product vectors
product_a = np.array([8, 2])  # High-end leather bag
product_b = np.array([7, 3])  # Designer shoes
product_c = np.array([2, 9])  # Cozy slippers

# Create the plot
plt.figure(figsize=(10, 10))

# Plot vectors as arrows from origin
plt.quiver(0, 0, product_a[0], product_a[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Product A')
plt.quiver(0, 0, product_b[0], product_b[1], angles='xy', scale_units='xy', scale=1, color='red', label='Product B')
plt.quiver(0, 0, product_c[0], product_c[1], angles='xy', scale_units='xy', scale=1, color='green', label='Product C')

# Add simple labels at the end of each vector
plt.annotate(f'Leather Bag (8, 2)', 
            xy=(product_a[0], product_a[1]), 
            xytext=(10, 0), 
            textcoords='offset points', 
            ha='left',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='blue', alpha=0.8))

plt.annotate(f'Designer Shoes (7, 3)', 
            xy=(product_b[0], product_b[1]), 
            xytext=(10, 0), 
            textcoords='offset points', 
            ha='left',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='red', alpha=0.8))

plt.annotate(f'Cozy Slippers (2, 9)', 
            xy=(product_c[0], product_c[1]), 
            xytext=(-20, 10), 
            textcoords='offset points', 
            ha='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='green', alpha=0.8))

# Add some visual improvements
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Set limits and labels
plt.xlim(-1, 11)
plt.ylim(-1, 11)
plt.xlabel('Luxury Score', fontsize=12)
plt.ylabel('Comfort Score', fontsize=12)
plt.title('Product Vectors: Luxury vs Comfort Scores', fontsize=14)

# Show the plot
plt.grid(True)
plt.show()

# Calculate and print cosine similarities separately
def cosine_similarity(v1, v2):
    return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

print("\nCosine Similarities:")
print(f"Leather Bag & Designer Shoes: {cosine_similarity(product_a, product_b):.3f}")
print(f"Leather Bag & Cozy Slippers: {cosine_similarity(product_a, product_c):.3f}")
print(f"Designer Shoes & Cozy Slippers: {cosine_similarity(product_b, product_c):.3f}")

print("\nAngles between vectors (degrees):")
print(f"Leather Bag & Designer Shoes: {np.degrees(np.arccos(cosine_similarity(product_a, product_b))):.1f}°")
print(f"Leather Bag & Cozy Slippers: {np.degrees(np.arccos(cosine_similarity(product_a, product_c))):.1f}°")
print(f"Designer Shoes & Cozy Slippers: {np.degrees(np.arccos(cosine_similarity(product_b, product_c))):.1f}°")

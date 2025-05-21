import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine

def calculate_cosine_similarity(vec_size):
  """Calculates the cosine similarity between two random Gaussian vectors."""
  vec1 = np.random.randn(vec_size)
  vec2 = np.random.randn(vec_size)
  # Cosine similarity is 1 - cosine distance
  return 1 - cosine(vec1, vec2)

vector_sizes = [1,2,4,5,10, 100, 1000, 5000, 10000]
num_trials = 50  # Number of times to repeat the calculation for robustness

results = {}
for size in vector_sizes:
  similarities = [calculate_cosine_similarity(size) for _ in range(num_trials)]
  results[size] = {
      'mean_similarity': np.mean(similarities),
      'std_dev_similarity': np.std(similarities)
  }

print("Cosine Similarity Test Results for Gaussian Vectors:")
for size, data in results.items():
  print(f"Vector Size: {size}")
  print(f"  Mean Cosine Similarity: {data['mean_similarity']:.4f}")
  print(f"  Standard Deviation: {data['std_dev_similarity']:.4f}")
  print("-" * 20)

# Optional: Plot the results
sizes = list(results.keys())
means = [results[size]['mean_similarity'] for size in sizes]
stds = [results[size]['std_dev_similarity'] for size in sizes]

fig = plt.figure(figsize=(8, 5), facecolor='w')
plt.errorbar(sizes, means, yerr=stds, fmt='o-', capsize=5)
plt.xscale('log')
plt.xlabel("Vector Size (log scale)")
plt.ylabel("Mean Cosine Similarity")
plt.title("Cosine Similarity vs. Vector Size for Gaussian Vectors")
plt.grid(True)

# plt.savefig(data)
# image = F"data:image/png;base64,{base64.b64encode(data.getvalue()).decode()}"
# alt = "Cosine Similarity vs. Vector Size"
# display.display(display.Markdown(F"""![{alt}]({image})"""))
# plt.close(fig)

import chromadb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Initialize ChromaDB client with persistent storage
chroma_client = chromadb.PersistentClient(path="chroma_storage")

# Get the collection
collection = chroma_client.get_collection(name="personal_collection")

# Fetch all embeddings and their metadata
results = collection.get(include=["embeddings", "metadatas"])

# Extract embeddings and their corresponding finger names
embeddings = np.array(results["embeddings"])
finger_names = [metadata["finger"] for metadata in results["metadatas"]]

# Dimensionality reduction using PCA (you can also use t-SNE)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Alternatively, you can use t-SNE for dimensionality reduction
# tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
# reduced_embeddings = tsne.fit_transform(embeddings)

# Plot the reduced embeddings
plt.figure(figsize=(10, 8))
for i, finger_name in enumerate(set(finger_names)):
    indices = [idx for idx, name in enumerate(finger_names) if name == finger_name]
    plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=finger_name)

plt.title("2D Visualization of Finger Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.show()
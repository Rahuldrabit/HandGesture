import chromadb

# Initialize ChromaDB client with persistent storage
chroma_client = chromadb.PersistentClient(path="chroma_storage")

# Check if collection exists, otherwise create it
try:
    collection = chroma_client.get_collection(name="personal_collection")
except:
    collection = chroma_client.create_collection(name="personal_collection")

# Delete embeddings by their ID or other criteria
# Example: Deleting embeddings with specific IDs
embedding_ids_to_delete = ['Puch', 'Middle']
collection.delete(embedding_ids_to_delete)

print("Embeddings deleted successfully.")

from qdrant_client import QdrantClient

# Connect to Qdrant
client = QdrantClient(url="http://localhost:6333")

# Retrieve the list of collections
collections = client.get_collections()

# Print the names of the collections
print("Available collections:", collections.collections)

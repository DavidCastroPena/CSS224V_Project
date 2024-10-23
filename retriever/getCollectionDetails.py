from qdrant_client import QdrantClient

# Connect to Qdrant
client = QdrantClient(url="http://localhost:6333")  # Ensure Qdrant is running and available at this address

# Name of the collection you're checking
collection_name = "paper_chunks"  # Replace this with the correct name of your collection

# Retrieve collection information
info = client.get_collection(collection_name)

# Print the collection details
print(info)

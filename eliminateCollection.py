from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams

# Connect to Qdrant
client = QdrantClient(url="http://localhost:6333")

# Check if the collection exists, and delete it if it does
collection_name = "paper_chunks"
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

# Now recreate the collection with the correct vector size
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance="Cosine")  # Set size to 384
)

print("Collection recreated with 384 dimensions.")

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams  # Import VectorParams

#Connect to Qdrant running locally
client = QdrantClient(host="localhost", port=6333)

# heck
try:
    # Create or recreate the collection in Qdrant
    client.recreate_collection(
        collection_name="paper_chunks",
        vectors_config=VectorParams(size=384, distance="Cosine")  # Correctly define the vector parameters
    )
    print("Connected to Qdrant and collection created.")
except Exception as e:
    print(f"Failed to connect to Qdrant: {e}")

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams


class QdrantCollection:
    def __init__(self, host="localhost", port=6333, collection_name="paper_chunks", vector_size=384, distance="Cosine"):
        """
        Initialize the QdrantCollection with connection parameters and recreate the collection.
        Args:
            host (str): Hostname for Qdrant.
            port (int): Port for Qdrant.
            collection_name (str): Name of the Qdrant collection.
            vector_size (int): Dimension of the vector embeddings.
            distance (str): Distance metric for similarity search ('Cosine', 'Euclid', etc.).
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

        self._initialize_collection()

    def _initialize_collection(self):
        """
        Recreate the Qdrant collection if it doesn't already exist.
        """
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=self.distance)
            )
            print(f"Connected to Qdrant and collection '{self.collection_name}' created.")
        except Exception as e:
            print(f"Failed to connect to Qdrant or initialize collection: {e}")
            raise

    def get_client(self):
        """
        Get the Qdrant client instance.
        Returns:
            QdrantClient: The Qdrant client.
        """
        return self.client

    def get_collection_name(self):
        """
        Get the name of the collection.
        Returns:
            str: The collection name.
        """
        return self.collection_name


if __name__ == "__main__":
    # Example standalone usage
    print("Testing QdrantCollection functionality...")

    # Initialize the QdrantCollection
    qdrant_collection = QdrantCollection()

    # Example access to the client and collection name
    client = qdrant_collection.get_client()
    collection_name = qdrant_collection.get_collection_name()

    print(f"Client connected to collection: {collection_name}")

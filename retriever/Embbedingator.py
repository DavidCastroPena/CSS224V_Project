import json
import torch
import uuid
from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import CollectionInfo, VectorParams, PointStruct


class Embbedingator:
    def __init__(self, model_name="BAAI/bge-small-en", qdrant_host="localhost", qdrant_port=6333):
        """
        Initialize the Embbedingator with a model and Qdrant client.
        Args:
            model_name (str): The Hugging Face model name for embedding generation.
            qdrant_host (str): Hostname for the Qdrant client.
            qdrant_port (int): Port for the Qdrant client.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

    def initialize_qdrant_collection(self, collection_name, vector_size=384, distance="Cosine"):
        """
        Ensure the Qdrant collection exists. Create it if it does not exist.
        Args:
            collection_name (str): Name of the collection to check or create.
            vector_size (int): The size of the vector embeddings.
            distance (str): The distance metric for similarity search (e.g., 'Cosine', 'Euclid').

        Returns:
            None
        """
        collections = self.qdrant_client.get_collections()
        if collection_name not in [col.name for col in collections.collections]:
            print(f"Creating Qdrant collection '{collection_name}'...")
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance)
            )
            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Collection '{collection_name}' already exists.")

    def embed_text(self, text):
        """
        Compute the embedding for a given text using the model.
        Args:
            text (str): The text to embed.

        Returns:
            numpy.ndarray: The embedding vector.
        """
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()

    def index_embedding(self, text, paper_id, collection_name="paper_chunks"):
        """
        Compute and index the embedding for a text chunk into Qdrant.
        Args:
            text (str): The text chunk to embed.
            paper_id (str): Identifier for the paper to which the chunk belongs.
            collection_name (str): Name of the Qdrant collection.

        Returns:
            None
        """
        # Ensure the collection exists
        self.initialize_qdrant_collection(collection_name)

        # Generate embedding
        embedding = self.embed_text(text)

        # Generate a unique chunk ID
        chunk_id = str(uuid.uuid4())

        # Upsert into Qdrant
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=chunk_id,
                    vector=embedding.tolist(),
                    payload={
                        "paper_id": paper_id,
                        "chunk_text": text,
                        "chunk_id": chunk_id
                    }
                )
            ]
        )
        print(f"Indexed chunk for paper '{paper_id}' with chunk ID: {chunk_id}")


if __name__ == "__main__":
    # Example standalone usage
    print("Testing Embbedingator functionality...")

    # Initialize the Embbedingator
    embbedingator = Embbedingator()

    # Simulated input data
    sample_text = "This is a sample text chunk to test embeddings."
    sample_paper_id = "TestPaper123"

    # Compute and index the embedding
    collection_name = "test_paper_chunks"
    embbedingator.index_embedding(sample_text, sample_paper_id, collection_name)

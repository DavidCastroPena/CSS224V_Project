import torch
import json
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from transformers import AutoTokenizer, AutoModel
import re
import datetime
import numpy as np


class PerformQuery:
    def __init__(self, model_name="BAAI/bge-small-en", qdrant_host="localhost", qdrant_port=6333, collection_name="paper_chunks"):
        """
        Initialize PerformQuery with a Qdrant client and embedding model.
        Args:
            model_name (str): Hugging Face model name for embedding generation.
            qdrant_host (str): Hostname for Qdrant.
            qdrant_port (int): Port for Qdrant.
            collection_name (str): Name of the Qdrant collection.
        """
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Ensure the Qdrant collection exists
        self._initialize_qdrant_collection()

    def _initialize_qdrant_collection(self):
        """
        Ensure the Qdrant collection exists. Raise an error if it doesn't exist.
        """
        try:
            collections = self.qdrant_client.get_collections()
            if self.collection_name not in [col.name for col in collections.collections]:
                raise Exception(f"Collection '{self.collection_name}' does not exist in Qdrant. Ensure it is created and populated before querying.")
        except ResponseHandlingException as e:
            print(f"Error while checking Qdrant collections: {e}")
            raise e

    def get_embedding(self, text):
        """
        Generate an embedding for the given text using the model.
        Args:
            text (str): Input text to embed.

        Returns:
            numpy.ndarray: The embedding vector.
        """
        inputs = self.tokenizer(text, return_tensors="pt", max_length=384, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()

    def calculate_similarity(self, query_vector, chunk_vector):
        """
        Calculate the cosine similarity between the query vector and a chunk vector.
        Args:
            query_vector (numpy.ndarray): Query embedding.
            chunk_vector (numpy.ndarray): Chunk embedding.

        Returns:
            float: Cosine similarity score.
        """
        dot_product = np.dot(query_vector, chunk_vector)
        norm_query = np.linalg.norm(query_vector)
        norm_chunk = np.linalg.norm(chunk_vector)
        return dot_product / (norm_query * norm_chunk)

    def query_qdrant(self, query_text, top_k=5):
        """
        Perform a similarity search in Qdrant for the given query text.
        Args:
            query_text (str): Text query for similarity search.
            top_k (int): Number of top results to retrieve.

        Returns:
            list: Search results from Qdrant.
        """
        # Get embedding for the query text
        query_embedding = self.get_embedding(query_text)

        try:
            # Perform search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k
            )
            return search_results
        except ResponseHandlingException as e:
            print(f"Error during Qdrant search: {e}")
            raise e

    def save_query_results(self, query_text, results):
        """
        Save query results to a JSON file.
        Args:
            query_text (str): The original query text.
            results (list): Search results to save.

        Returns:
            str: Path to the saved JSON file.
        """
        # Sanitize query text for filename
        sanitized_query = re.sub(r'[^a-zA-Z0-9_]', '_', query_text)[:50]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./query_results_{sanitized_query}_{timestamp}.json"

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "Score": result.score,
                "PaperID": result.payload.get("paper_id"),
                "ChunkText": result.payload.get("chunk_text"),
                "ChunkID": result.payload.get("chunk_id"),
                "ID": result.id
            })

        # Save results to JSON
        with open(filename, 'w') as file:
            json.dump(formatted_results, file, indent=4)

        print(f"Query results saved to {filename}")
        return filename


if __name__ == "__main__":
    # Example standalone usage
    print("Testing PerformQuery functionality...")

    # Initialize PerformQuery
    perform_query = PerformQuery()

    # Example query
    query_text = "I want to create a policy that fosters financial inclusion and economic growth in California."
    print("I'm using the wrong query")
    try:
        # Perform the query
        results = perform_query.query_qdrant(query_text, top_k=20)

        # Display results
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"Score: {result.score}")
            print(f"Paper ID: {result.payload.get('paper_id')}")
            print(f"Chunk Text: {result.payload.get('chunk_text')[:200]}...")
            print(f"Chunk ID: {result.payload.get('chunk_id')}")
            print(f"ID: {result.id}\n")

        # Save the results
        perform_query.save_query_results(query_text, results)
    except Exception as e:
        print(f"Error during testing: {e}")

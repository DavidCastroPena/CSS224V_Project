'''
DO NOT USE--STORE UNTIL FIRST TEST IS SUCCESFULL, THEN ELIMINATE
import os
import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import torch
from transformers import AutoTokenizer, AutoModel

# Path to your chunk folder
chunks_folder = "C:/Users/engin/OneDrive/Desktop/Carpetas/STANFORD/fourthQuarter/CS224V/Project/chunks"  # Change to your actual chunk folder path

# Initialize the Qdrant Client
qdrant_client = QdrantClient(host="localhost", port=6333)

# Model and tokenizer initialization (using the BAAI/bge-small-en model)
model_name = "BAAI/bge-small-en"  # The embedding model you're using
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Take the average of the last hidden state as the embedding
    return torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()

# Loop over the chunks and insert embeddings into Qdrant
for chunk_file in os.listdir(chunks_folder):
    if chunk_file.endswith(".json"):
        file_path = os.path.join(chunks_folder, chunk_file)
        with open(file_path, "r") as f:
            chunks = json.load(f)

        points = []
        for chunk in chunks:
            text = chunk["text"]
            paper_id = chunk["paper_id"]
            chunk_id = chunk["chunk_id"]

            # Generate embedding for the chunk
            embedding = get_embedding(text)

            # Prepare point structure for upserting
            point = PointStruct(
                id=f"{paper_id}_{chunk_id}",  # Unique ID for each chunk
                vector=embedding.tolist(),    # Convert embedding to list for JSON serialization
                payload={
                    "paper_id": paper_id, 
                    "chunk_id": chunk_id, 
                    "text": text
                }
            )
            points.append(point)

        # Insert all points at once to Qdrant
        qdrant_client.upsert(
            collection_name="paper_chunks",
            points=points
        )

print("Embeddings indexed successfully into Qdrant.")


'''


import json
import torch
import uuid  # For generating UUIDs
from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

#Path
jsonl_file_path = "./paper_chunk.jsonl"

#Initialize the Qdrant Client (assuming Qdrant is running locally)
qdrant_client = QdrantClient(host="localhost", port=6333)

#Load the embedding model (BAAI/bge-small-en used as an example)
model_name = "BAAI/bge-small-en"  # Replace with the model of your choice
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

#Function to compute embedding from text
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()

#Read the JSONL file and process each chunk
with open(jsonl_file_path, 'r') as jsonl_file:
    for line in jsonl_file:
        chunk_data = json.loads(line)
        
        #Extract necessary fields from the chunk data
        text = chunk_data["content_string"]
        paper_id = chunk_data["article_title"] 
        chunk_id = str(uuid.uuid4())  # Generate a UUID for the chunk ID to ensure uniqueness

        # Compute the embedding for the text chunk
        embedding = get_embedding(text)

        # Insert the embedding and related metadata into Qdrant
        qdrant_client.upsert(
            collection_name="paper_chunks",
            points=[
                PointStruct(
                    id=chunk_id,  #Use the generated UUID 
                    vector=embedding.tolist(),  
                    payload={
                        "paper_id": paper_id,
                        "chunk_text": text,
                        "chunk_id": chunk_id 
                    }
                )
            ]
        )

print("Embeddings processed and indexed into Qdrant successfully.")

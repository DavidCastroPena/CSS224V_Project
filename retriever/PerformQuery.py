import torch
import json
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
import re
import datetime

# Initialize Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)

# Initialize the same embedding model used for indexing
model_name = "BAAI/bge-small-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get embedding for a query
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()

# Function to query Qdrant
def query_qdrant(query_text, top_k=5):
    # Get the embedding of the query text
    query_embedding = get_embedding(query_text)

    # Perform search in Qdrant
    search_results = qdrant_client.search(
        collection_name="paper_chunks",  # Ensure this is the correct collection name
        query_vector=query_embedding.tolist(),  # Convert the embedding to a list
        limit=top_k,  # How many top results you want to retrieve
    )

    return search_results

# Function to save query results to a JSON file
def save_query_results(query_text, results):
    # Sanitize query text to create a valid filename
    sanitized_query = re.sub(r'[^a-zA-Z0-9_]', '_', query_text)[:50]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./query_results_{sanitized_query}_{timestamp}.json"

    # Prepare data to save
    formatted_results = []
    for i, result in enumerate(results):
        formatted_results.append({
            "Score": result.score,
            "PaperID": result.payload.get("paper_id"),
            "ChunkText": result.payload.get("chunk_text"),
            "ChunkID": result.payload.get("chunk_id"),
            "ID": result.id
        })

    # Save results as JSON
    with open(filename, 'w') as file:
        json.dump(formatted_results, file, indent=4)

    print(f"Query results saved to {filename}")

# Example usage
query_text = "banking challenges"
results = query_qdrant(query_text, top_k=10)

# Display and save the results
for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Score: {result.score}")
    print(f"Paper ID: {result.payload.get('paper_id')}")
    print(f"Chunk Text: {result.payload.get('chunk_text')[:200]}...")
    print(f"Chunk ID: {result.payload.get('chunk_id')}")
    print(f"ID: {result.id}\n")

# Save the results to a file
save_query_results(query_text, results)

import os
import json
from transformers import AutoTokenizer
import PyPDF2

# Define the folder paths
chunks_folder = "C:/Users/engin/OneDrive/Desktop/Carpetas/STANFORD/fourthQuarter/CS224V/Project/chunks"
papers_folder = "C:/Users/engin/OneDrive/Desktop/Carpetas/STANFORD/fourthQuarter/CS224V/Project/papers"

# Initialize the tokenizer for chunking (GPT-2 tokenizer used as an example, you can adjust this)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Function to split text into chunks ensuring each is less than 500 tokens and handling max token limit
def chunk_text(text, max_tokens=500, model_max_length=1024):
    tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        # Ensure the chunk doesn't exceed the model's max token length
        if len(tokenizer(chunk_text, return_tensors="pt", max_length=model_max_length, truncation=True).input_ids[0]) <= model_max_length:
            chunks.append(chunk_text)
    return chunks

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Define the output .jsonl file
output_jsonl = "C:/Users/engin/OneDrive/Desktop/Carpetas/STANFORD/fourthQuarter/CS224V/Project/paper_chunks.jsonl"

# Open the output file in write mode
with open(output_jsonl, 'w') as outfile:
    # Loop through all PDF files in the papers folder
    for paper_file in os.listdir(papers_folder):
        if paper_file.endswith(".pdf"):
            paper_id = paper_file.split(".")[0]  # Use the file name as the paper ID
            pdf_path = os.path.join(papers_folder, paper_file)

            # Extract text from the PDF
            full_text = extract_text_from_pdf(pdf_path)

            # Chunk the extracted text
            chunks = chunk_text(full_text)

            # Write each chunk to the JSONL file with the appropriate metadata
            for i, chunk in enumerate(chunks):
                json_line = {
                    "id": f"{paper_id}_{i}",  # Unique chunk ID (paper_id_chunk_number)
                    "content_string": chunk,  # The chunked text
                    "article_title": paper_id,  # The title can be the paper ID or extracted separately
                    "full_section_title": "Section",  # Placeholder for section title
                    "block_type": "text",  # Since these are text chunks
                    "language": "en",  # Assuming all papers are in English
                    "last_edit_date": "2024-01-01",  # Optional; you can provide real data if available
                    "num_tokens": len(tokenizer(chunk, return_tensors="pt").input_ids[0])  # Number of tokens in this chunk
                }
                # Write the chunk as a JSON line in the output file
                outfile.write(json.dumps(json_line) + "\n")

print(f"Chunks successfully saved to {output_jsonl}")

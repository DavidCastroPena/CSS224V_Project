import os
import json
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken


# Define the folder paths
papers_folder = "./papers"


# Define the model to count the tokens
encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model('gpt-4o-mini')

# Initialize the RecursiveCharacterTextSplitter
recur_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=50, separators=[".", ",", " ", ""]
)

# Function to split text into chunks ensuring each is less than 500 tokens and handling max token limit
def chunk_text(text):
    # Split the text into chunks
    chunks = recur_text_splitter.split_text(text)
    return chunks

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()

    # Save the extracted text into a txt file
    #with open("extracted_text.txt", "w") as text_file:
    #    text_file.write(text)

    return text

# Function to count tokens using tiktoken
def count_openai_tokens(text, encoding_name = "cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

# Define the output .jsonl file
output_jsonl = "paper_chunk.jsonl"

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
                    "full_section_title": "",  # Placeholder for section title
                    "block_type": "text",  # Since these are text chunks
                    "language": "en",  # Assuming all papers are in English
                    "last_edit_date": "2024-01-01",  # Optional; you can provide real data if available
                    "num_tokens": count_openai_tokens(chunk)  # Number of tokens in this chunk
                }
                # Write the chunk as a JSON line in the output file
                outfile.write(json.dumps(json_line) + "\n")

print(f"Chunks successfully saved to {output_jsonl}")

import os
import json
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# Define the folder path for PDFs
papers_folder = "C:/Users/engin/OneDrive/Desktop/Carpetas/STANFORD/fourthQuarter/CS224V/Project/papers"

# Set up the token encoding model
encoding = tiktoken.encoding_for_model("gpt-4")

# Initialize the RecursiveCharacterTextSplitter with appropriate chunk settings
recur_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=50, separators=[".", ",", " ", ""]
)

# Function to split text into chunks
def chunk_text(text):
    chunks = recur_text_splitter.split_text(text)
    if not chunks:
        print("Warning: No chunks generated for text.")
    else:
        print(f"Generated {len(chunks)} chunks.")
    return chunks

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:  # Check if text was successfully extracted
                    text += page_text
                else:
                    print(f"Warning: Page {page_num} of {pdf_path} has no extractable text.")
        
        if text.strip() == "":
            print(f"Warning: No text extracted from {pdf_path}. Skipping this file.")
        
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

# Function to count tokens in text using tiktoken
def count_openai_tokens(text, encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

# Define the output .jsonl file
output_jsonl = "paper_chunk.jsonl"

# Process each PDF in the papers folder and save chunks to the output JSONL file
with open(output_jsonl, 'w', encoding="utf-8") as outfile:
    for paper_file in os.listdir(papers_folder):
        if paper_file.endswith(".pdf"):
            paper_id = paper_file.split(".")[0]  # Use the filename without extension as paper ID
            pdf_path = os.path.join(papers_folder, paper_file)

            # Extract text from the PDF
            full_text = extract_text_from_pdf(pdf_path)
            if not full_text.strip():
                print(f"Warning: Skipping {pdf_path} due to empty text.")
                continue  # Skip this file if no text was extracted

            # Chunk the extracted text
            chunks = chunk_text(full_text)
            if not chunks:
                print(f"Warning: No chunks generated for {pdf_path}. Skipping this file.")
                continue  # Skip if no chunks were generated

            # Write each chunk to the JSONL file with appropriate metadata
            for i, chunk in enumerate(chunks):
                json_line = {
                    "id": f"{paper_id}_{i}",  # Unique chunk ID
                    "content_string": chunk,  # The chunked text
                    "article_title": paper_id,  # The title can be the paper ID
                    "full_section_title": "",  # Placeholder for section title
                    "block_type": "text",  # These are text chunks
                    "language": "en",  # Assuming papers are in English
                    "last_edit_date": "2024-01-01",  # Optional; adjust as needed
                    "num_tokens": count_openai_tokens(chunk)  # Token count for the chunk
                }
                # Write the chunk as a JSON line in the output file
                outfile.write(json.dumps(json_line) + "\n")

print(f"Chunks successfully saved to {output_jsonl}")


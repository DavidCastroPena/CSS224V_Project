from pathlib import Path
import json
import os
import google.generativeai as genai
import PyPDF2
import numpy as np
from scipy.spatial.distance import cosine
from datetime import datetime
from dotenv import load_dotenv

# Define paths
query_results_path = r"C:/Users/engin/OneDrive/Desktop/Carpetas/STANFORD/fourthQuarter/CS224V/Project/query_results_banking_challenges_20241030_142254.json"
papers_dir = r"C:/Users/engin/OneDrive/Desktop/Carpetas/STANFORD/fourthQuarter/CS224V/Project/papers"

# Load environment variables
load_dotenv()

# Retrieve the API key
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Ensure that the API key is properly loaded
if not gemini_api_key:
    raise ValueError("API key not found. Please check your .env file for GEMINI_API_KEY.")

# Configure the genai library with the API key
genai.configure(api_key=gemini_api_key)

class QueryAnalyzer:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
        self.PROJECT_DIR = Path(".")
        self.user_query = "I want to know about policies of financial inclusion"  # User's specific interest

    def get_latest_query_results_file(self):
        """Use the specified query results path."""
        if os.path.exists(query_results_path):
            print(f"Using specified query results file: {query_results_path}")
            return query_results_path
        else:
            print(f"Specified query results file not found at: {query_results_path}")
            return None

    def load_relevant_papers(self, filename):
        """Load unique paper IDs from the query results JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                query_results = json.load(file)
            print(f"Loaded query results from {filename}")
            return {entry["PaperID"] for entry in query_results}
        except Exception as e:
            print(f"Error loading query results: {e}")
            return None

    def extract_abstract_from_pdf(self, paper_id):
        """Extract the abstract section from the PDF."""
        pdf_path = os.path.join(papers_dir, f"{paper_id}.pdf")
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                full_text = ""
                for page_num in range(len(reader.pages)):
                    page_text = reader.pages[page_num].extract_text()
                    full_text += page_text
            # Flexible search for abstract based on text structure
            if "Abstract" in full_text:
                start_idx = full_text.find("Abstract") + len("Abstract")
                end_idx = full_text.find("\n\n", start_idx)  # Assuming first paragraph break marks end
                return full_text[start_idx:end_idx].strip()
            else:
                # Use the first paragraph if "Abstract" is not found
                first_paragraph = full_text.split("\n\n")[0]
                return first_paragraph.strip()
        except FileNotFoundError:
            print(f"File {pdf_path} not found.")
            return ""
    
    def generate_embeddings(self, text):
        """Generate embeddings for the given text (placeholder)."""
        return np.random.rand(512)  # Replace with actual embedding generation model if available

    def is_embedding_close(self, abstract_embedding, question_embeddings, threshold=0.3):
        """Check if any question embeddings are close enough to the abstract embedding based on cosine similarity."""
        for question_embedding in question_embeddings:
            distance = cosine(abstract_embedding, question_embedding)
            if distance < threshold:
                return True
        return False

    def generate_questions_from_abstract(self, abstract, abstract_embedding):
        """Generate three questions based on the abstract, ensuring similarity in embeddings."""
        max_attempts = 5
        for _ in range(max_attempts):
            # Update the prompt with the user's query to inform question generation
            prompt = (
                f"Generate three insightful questions about policies of financial inclusion "
                f"based on the content of the following abstract: {abstract}. "
                f"{self.user_query}"
            )
            questions = self.fetch_questions(prompt)

            # Generate embeddings for each question
            question_embeddings = [self.generate_embeddings(question) for question in questions]

            # Check if any question is close enough to the abstract embedding
            if self.is_embedding_close(abstract_embedding, question_embeddings):
                return questions[:3]  # Limit to three questions
        
        return ["Error: Unable to generate suitable questions"]  # Fallback if no suitable questions are found

    def fetch_questions(self, prompt):
        """Fetch questions using the generative model."""
        try:
            response = self.model.generate_content(prompt)
            if response and response.text:
                return response.text.splitlines()[:3]  # Ensure only 3 questions are returned
            else:
                return ["Error: No questions generated"]
        except Exception as e:
            print(f"Error during question generation: {e}")
            return ["Error: Generation failed"]

    def process_papers(self, paper_ids):
        """Process each paper and generate exactly three questions grounded in the abstract."""
        output_path = self.PROJECT_DIR / f"questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for paper_id in paper_ids:
                abstract = self.extract_abstract_from_pdf(paper_id)
                if not abstract:
                    print(f"No abstract found for paper {paper_id}")
                    continue

                # Generate embedding for the abstract
                abstract_embedding = self.generate_embeddings(abstract)

                # Generate three questions that are grounded in the abstract and informed by the user's query
                questions = self.generate_questions_from_abstract(abstract, abstract_embedding)

                # Ensure only three questions per paper
                if len(questions) > 3:
                    questions = questions[:3]

                entry = {
                    "paper_id": paper_id,
                    "abstract": abstract,
                    "questions": questions
                }
                json.dump(entry, outfile)
                outfile.write("\n")
            print(f"Questions saved at {output_path}")

def main():
    analyzer = QueryAnalyzer()
    latest_file = analyzer.get_latest_query_results_file()
    if latest_file:
        paper_ids = analyzer.load_relevant_papers(latest_file)
        if paper_ids:
            analyzer.process_papers(paper_ids)

if __name__ == "__main__":
    main()





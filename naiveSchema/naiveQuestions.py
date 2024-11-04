from pathlib import Path
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai
import PyPDF2
import random
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast

# Define the correct path to the query results and papers directory
query_results_path = r"C:/Users/engin/OneDrive/Desktop/Carpetas/STANFORD/fourthQuarter/CS224V/Project/query_results_banking_challenges_20241030_142254.json"
papers_dir = r"C:/Users/engin/OneDrive/Desktop/Carpetas/STANFORD/fourthQuarter/CS224V/Project/papers"

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

class PaperEmbeddingAnalyzer:
    def __init__(self):
        # Load a pretrained BERT model and tokenizer specialized in policy/economics
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # hypothetical model
        self.model = AutoModel.from_pretrained("bert-base-uncased")
    
    def embed_text(self, text):
        """Generate embeddings for a given text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Mean pooling of token embeddings

    def sample_and_embed_section(self, section_text, iterations=4, sample_size=3):
        """Sample sentences and calculate cosine similarities to identify section's general idea."""
        sentences = section_text.split('.')
        accumulated_embeddings = []
        
        for _ in range(iterations):
            sample_sentences = random.sample(sentences, sample_size)
            sample_text = ' '.join(sample_sentences)
            embedding = self.embed_text(sample_text)
            
            # Calculate similarity if we have previous embeddings
            if accumulated_embeddings:
                similarities = [cosine_similarity(embedding, prev_emb) for prev_emb in accumulated_embeddings]
                avg_similarity = np.mean(similarities)
                print(f"Avg Cosine Similarity at Iteration {_+1}: {avg_similarity}")
            
            accumulated_embeddings.append(embedding)
            
        # Average the accumulated embeddings for the final section representation
        section_embedding = torch.mean(torch.stack(accumulated_embeddings), dim=0)
        return section_embedding

    def analyze_paper(self, abstract, summary, findings):
        """Extract embeddings for key sections and combine them."""
        abstract_emb = self.sample_and_embed_section(abstract)
        summary_emb = self.sample_and_embed_section(summary)
        findings_emb = self.sample_and_embed_section(findings)

        # Combine section embeddings to represent the full paper
        combined_embedding = torch.mean(torch.stack([abstract_emb, summary_emb, findings_emb]), dim=0)
        return combined_embedding


class QueryAnalyzer:
    def __init__(self, embedding_analyzer):
        # Set up Gemini API with specified model and embedding analyzer
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.embedding_analyzer = embedding_analyzer
        self.PROJECT_DIR = Path(r".")

    def get_latest_query_results_file(self):
        """Directly use the specified query_results_path."""
        if os.path.exists(query_results_path):
            print(f"Using specified query results file: {query_results_path}")
            return query_results_path
        else:
            print(f"Specified query results file not found at: {query_results_path}")
            return None

    def load_relevant_papers(self, filename):
        """Load query results from a JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                query_results = json.load(file)
            print(f"Successfully loaded query results from {filename}")
            unique_paper_ids = {entry["PaperID"] for entry in query_results}
            print(f"Number of relevant papers found: {len(unique_paper_ids)}")
            return unique_paper_ids

        except Exception as e:
            print(f"Error loading query results: {e}")
            return None

    def extract_text_from_pdf(self, paper_id):
        pdf_path = os.path.join(papers_dir, f"{paper_id}.pdf")
        
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text()
            return text
        except FileNotFoundError:
            print(f"File {pdf_path} not found.")
            return ""

    def generate_comparison_questions(self, paper_embeddings, topic, question_number=3):
        """Generate comparison questions using cosine similarity of embeddings."""
        questions = []
        for idx, (paper_id, embedding) in enumerate(paper_embeddings.items()):
            prompt = f"Given the topic of {topic} and the similarities across papers, create {question_number} questions that explore unique aspects of the methodologies, findings, or conclusions in paper '{paper_id}' based on embedding-based similarities."
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.7, "max_output_tokens": 500}
            )
            if response and response.text.strip():
                questions.append(response.text.strip())
            else:
                questions.append("Error generating questions.")
        return questions

    def run(self):
        """Main function to load results and generate questions."""
        latest_file = self.get_latest_query_results_file()
        
        if not latest_file:
            print("\nNo query results available.")
            return
        
        print(f"\nUsing latest query results file: {latest_file}")

        # Load set of relevant papers
        relevant_papers_ids = self.load_relevant_papers(latest_file)
        if not relevant_papers_ids:
            return

        # Step 1: Process each paper's sections to create embeddings
        paper_embeddings = {}
        for paper_id in relevant_papers_ids:
            print(f"Processing paper {paper_id}...")
            paper_text = self.extract_text_from_pdf(paper_id)
            abstract, summary, findings = self.extract_sections(paper_text)
            embedding = self.embedding_analyzer.analyze_paper(abstract, summary, findings)
            paper_embeddings[paper_id] = embedding
        
        # Step 2: Generate and save comparison questions
        comparison_questions = self.generate_comparison_questions(paper_embeddings, topic="banking challenges")
        
        # Step 3: Save the generated questions to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        questions_file = self.PROJECT_DIR / f"comparison_questions_{timestamp}.txt"

        try:
            with open(questions_file, 'w', encoding='utf-8') as f:
                for question in comparison_questions:
                    f.write(question + "\n")
            print(f"\nGenerated questions saved to: {questions_file}")
        except Exception as e:
            print(f"\nError saving questions: {e}")

    def extract_sections(self, paper_text):
        """Extracts abstract, summary, and findings sections from paper text."""
        # This would be refined to identify each section in actual implementation
        # For simplicity, we're assuming these are roughly split into thirds here
        abstract = paper_text[:len(paper_text)//3]
        summary = paper_text[len(paper_text)//3:2*len(paper_text)//3]
        findings = paper_text[2*len(paper_text)//3:]
        return abstract, summary, findings


def main():
    embedding_analyzer = PaperEmbeddingAnalyzer()
    analyzer = QueryAnalyzer(embedding_analyzer)
    analyzer.run()

if __name__ == "__main__":
    main()
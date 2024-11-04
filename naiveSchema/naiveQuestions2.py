from pathlib import Path
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bertopic import BERTopic
import re
import yake

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Define paths with the user-specified locations
query_results_path = "C:/Users/engin/OneDrive/Desktop/Carpetas/STANFORD/fourthQuarter/CS224V/Project/query_results_banking_challenges_20241030_142254.json"
papers_dir = "C:/Users/engin/OneDrive/Desktop/Carpetas/STANFORD/fourthQuarter/CS224V/Project/papers"

class PaperEmbeddingAnalyzer:
    def __init__(self):
        # Initialize SciBERT tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.keyword_extractor = yake.KeywordExtractor()
        self.topic_model = BERTopic()  # BERTopic model for topic modeling
    
    def embed_text(self, text):
        """Generate embeddings for a given text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def extract_keywords(self, text, top_k=10):
        """Extract key phrases from text using YAKE."""
        keywords = self.keyword_extractor.extract_keywords(text)
        # Select top-k keywords sorted by score (lower is better in YAKE)
        keywords = sorted(keywords, key=lambda x: x[1])[:top_k]
        return [kw[0] for kw in keywords]
    
    def analyze_paper(self, title, abstract, findings):
        """Generate a composite embedding by combining title, abstract, and findings embeddings."""
        title_emb = self.embed_text(title)
        abstract_emb = self.embed_text(abstract)
        findings_emb = self.embed_text(findings)
        
        # Composite embedding using weighted averages
        combined_embedding = torch.mean(torch.stack([title_emb * 1.5, abstract_emb, findings_emb]), dim=0)
        return combined_embedding
    
    def extract_topics(self, text):
        """Extract main topics from text using BERTopic."""
        topics, _ = self.topic_model.fit_transform([text])
        return self.topic_model.get_topic(topics[0])

class QueryAnalyzer:
    def __init__(self, embedding_analyzer):
        # Configure the Gemini API
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-pro")
        self.embedding_analyzer = embedding_analyzer
        self.PROJECT_DIR = Path(".")
        self.output_file = self.PROJECT_DIR / f"question_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    def load_relevant_papers(self, filename):
        """Load query results from a JSON file."""
        try:
            with open(filename, "r", encoding="utf-8") as file:
                query_results = json.load(file)
            unique_paper_ids = {entry["PaperID"] for entry in query_results}
            return unique_paper_ids
        except Exception as e:
            print(f"Error loading query results: {e}")
            return None

    def extract_text_from_pdf(self, paper_id):
        pdf_path = os.path.join(papers_dir, f"{paper_id}.pdf")
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "".join(page.extract_text() for page in reader.pages)
            return text
        except FileNotFoundError:
            print(f"File {pdf_path} not found.")
            return ""

    def extract_sections(self, paper_text):
        """Extracts title, abstract, and findings sections from paper text."""
        title = paper_text.split("\n")[0]  # Assume the title is the first line
        abstract = paper_text[:len(paper_text) // 3]
        findings = paper_text[2 * len(paper_text) // 3:]
        return title, abstract, findings

    def generate_questions(self, topic, keywords, paper_embedding):
        """Generate three questions for each paper based on its main topic, keywords, and embedding."""
        prompt = (
            f"Based on the following topic and keywords, generate three questions that "
            f"help compare the subjects, main findings, and differences of this paper to others.\n\n"
            f"Topic: {topic}\n"
            f"Keywords: {', '.join(keywords)}\n"
            f"Embedding (used for context, not shown here): {paper_embedding}\n"
            "Questions:"
        )
        
        response = self._call_with_retry(lambda: self.model.generate_content(
            prompt,
            generation_config={"temperature": 0.7, "max_output_tokens": 300}
        ))
        
        if response and response.text.strip():
            return response.text.strip().split("\n")
        else:
            return ["Error generating questions."]

    def save_questions(self, paper_id, questions):
        """Save questions in a JSONL file with paper ID and generated questions."""
        with open(self.output_file, "a", encoding="utf-8") as f:
            json.dump({"paper_id": paper_id, "questions": questions}, f)
            f.write("\n")

    def analyze_and_generate_questions(self):
        """Process each relevant paper to extract topics, keywords, and generate questions."""
        relevant_papers_ids = self.load_relevant_papers(query_results_path)
        if not relevant_papers_ids:
            return

        for paper_id in relevant_papers_ids:
            paper_text = self.extract_text_from_pdf(paper_id)
            if paper_text:
                # Extract sections and generate embeddings
                title, abstract, findings = self.extract_sections(paper_text)
                keywords = self.embedding_analyzer.extract_keywords(paper_text)
                paper_embedding = self.embedding_analyzer.analyze_paper(title, abstract, findings)
                
                # Extract topics
                topics = self.embedding_analyzer.extract_topics(paper_text)
                main_topic = topics[0][0] if topics else "No main topic found"
                
                # Generate three comparison-focused questions per paper
                questions = self.generate_questions(main_topic, keywords, paper_embedding)
                
                # Save questions to JSONL
                self.save_questions(paper_id, questions)

    def _call_with_retry(self, func, retries=3, backoff=2):
        """Helper method to handle retries with exponential backoff on API limit errors."""
        for attempt in range(retries):
            try:
                return func()
            except ResourceExhausted:
                wait_time = backoff ** attempt
                print(f"API limit hit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        print("API limit exceeded and retry attempts exhausted.")
        return None

    def run(self):
        """Run the analyzer to generate questions."""
        self.analyze_and_generate_questions()


def main():
    embedding_analyzer = PaperEmbeddingAnalyzer()
    analyzer = QueryAnalyzer(embedding_analyzer)
    analyzer.run()

if __name__ == "__main__":
    main()

from pathlib import Path
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModel
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.decomposition import TruncatedSVD
import yake
import ast
import time
from google.api_core.exceptions import ResourceExhausted
import glob
import re

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Define paths with the user-specified locations

# Directory where your JSON files are stored
directory = '.'  # Replace with your actual path

# Pattern to match filenames and extract timestamp
pattern = re.compile(r"query_results_.*_(\d{8}_\d{6})\.json")

# Find all JSON files matching the pattern
json_files = glob.glob(os.path.join(directory, "query_results_*.json"))

# Extract timestamps and find the most recent file
most_recent_file = None
latest_timestamp = None

for file in json_files:
    match = pattern.search(os.path.basename(file))
    if match:
        timestamp = match.group(1)
        # Convert timestamp to an integer for comparison
        timestamp_int = int(timestamp)
        # Update the most recent file if the timestamp is newer
        if latest_timestamp is None or timestamp_int > latest_timestamp:
            latest_timestamp = timestamp_int
            most_recent_file = file

# Store the most recent path in query_results_path
query_results_path = most_recent_file
print("Most recent file:", query_results_path)

papers_dir = "./papers"

class PaperEmbeddingAnalyzer:
    def __init__(self):
        # Initialize SciBERT tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.keyword_extractor = yake.KeywordExtractor()
        self.topic_model = None
        self.fallback_mode = False

    def _initialize_topic_model(self, n_docs):
        """Initialize topic model based on dataset size"""
        if n_docs < 5:  # For very small datasets
            self.fallback_mode = True
            # Use SVD instead of UMAP for small datasets
            dim_reducer = TruncatedSVD(n_components=min(2, n_docs - 1))
            
            hdbscan_model = HDBSCAN(
                min_cluster_size=2,
                metric='euclidean',
                prediction_data=True
            )
            
            self.topic_model = BERTopic(
                umap_model=dim_reducer,
                hdbscan_model=hdbscan_model,
                nr_topics=min(2, n_docs),
                verbose=True
            )
        else:
            self.fallback_mode = False
            umap_model = UMAP(
                n_neighbors=min(2, n_docs - 1),
                n_components=min(2, n_docs - 1),
                min_dist=0.0,
                metric='cosine'
            )
            
            hdbscan_model = HDBSCAN(
                min_cluster_size=2,
                metric='euclidean',
                prediction_data=True
            )
            
            self.topic_model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                nr_topics="auto",
                verbose=True
            )
    
    def embed_text(self, text):
        """Generate embeddings for a given text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def extract_keywords(self, text, top_k=5):
        """Extract key phrases from text using YAKE."""
        if not text or len(text.strip()) == 0:
            return ["no_keywords_found"]
        try:
            keywords = self.keyword_extractor.extract_keywords(text)
            keywords = sorted(keywords, key=lambda x: x[1])[:top_k]
            return [kw[0] for kw in keywords]
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return ["keyword_extraction_failed"]

    def analyze_paper(self, title, abstract, findings):
        """Generate a composite embedding by combining title, abstract, and findings embeddings."""
        try:
            title_emb = self.embed_text(title)
            abstract_emb = self.embed_text(abstract)
            findings_emb = self.embed_text(findings)
            
            # Composite embedding using weighted averages
            combined_embedding = torch.mean(torch.stack([title_emb * 1.5, abstract_emb, findings_emb]), dim=0)
            return combined_embedding
        
        except Exception as e:
            print(f"Error in analyze_paper: {e}")
            # Return a zero embedding as fallback
            return torch.zeros((1, 768))

    def fit_topic_model(self, documents):
        """Fit topic model on documents with fallback for small datasets"""
        try:
            if not documents or len(documents) == 0:
                print("No documents provided for topic modeling.")
                return

            print(f"Fitting topic model on {len(documents)} documents")
            
            # Initialize appropriate model based on dataset size
            self._initialize_topic_model(len(documents))
            
            # For very small datasets, use simple topic assignment
            if self.fallback_mode:
                print("Using fallback mode for small dataset")
                self.topic_model.fit_transform(documents)
                print("Topic modeling completed in fallback mode")
            else:
                self.topic_model.fit_transform(documents)
                print("Topic modeling completed successfully")
                
        except Exception as e:
            print(f"Error during topic modeling: {e}")
            self.fallback_mode = True
            print("Falling back to simple topic assignment")
            # Create a simple fallback topic assignment
            self.topic_model = None

    def get_topics_for_paper(self, text):
        """Get topics for a single paper with fallback handling"""
        try:
            if self.fallback_mode or self.topic_model is None:
                return [("General Topic", 1.0)]
                
            topics, _ = self.topic_model.transform([text])
            return self.topic_model.get_topic(topics[0]) if topics[0] != -1 else [("General Topic", 1.0)]
        except Exception as e:
            print(f"Error getting topics: {e}")
            return [("General Topic", 1.0)]

#NOTA: el JSONL se genera con el nombre question results###

class NuancedQuestions:
    def __init__(self, embedding_analyzer):
        # Configure the Gemini API
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.embedding_analyzer = embedding_analyzer
        self.PROJECT_DIR = Path(".")
        self.output_file = self.PROJECT_DIR / f"question_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

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
        pdf_path = f"./papers/{paper_id}.pdf"
        
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
            f"Format the output as a string that has the python list of strings format and looks like this [question 1, question 2, ...] in which each element only contains the question, no enumeration. Make sure that the output is only a text that looks like a python list"
        )
        
        response = self._call_with_retry(lambda: self.model.generate_content(
            prompt,
            generation_config={"temperature": 0.7, "max_output_tokens": 300}
        ))

        if response and response.text.strip():
            respone_clean = response.text.strip("```python\n").strip("```")
            questions_list = ast.literal_eval(respone_clean)
            formatted_questions = [
                 f"\"{q}\"" for q in questions_list
            ]
            final_result = f"[{', '.join(formatted_questions)}] \n"


            return final_result
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

        # Collect all paper texts for topic modeling
        all_texts = []
        for paper_id in relevant_papers_ids:
            paper_text = self.extract_text_from_pdf(paper_id)
            if paper_text:
                all_texts.append(paper_text)
        
        # Fit BERTopic on all documents
        self.embedding_analyzer.fit_topic_model(all_texts)

        # Process each paper to generate questions
        for paper_id in relevant_papers_ids:
            paper_text = self.extract_text_from_pdf(paper_id)
            if paper_text:
                # Extract sections and generate embeddings
                title, abstract, findings = self.extract_sections(paper_text)
                keywords = self.embedding_analyzer.extract_keywords(paper_text)
                paper_embedding = self.embedding_analyzer.analyze_paper(title, abstract, findings)
                
                # Extract topic for this paper
                topics = self.embedding_analyzer.get_topics_for_paper(paper_text)
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


import json
import os
import re
import datetime
import requests
from Chunkenizer import Chunkenizer
from Embbedingator import Embbedingator
from PerformQuery import PerformQuery


class Coordinator:
    def __init__(self, user_inputs_file="user_inputs.json"):
        """
        Initialize the Coordinator with user inputs and pipeline components.
        Args:
            user_inputs_file (str): Path to the JSON file containing user inputs.
        """
        # Load user inputs
        with open(user_inputs_file, "r") as f:
            self.user_inputs = json.load(f)

        self.query = self.user_inputs["query"]
        self.papers_folder = self.user_inputs["papers_folder"]
        self.local_papers = self.user_inputs["local_papers"]
        self.option = self.user_inputs["option"]
        self.genie_api_url = "https://search.genie.stanford.edu/semantic_scholar"

        # Initialize components
        self.chunkenizer = Chunkenizer(self.papers_folder)
        self.embbedingator = Embbedingator()
        self.perform_query = PerformQuery()

    def process_local_papers(self):
        """
        Process and chunk local papers selected by the user.
        Returns:
            list: List of chunks from local papers.
        """
        chunks = []
        for paper in self.local_papers:
            file_path = os.path.join(self.papers_folder, paper)
            paper_chunks = self.chunkenizer.process_file(file_path)
            for chunk in paper_chunks:
                chunks.append({"source": paper, "content": chunk})
        return chunks

    def fetch_external_papers(self):
        """
        Fetch external papers from Genie API.
        Returns:
            list: List of external paper data including content.
        """
        print("Retrieving external papers from Genie API...")
        payload = {
            "query": [self.query],
            "num_blocks": 10,  # Number of results to retrieve
            "rerank": True,  # Use LLM reranking
            "num_blocks_to_rerank": 10
        }

        try:
            response = requests.post(self.genie_api_url, json=payload)
            response.raise_for_status()
            data = response.json()
            external_papers = []

            for result in data[0]["results"]:  # Adjusting to match the API response structure
                external_papers.append({
                    "title": result["document_title"],
                    "content": result["content"],
                    "url": result["url"]
                })

            print(f"Retrieved {len(external_papers)} papers from Genie API.")
            return external_papers
        except requests.exceptions.RequestException as e:
            print(f"Error fetching external papers: {e}")
            return []

    def process_external_papers(self, external_papers):
        """
        Chunk external papers retrieved from Genie API.
        Args:
            external_papers (list): List of external paper contents.

        Returns:
            list: List of chunks from external papers.
        """
        chunks = []
        for paper in external_papers:
            paper_chunks = self.chunkenizer.chunk_text(paper["content"])
            for chunk in paper_chunks:
                chunks.append({
                    "source": paper["title"],
                    "content": chunk,
                    "url": paper["url"]
                })
        return chunks

    def calculate_similarities(self, chunks):
        """
        Calculate similarity scores between the query and each chunk.
        Args:
            chunks (list): List of chunks to compare.

        Returns:
            list: List of chunks with similarity scores.
        """
        query_embedding = self.embbedingator.embed_text(self.query)
        results = []
        for chunk in chunks:
            chunk_embedding = self.embbedingator.embed_text(chunk["content"])
            similarity = self.perform_query.calculate_similarity(query_embedding, chunk_embedding)
            results.append({
                "source": chunk["source"],
                "content": chunk["content"],
                "similarity": float(similarity),  # Ensure JSON serialization compatibility
                "url": chunk.get("url")  # Include URL if available
            })
        return sorted(results, key=lambda x: x["similarity"], reverse=True)

    def save_results(self, results, report_name, include_url=False):
        """
        Save results to a JSONL report file.
        Args:
            results (list): List of similarity results.
            report_name (str): Name of the report file.
            include_url (bool): Whether to include URL in the report (for external papers).

        Returns:
            str: Path to the saved report.
        """
        sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", report_name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"./{sanitized_name}_{timestamp}.jsonl"

        with open(report_path, "w") as f:
            for result in results:
                # Prepare the JSON line
                report_line = {
                    "Source": result["source"],
                    "Content": result["content"],
                    "Similarity Score": result["similarity"]
                }
                # Add URL for external papers
                if include_url and result.get("url"):
                    report_line["URL"] = result["url"]

                json.dump(report_line, f)
                f.write("\n")

        print(f"Report saved to {report_path}")
        return report_path

    def run_pipeline(self):
        """
        Execute the pipeline based on user inputs.
        """
        print("Processing local papers...")
        local_chunks = self.process_local_papers()

        print("Calculating similarities for local papers...")
        local_results = self.calculate_similarities(local_chunks)
        self.save_results(local_results, "local_papers_report")

        if self.option == "2":
            print("Fetching and processing external papers...")
            external_papers = self.fetch_external_papers()
            external_chunks = self.process_external_papers(external_papers)

            print("Calculating similarities for external papers...")
            external_results = self.calculate_similarities(external_chunks)
            self.save_results(external_results, "external_papers_report", include_url=True)

            print("Generating combined report...")
            combined_results = local_results + external_results
            self.save_results(combined_results, "combined_report", include_url=True)


if __name__ == "__main__":
    # Run the pipeline
    coordinator = Coordinator()
    coordinator.run_pipeline()

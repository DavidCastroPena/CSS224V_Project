from pathlib import Path
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

class QueryAnalyzer:
    def __init__(self):
        #Initialize the OpenAI client
        self.client = OpenAI(api_key=api_key)
        if not self.client.api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable")
        
        #Use the exact path where query_results exists
        self.PROJECT_DIR = Path(r"..")

    def debug_directory_contents(self):
        """Debug function to list directory contents"""
        print("\nDebugging Directory Contents:")
        print(f"Checking directory: {self.PROJECT_DIR}")
        print("Files in directory:")
        for item in self.PROJECT_DIR.iterdir():
            print(f"- {item.name}")
            if item.is_dir():
                print("  Subdirectory contents:")
                try:
                    for subitem in item.iterdir():
                        print(f"  - {subitem.name}")
                except Exception as e:
                    print(f"  Error reading subdirectory: {e}")

    def get_latest_query_results_file(self):
        """Find the most recent query results file."""
        try:
            # First try looking in the main directory
            files = list(self.PROJECT_DIR.glob("query_results*.json"))
            
            if not files:
                # If not found, try looking in a query_results subdirectory
                query_results_dir = self.PROJECT_DIR / "query_results"
                if query_results_dir.exists():
                    files = list(query_results_dir.glob("*.json"))
            
            if not files:
                self.debug_directory_contents()
                print("No query results JSON files found.")
                return None
            
            # Sort files by modification time in descending order
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            print(f"Found latest file: {latest_file}")
            return latest_file
        except Exception as e:
            print(f"Error accessing directory: {e}")
            self.debug_directory_contents()
            return None

    def load_query_results(self, filename):
        """Load query results from a JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                query_results = json.load(file)
            print(f"Successfully loaded query results from {filename}")
            print(f"Number of results loaded: {len(query_results)}")
            return query_results
        except Exception as e:
            print(f"Error loading query results: {e}")
            return None

    def generate_comparison_questions(self, query_results):
        """Generate comparison questions using OpenAI API."""
        if not query_results:
            return "No query results available to generate questions."

        try:
            # Prepare the context from the loaded query results
            context = "\n\n".join([
                f"Paper ID: {result['PaperID']}\nChunk: {result['ChunkText']}" 
                for result in query_results
            ])

            # Define the prompt for OpenAI
            prompt = f"""
            Based on the following excerpts from studies, generate a schema of comparison questions that would help compare findings, methodologies, and conclusions of these studies for a policymaker interested in evaluating interventions.
            """

            # Call OpenAI with the new API format
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": context}
                ],
                max_tokens=300,
                temperature=0.5
            )

            # Extract the generated questions from the response using the new format
            questions = response.choices[0].message.content.strip()
            return questions
        except Exception as e:
            return f"Error generating questions: {e}"

    def run(self):
        """Main function to load results and generate questions."""
        print(f"Starting script...")
        print(f"Looking for query results in: {self.PROJECT_DIR}")
        
        # Find the latest query results file
        latest_file = self.get_latest_query_results_file()
        
        if not latest_file:
            print("No query results available.")
            return
        
        print(f"Using latest query results file: {latest_file}")

        # Load the saved query results
        query_results = self.load_query_results(latest_file)
        if not query_results:
            return

        # Generate comparison questions
        comparison_questions = self.generate_comparison_questions(query_results)

        # Save the generated questions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        questions_file = self.PROJECT_DIR / f"comparison_questions_{timestamp}.txt"
        
        try:
            with open(questions_file, 'w', encoding='utf-8') as f:
                f.write(comparison_questions)
            print(f"\nGenerated questions saved to: {questions_file}")
            print("\nGenerated Comparison Questions:\n", comparison_questions)
        except Exception as e:
            print(f"Error saving questions: {e}")
            print("\nGenerated Comparison Questions:\n", comparison_questions)

def main():
    analyzer = QueryAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()
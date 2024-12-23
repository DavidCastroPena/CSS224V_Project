from pathlib import Path
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime
import ast



load_dotenv()

open_ai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")


class NaiveQuestions:
    def __init__(self):
        #Initialize the OpenAI client
        self.client = OpenAI(api_key=open_ai_api_key)
        if not self.client.api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable")
        
        #Use the exact path where query_results exists
        self.PROJECT_DIR = Path(r".")

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
            files = list(self.PROJECT_DIR.glob("combined_report_*.jsonl"))
            
            if not files:
                # If not found, try looking in a query_results subdirectory
                query_results_dir = self.PROJECT_DIR / "combined_report"
                if query_results_dir.exists():
                    files = list(query_results_dir.glob("*.jsonl"))
            
            if not files:
                self.debug_directory_contents()
                print("No combined report JSONl files found.")
                return None
            
            # Sort files by modification time in descending order
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            print(f"Found latest file: {latest_file}")
            return latest_file
        
        except Exception as e:
            print(f"Error accessing directory: {e}")
            self.debug_directory_contents()
            return None

    def load_relevant_papers(self, filename):
        """Load query results from a JSONL file and extract unique sources."""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                # Read the JSONL file line by line
                query_results = [json.loads(line) for line in file]
            
            print(f"Successfully loaded query results from {filename}")
            
            # Extract unique sources
            unique_sources = {entry["Source"] for entry in query_results}
            print(f"Number of relevant sources found: {len(unique_sources)}")
            
            return unique_sources

        except Exception as e:
            print(f"Error loading query results: {e}")
            return None
    

    def generate_comparison_questions(self, user_query, question_number, relevant_papers_ids):
        """Generate comparison questions using OpenAI API."""
        if not relevant_papers_ids:
            return "No query results available to generate questions."

        try:
            # Extract topic
            prompt = "What is the general topic of this query? give it in 3 words: {}".format(user_query)

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt}#,
                ],
                max_tokens=30,
                temperature=0.5
            )

            topic = response.choices[0].message.content.strip()

            # Prompt
            prompt = "Create {} questions that can be allow a thorough comparison of findings, methodologies, and conclusions among policy and econometric papers on the topic of '{}'. Note that the questions will be individually asked to each paper. Format the output as a python list of strings that looks like this [question 1, question 2, ...] in which each element only contains the question, no enumeration. Make sure that the output is a python list".format(question_number, topic)
            
            # Call OpenAI with the new API format
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt}#,
                    #{"role": "user", "content": context}
                ],
                max_tokens=300,
                temperature=0.5
            )

            # Extract the generated questions from the response using the new format
            questions = response.choices[0].message.content.strip()
            return questions
        
        except Exception as e:
            return f"Error generating questions: {e}"


    def run(self, user_query):
        """Main function to load results and generate questions."""
        print(f"\nStarting script to generate naive questions...")
        print(f"\nLooking for query results in: {self.PROJECT_DIR}")
        
        # Find the latest query results file
        latest_file = self.get_latest_query_results_file()
        
        if not latest_file:
            print("\nNo combined report available.")
            return
        
        print(f"\nUsing latest combined report file to generate naive questions: {latest_file}")

        # Load set of relevant papers
        relevant_papers_ids = self.load_relevant_papers(latest_file)
        if not relevant_papers_ids:
            return
        
        
        # Generate comparison questions
        print("\nGenerating and saving naive comparison questions for this topic...")
        comparison_questions = self.generate_comparison_questions(user_query=user_query, question_number=3, relevant_papers_ids=relevant_papers_ids)
        
        # Add questions related to time and place
        comparison_questions = ast.literal_eval(comparison_questions)
        comparison_questions.insert(0, "When was this paper writen?")
        comparison_questions.insert(0, "What is the main place where this paper is refering to?")
        comparison_questions.insert(0, "What is the title of the paper?")

        comparison_questions = str(comparison_questions)

        # Save the generated questions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        questions_file = self.PROJECT_DIR / f"comparison_questions_{timestamp}.txt"

        try:
            with open(questions_file, 'w', encoding='utf-8') as f:
                f.write(comparison_questions)
            print(f"\nGenerated questions saved to: {questions_file}")
            print("\nGenerated Comparison Questions:\n", comparison_questions)
        except Exception as e:
            print(f"\nError saving questions: {e}")

        return relevant_papers_ids
        


if __name__ == "__main__":
    naive_questions = NaiveQuestions()
    user_query = "I am the mayor of SF and I want to create a policy that fosters financial inclusion on the mission district. I want to implement this from a gender perspective focused on women that are substance users"
    relevant_papers_ids = naive_questions.run(user_query=user_query)

from pathlib import Path
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai
import PyPDF2
import ast


load_dotenv()

open_ai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")


class QueryAnalyzer:
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
    

    def generate_comparison_questions(self, topic, question_number, relevant_papers_ids):
        """Generate comparison questions using OpenAI API."""
        if not relevant_papers_ids:
            return "No query results available to generate questions."

        try:
            prompt = "Create {} questions that can be allow a thorough comparison of findings, methodologies, and conclusions among policy and econometric papers on the topic of {}. Note that the questions will be individually asked to each paper. Format the output as a python list of strings that looks like this [question 1, question 2, ...] in which each element only contains the question, no enumeration. Make sure that the output is a python list".format(question_number, topic)

            
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
        
    def clean_json_string(self, json_string):
        retString = json_string.rstrip()
        if retString.endswith(")}"):
            retString = retString[:-2] + retString[-1]
        return retString
        
    def answer_question_gemini(self, questions, paper_text):
        
        genai.configure(api_key=gemini_api_key)
        # GEMINI SET UP
        generation_config = {
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }

        # Generating the formatted list of questions
        formatted_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

        # Creating the JSON schema
        schema = "paper_json = {\n"
        for question in questions:
            question_key = question.lower().replace(" ", "_").replace("?", "")
            schema += f'    "{question_key}": {{"type": "string"}},\n'
        schema = schema.rstrip(",\n") + "\n}"

        # Creating the prompt
        prompt = f"""STEP 1 - Answer the following questions based on the provided paper below, respond using only the provided text and keep your answers concise and detailed.

        {formatted_questions}

        STEP 2 - Using this JSON schema, return a JSON with the answers to the questions previously retrieved:
        {schema}

        If the provided paper contains no data to respond to a question, leave the field as an empty string and don't make up any data.
        """

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            system_instruction=prompt,
        )
        response = model.generate_content(paper_text)

        response_cleaned = self.clean_json_string(response.text)

        # Use json.loads to convert the cleaned string into a dictionary
        try:
            response_json = json.loads(response_cleaned)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return {}

        return response_json
                

    def run(self):
        """Main function to load results and generate questions."""
        print(f"\nStarting script...")
        print(f"\nLooking for query results in: {self.PROJECT_DIR}")
        
        # Find the latest query results file
        latest_file = self.get_latest_query_results_file()
        
        if not latest_file:
            print("\nNo query results available.")
            return
        
        print(f"\nUsing latest query results file: {latest_file}")

        # Load set of relevant papers
        relevant_papers_ids = self.load_relevant_papers(latest_file)
        if not relevant_papers_ids:
            return
        
        
        # Generate comparison questions
        print("\nGenerating and saving comparison questions for this topic...")
        comparison_questions = self.generate_comparison_questions(topic = "banking challenges", question_number=3, relevant_papers_ids=relevant_papers_ids)
        
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

        # Questions string to list
        comparison_questions = ast.literal_eval(comparison_questions)
        
        final_json = {}

        # Answer questions for each paper
        for paper_id in relevant_papers_ids: 
            print("\nTransforming paper {} in pdf to text ...".format(paper_id))
            paper_text = self.extract_text_from_pdf(paper_id)
            print("Answering questions for {}".format(paper_id))
            
            # Call the function to get the answers for the questions
            answers = self.answer_question_gemini(comparison_questions, paper_text)
            
            # Add the answers to the final JSON dictionary under the paper_id
            final_json[paper_id] = answers

        # Specify the filename and path to save the JSON
        output_path = os.path.join(os.getcwd(), "paper_answers.json")

        # Save the final JSON to the current directory
        with open(output_path, "w") as json_file:
            json.dump(final_json, json_file, indent=4)

        print(f"Output JSON saved at {output_path}")

        


def main():
    analyzer = QueryAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()
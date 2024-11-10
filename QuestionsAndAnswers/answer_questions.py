import os
import json
import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv
import ast
import glob
from pathlib import Path  # Make sure Path is imported
from naiveQuestions import NaiveQuestions
from nuancedQuestions import PaperEmbeddingAnalyzer, NuancedQuestions

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

class QuestionAnswerer:
    def __init__(self):
        self.questions_list = []   
        self.relevant_papers_ids = [] 

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
        prompt = f"""STEP 1 - Answer the following questions based on the provided paper below, respond using only the provided text and keep your answers concise and detailed. Express dates in the format Month, Year.

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
    
    def retrieve_naive(self): 
        print("Calling retrieve naive function")
        # Run Naive Question class which identifies relevant papers and creates naive questions
        naive_questions = NaiveQuestions()
        self.relevant_papers_ids = naive_questions.run()

        # Find latest comparison question file and import it
        try:
            files = list(Path(".").glob("comparison_questions_*.txt"))
            
            if not files:
                print("No comparison questions files found.")
                return None
            
            # Sort files by modification time in descending order
            latest_questions_file = max(files, key=lambda f: f.stat().st_mtime)
            print(f"Found latest file: {latest_questions_file}")
        
        except Exception as e:
            print(f"Error accessing directory: {e}")
            return None
        
        # Transform txt into python list of questions
        try:
            with open(latest_questions_file, 'r', encoding='utf-8') as f:
                return  ast.literal_eval(f.read())
        except Exception as e:
            print(f"\nError reading questions file: {e}")
            return
        

    def generate_nuanced(self):
        # This creates a file with nuanced for all relevant pappers 
        print("Generating nuanced questions.... ")
        embedding_analyzer = PaperEmbeddingAnalyzer()
        analyzer = NuancedQuestions(embedding_analyzer)
        analyzer.run()
        return
    
    def retrieve_nuanced(self, paper_id):
        pattern = "question_results_*.jsonl"
        files = glob.glob(pattern)
        
        if not files:
            raise FileNotFoundError("No question results files found")
        
        # Get the most recent file based on filename
        latest_file = max(files)
        
        # Read the file and find the matching paper_id
        with open(latest_file, 'r') as file:
            for line in file:
                entry = json.loads(line.strip())
                if entry['paper_id'] == paper_id:
                    # Convert string representation of list to actual list
                    questions_str = entry['questions'][0]
                    # Parse the string to get the actual list
                    questions_list = json.loads(questions_str)
                    return questions_list
                    
        raise ValueError(f"Paper ID '{paper_id}' not found in {latest_file}")

    def run(self):
        print(f"\nStarting question answering script with naive and nuanced questions...")
        final_json = {}

        # Update questions list
        self.questions_list = self.retrieve_naive()

        self.generate_nuanced()

        # Answer questions for each paper
        for paper_id in self.relevant_papers_ids: 
            print("\nTransforming paper {} in pdf to text ...".format(paper_id))
            paper_text = self.extract_text_from_pdf(paper_id)
            print("Answering questions for {}".format(paper_id))
            
            # Retrieve nuanced questions
            nuanced = self.retrieve_nuanced(paper_id)

            all_questions = self.questions_list + nuanced

            # Call the function to get the answers for the questions
            answers = self.answer_question_gemini(all_questions, paper_text)
            
            # Add the answers to the final JSON dictionary under the paper_id
            final_json[paper_id] = answers

        # Specify the filename and path to save the JSON
        output_path = os.path.join(os.getcwd(), "paper_answers.json")

        # Save the final JSON to the current directory
        with open(output_path, "w") as json_file:
            json.dump(final_json, json_file, indent=4)

        print(f"Output JSON saved at {output_path}")

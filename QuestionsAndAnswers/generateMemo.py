from answer_questions import QuestionAnswerer
import os
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai
import json



load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

class GenerateMemo:
    def __init__(self, answer_list):
        self.answer_list = answer_list   

    def generate_memo(self, query):

        print("Prompting Gemini to generate Memo...")

        genai.configure(api_key=gemini_api_key)
        
        # GEMINI SET UP
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }

        # Creating the prompt
        prompt = f"""You will receive a JSON like txt that contains information about different academic papers related to particular topic, with each paper represented as a nested object containing the some common questions and some unique questions that vary based on the paper's specific focus. 

        Using information only from this JSON file provided, generate a policy memo that helps give clarity about this question from the user: {query} , and has the following structure: 

        - First present a table in which the first column is what_is_the_title_of_the_paper and the other columns correspond to the rest of the questions that are common to all papers. Each key in the txt provided corresponds to a paper and each one should be one row in the table, with the correspondent answers. Make sure the number of keys in the JSON file is equal to the number of rows in the table. 
        - After the table, include a bullet point list of each paper with its title, giving details found in the questions that are specific to each paper (the last entries in the JSON). 

        The output should be a like a 1-pager in .md format and the format of the table should be like PrettyTable
        """

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            system_instruction=prompt,
        )
        response = model.generate_content(self.answer_list)

        response = response.text

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        memo_file = f"memo_{timestamp}.md"

        # Write the response to the file
        with open(memo_file, "w") as file:
            file.write(response)
        
        print(f"Memo saved to {memo_file}")

        return 


def main():

    answer = QuestionAnswerer()
    answer.run()

    query_text = "I want to create a policy that fosters financial inclusion and economic growth in California. "
    with open('paper_answers.json', 'r') as file:
        answer_list = json.load(file)
    
    answer_list = str(answer_list)
    
    memo = GenerateMemo(answer_list)
    memo.generate_memo(query_text)



    


if __name__ == "__main__":
    main()
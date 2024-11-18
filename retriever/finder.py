import os
import json

class Finder:
    def __init__(self, output_file="user_inputs.json"):
        """
        Initialize Finder to gather user inputs and save them in a structured format.
        Args:
            output_file (str): The file where user inputs will be saved.
        """
        self.query = None
        self.option = None
        self.papers_folder = None
        self.local_papers = []
        self.output_file = output_file

    def welcome_user(self):
        """Welcome the user and collect their policy situation (query)."""
        print("Hello, I am PolicyChat! What about you telling me about your policy situation?")
        self.query = input("Enter your policy situation: ").strip()

    def ask_analysis_option(self):
        """Ask the user to choose an analysis option."""
        print("\nChoose an analysis option:")
        print("1. Analyze up to 5 local papers.")
        print("2. Analyze up to 5 local papers and 10 additional papers retrieved from the web.")
        self.option = input("Enter 1 or 2: ").strip()
        if self.option not in {"1", "2"}:
            raise ValueError("Invalid option. Please choose either 1 or 2.")

    def set_papers_folder(self):
        """Ask the user for the location of their papers folder."""
        self.papers_folder = input("\nPlease provide the full path to your papers folder: ").strip()
        if not os.path.exists(self.papers_folder):
            raise FileNotFoundError(f"The folder '{self.papers_folder}' does not exist.")
        print(f"Papers folder set to: {self.papers_folder}")

    def list_local_papers(self):
        """List all valid local papers in the user's folder."""
        files = [f for f in os.listdir(self.papers_folder) if f.endswith('.pdf') or f.endswith('.txt')]
        if not files:
            raise FileNotFoundError("No valid papers found in the papers directory.")
        return files

    def select_local_papers(self):
        """Prompt the user to select up to 5 local papers."""
        print("\nAvailable Local Papers:")
        files = self.list_local_papers()
        for i, file in enumerate(files):
            print(f"{i+1}. {file}")

        indices = input("Select up to 5 papers by their numbers (comma-separated): ")
        indices = [int(i) - 1 for i in indices.split(",")]
        self.local_papers = [os.path.join(self.papers_folder, files[i]) for i in indices]
        print(f"\nSelected Papers: {self.local_papers}")

    def save_user_inputs(self):
        """Save the gathered inputs to a JSON file."""
        user_inputs = {
            "query": self.query,
            "option": self.option,
            "papers_folder": self.papers_folder,
            "local_papers": self.local_papers
        }
        with open(self.output_file, 'w') as f:
            json.dump(user_inputs, f, indent=4)
        print(f"\nUser inputs saved to {self.output_file}")

    def run(self):
        """Run the entire user input collection process."""
        self.welcome_user()
        self.ask_analysis_option()
        self.set_papers_folder()
        self.select_local_papers()
        self.save_user_inputs()


if __name__ == "__main__":
    # Run the Finder to collect user information
    finder = Finder()
    finder.run()

import os
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunkenizer:
    def __init__(self, papers_folder):
        """
        Initialize the Chunkenizer with the specified papers folder.
        Args:
            papers_folder (str): Path to the folder containing the papers.
        """
        if not os.path.exists(papers_folder):
            raise FileNotFoundError(f"The provided papers folder '{papers_folder}' does not exist.")
        self.papers_folder = papers_folder
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=50,
            separators=[".", ",", " ", ""]
        )

    def process_file(self, file_path):
        """
        Process a file to extract and chunk its content.
        Args:
            file_path (str): Path to the file.

        Returns:
            list: A list of text chunks from the file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        extension = os.path.splitext(file_path)[1].lower()
        if extension == ".pdf":
            return self._process_pdf(file_path)
        elif extension == ".txt":
            return self._process_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

    def _process_pdf(self, pdf_path):
        """
        Extract and chunk text from a PDF file.
        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            list: A list of text chunks from the PDF.
        """
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        return self.splitter.split_text(text)

    def _process_txt(self, txt_path):
        """
        Extract and chunk text from a TXT file.
        Args:
            txt_path (str): Path to the text file.

        Returns:
            list: A list of text chunks from the TXT file.
        """
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.splitter.split_text(text)

    def chunk_text(self, text):
        """
        Directly chunk a given text.
        Args:
            text (str): The text to be chunked.

        Returns:
            list: A list of text chunks.
        """
        return self.splitter.split_text(text)


if __name__ == "__main__":
    print("Testing Chunkenizer functionality...")

    # Example standalone usage
    folder = input("Enter the path to your papers folder: ").strip()
    chunkenizer = Chunkenizer(folder)

    file_name = input("Enter the name of a file in the folder: ").strip()
    file_path = os.path.join(folder, file_name)

    try:
        chunks = chunkenizer.process_file(file_path)
        print(f"Generated {len(chunks)} chunks for '{file_name}'.")
        for i, chunk in enumerate(chunks[:5]):  # Display first 5 chunks as a sample
            print(f"Chunk {i+1}: {chunk[:100]}...")
    except Exception as e:
        print(f"Error: {e}")

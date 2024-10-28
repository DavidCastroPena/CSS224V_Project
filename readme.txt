Instructions to Run the Retriever for PolicyChat
This document provides step-by-step instructions to run the retriever for PolicyChat, a tool designed to assist policymakers in finding evidence-based policies. PolicyChat uses a retriever system based on Qdrant to store and search embeddings of research papers.


0- Update reqs

cd "/c/Users/engin/OneDrive/Desktop/Carpetas/STANFORD/fourthQuarter/CS224V/Project"
pipreqs "C:\Users\engin\OneDrive\Desktop\Carpetas\STANFORD\fourthQuarter\CS224V\Project"
pipreqs .




1. Setup Docker and Qdrant
Ensure Docker is installed and running on your machine. Verify Docker by running:

bash
Copy code
docker --version
Run the following command to start the Qdrant server as a Docker container:

bash
Copy code
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
Check if Qdrant is running by using:

bash
C:\Users\engin\OneDrive\Desktop\Carpetas\STANFORD\fourthQuarter\CS224V\Project\retriever\Chunkenizer.py

 cd  C:/Users/engin/OneDrive/Desktop/Carpetas/STANFORD/fourthQuarter/CS224V/Project

Copy code
docker ps
If the container already exists, you can restart it with:

bash
Copy code
docker start qdrant
2. Activate the Anaconda Environment
Ensure that you're in the correct environment where the dependencies are installed:

bash
Copy code
conda activate wikichat
3. Chunk the Papers
Use the Chunkenizer.py script to break down your research papers into smaller chunks (less than 500 tokens) for processing:

bash
Copy code
python C:/Users/engin/OneDrive/Desktop/Carpetas/STANFORD/fourthQuarter/CS224V/Project/retriever/Chunkenizer.py
4. Create the Qdrant Collection (if necessary)
Ensure that the Qdrant collection exists by running the following script:

bash
Copy code
python C:/Users/engin/OneDrive/Desktop/Carpetas/STANFORD/fourthQuarter/CS224V/Project/retriever/QdrantCollection.py
5. Embed the Chunks
Generate embeddings for each chunk and store them in the Qdrant collection by running the Embbedingator.py script:

bash
Copy code
python C:/Users/engin/OneDrive/Desktop/Carpetas/STANFORD/fourthQuarter/CS224V/Project/retriever/Embbedingator.py
6. Query the Indexed Embeddings
After successfully indexing the embeddings, you can query them by using the PerformQuery.py script:

bash
Copy code
python C:/Users/engin/OneDrive/Desktop/Carpetas/STANFORD/fourthQuarter/CS224V/Project/retriever/PerformQuery.py
Notes and Troubleshooting
If you encounter errors or issues, refer to the following notes:

Docker Container Already Running: If you encounter an error about an existing Qdrant container, stop or remove it using:
bash
Copy code
docker rm -f qdrant
File Not Found: Double-check file paths for typos.
Python Environment: Ensure you are in the correct environment where all dependencies are installed.
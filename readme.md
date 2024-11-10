# Instructions to Run the Retriever for PolicyChat
This document provides step-by-step instructions to run the retriever for PolicyChat, a tool designed to assist policymakers in finding evidence-based policies. PolicyChat uses a retriever system based on Qdrant to store and search embeddings of research papers.


### 0. Create env and install requirements
Navigate to the repository and create an environment by: 
```
python3 -m venv myenv
source myenv/bin/activate
```

Next, install requirements: 
```
pipreqs .
pip install -r requirements.txt
```

### 1. Setup Docker and Qdrant
Ensure Docker is installed and running: 
```
docker --version
```

If this is the first time you are running this, run the following command to start the Qdrant server as a Docker container:
```
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

Check if Qdrant is running by using:
```
docker start qdrant
```

### 2. Chunk the Papers
Use the Chunkenizer.py script to break down your research papers into smaller chunks (ideally less than 500 tokens) for processing. This will take into account all the papers stored in the papers directory. 
```
python retriever/Chunkenizer.py
```

### 3. Create the Qdrant Collection (if necessary)
Ensure that the Qdrant collection exists by running the following script:
```
python retriever/QdrantCollection.py
```

### 4. Embed the Chunks
Generate embeddings for each chunk and store them in the Qdrant collection by running the Embbedingator.py script:
```
python retriever/Embbedingator.py
```

### 5. Query the Indexed Embeddings
After successfully indexing the embeddings, you can query them by using the PerformQuery.py script:
```
python retriever/PerformQuery.py
```

### 6. Excecute the generateMemo script
After we performed the query, we are ready to execute the generateMemo script. This script does the following:

- Retrieves the relevant papers for the query.
- Extracts text from the corresponding PDF files for each relevant paper.
- Retrieves "naive" or general questions from the most recent file containing comparison questions.
- Generates "nuanced" questions for each relevant paper using embedding analysis.
- Sends the extracted paper text and both naive and nuanced questions to the Gemini AI API to generate answers.
- Collects the answers in JSON format, associating them with their respective paper IDs.
- Saves the final answers in a paper_answers.json file.
- Prompts gemini to generate memo in .md format

```
python generateMemo.py
```

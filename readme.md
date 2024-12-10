# Instructions to Run the Retriever for PolicyChat
This document provides step-by-step instructions to run the chatbot application PolicyChat, a tool designed to assist policymakers in finding evidence-based policies. PolicyChat is an AI-powered tool designed to bridge this gap by providing a scalable and efficient solution for evidence retrieval, comparison and summarization. Built on state-of-the-art natural language processing (NLP) technologies, PolicyChat combines embedding-based retrieval, topic modeling, and grounded generative AI to streamline the policymaking workflow. The system retrieves and ranks relevant papers based on semantic similarity with respect to a user query, generates tailored questions to extract a comparison schema across relevant papers, and synthesizes the results into structured memos for policymakers.


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

### 2. Run the Streamlit application 
```
streamlit run ux.py
```

# PolicyChat - README

**PolicyChat** is a conversational assistant aimed at helping policymakers quickly find and interpret **evidence-based policies**. This chatbot provides recommendations based on policies that have been tested in the real world through **Randomized Controlled Trials (RCTs)** or quasi-experiments. Policymakers can use PolicyChat to receive expert advice on which policies have been effective and which haven't when considering new interventions.

**PolicyChat** builds upon the previous work of **WikiChat**, but it is specifically designed to analyze and interpret **economic and econometric papers**. Its primary goal is to offer policymakers the **most pertinent guidance** by summarizing research and policy effectiveness, helping them make informed decisions.

---

## Project Overview

The **PolicyChat** assistant:
- **Retrieves and interprets** research papers from trusted sources such as the **NBER**, **Urban Institute**, and **JPAL**.
- **Summarizes results** from policies tested through **RCTs or quasi-experiments**, providing policymakers with clear insights.
- **Guides decisions** by indicating the policies that have worked (or failed) based on **measurable results**.

---

# PolicyChat: A Chatbot for Evidence-Based Policy Guidance

**PolicyChat** is designed to help policymakers find evidence-based policies using a conversational agent. The agent focuses on guiding users through randomized controlled trials (RCTs) and quasi-experiment-based research, providing the most pertinent guidance on what has worked or hasn't in different contexts. Inspired by WikiChat, PolicyChat is optimized for the interpretation of economic and econometric papers.

## Install Dependencies

### 1. Clone the repository:

```bash
git clone https://github.com/stanford-oval/WikiChat.git
cd WikiChat
```
### 2. Set upt the environment
We recommend using the conda environment specified in conda_env.yaml. This environment includes Python 3.10, pip, gcc, g++, make, Redis, and all required Python packages. Ensure you have either Conda, Anaconda, or Miniconda installed. Then create and activate the conda environment:

```bash
conda env create --file conda_env.yaml
conda activate wikichat
python -m spacy download en_core_web_sm  # Spacy is only needed for certain WikiChat configurations
```

If you see Error: Redis lookup failed after running the chatbot, it probably means Redis is not properly installed. You can try reinstalling it by following its official documentation.

Keep this environment activated for all subsequent commands.

###3. Download and install Docker

We will be using Docker to run a vector database for storing and retrieving research paper embeddings. Download Docker and install it based on your operating system. Follow instructions at Docker Installation.

### 3. Start the Qdrant Server

PolicyChat uses Qdrant to manage embeddings of research papers. Run the following command to start Qdrant as a container:

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```
Make sure Qdrant is running locally on port 6333. Verify it's running by typing:

```bash
docker ps
```
### 4. Prepare and chunk papers

The next step is to prepare research papers by dividing them into manageable chunks for embedding and retrieval.

## a. Chunking the Papers

PolicyChat processes papers using the Chunkenizer script. This script breaks papers into smaller parts of fewer than 500 tokens, which makes it easier for embedding. Remember to place your PDF research papers in the appropriate directory. Also run the Chunkenizer.py script to chunk the papers. This will generate a JSONL file with the chunked text. The script should also extract metadata like paper titles, sections, and content.

Example output structure for each chunk in the JSONL file:

paper_id: The title of the paper.
section_title: The section of the paper.
content_string: The text of the chunk.
The output file will be used in the embedding step.

### 5. Embedding the Paper Chunks

Once the papers have been chunked, the next step is to generate embeddings for each chunk using a pre-trained model.

#### a. Generating Embeddings

The Embbedingator.py script takes the chunks from the JSONL file and generates embeddings using the BAAI/bge-small-en model or a similar model from Hugging Face.

Ensure the Embbedingator.py script is set up to point to the path of the JSONL file generated in the previous step.

#### b. Index the Embeddings in Qdrant

While the embeddings are being generated, they will be automatically indexed in Qdrant, a vector database. This indexing step allows the embeddings to be stored for fast retrieval during PolicyChat’s operations.

### 6. Querying the Qdrant Database

Once the embeddings have been successfully indexed, PolicyChat can now search for relevant papers based on user queries.

#### a. Perform a Query

Use the PerformQuery.py script to search the indexed papers in Qdrant. The script will take the user’s query (such as “financial inclusion” or “education policy”) and return the most relevant sections from the indexed papers.



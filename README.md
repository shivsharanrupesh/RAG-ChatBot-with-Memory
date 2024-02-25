# RAGChat - An LLM-powered Chat Bot

RAGChat is a chatbot powered by Language Models (LLM) for interacting with a custom knowledge database. It uses the CohereEmbeddings for natural language understanding and Qdrant for storing and retrieving vectors.

## Setup

### Requirements

Make sure to install the required packages before running the application.

```bash
pip install streamlit qdrant-client langchain
```

## Environment Variables

Set the following environment variables:

```bash
export COHERE_API_KEY=your_cohere_api_key
export QDRANT_HOST=your_qdrant_host
export QDRANT_API_KEY=your_qdrant_api_key
```

## Usage

```bash
git clone https://github.com/your-username/RAGChat.git
cd RAGChat
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run main.py
```

## Configuration

Make sure to configure your Cohere API key, Qdrant host, and Qdrant API key in the environment variables as mentioned in the Setup section.

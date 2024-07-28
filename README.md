# Simple RAG Template

This is a starter repo for an easy and quick Retrieval-Augmented Generation (RAG) system.

## Purpose

This application demonstrates a basic implementation of a RAG system, which combines the power of large language models with a custom knowledge base. It uses Streamlit for the frontend interface.

## How it Works

> A Jupyter Notebook is available `rag_tutorial.ipynb` if needed.

### Requirements

- Python 3.7+
- Create a new virtual environment
- Install the required Python libraries:
  ```
  pip install -r requirements.txt
  ```
- Add your OpenAI API key to the environment variable: `OPENAI_API_KEY`


### Create a Vector Database

Run `loader.py` to create the Vector Database and vectorize your PDF documents:

```bash
python loader.py
```

This script will create a ChromaDB instance, which is an open-source embedding database. For more information, visit: https://docs.trychroma.com/

### Run the Application (not available)

To start the Streamlit app, run: 

```bash
streamlit run app.py
```

## Features

- PDF document ingestion and vectorization
- Natural language querying of the knowledge base
- Integration with OpenAI's language models for response generation

## Customization

You can customize this template by:
- Adding more document types for ingestion
- Implementing different embedding models
- Enhancing the user interface with additional Streamlit components

## Limitations

- Currently only supports PDF documents
- Requires an OpenAI API key


Readme.md written by Claude 3.5 and checked by myself.
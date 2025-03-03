# RAG with Ollama and Gradio

This project implements a Retrieval-Augmented Generation (RAG) system using FastAPI, LangChain, Ollama, and Gradio. The system allows users to upload documents through a Gradio UI, which are then processed into vector embeddings using ChromaDB. Users can then query the system to retrieve answers based on the stored documents.

## Features

- **Document Upload**: Upload text files via a Gradio UI.
- **Vector Storage**: Store processed documents as vector embeddings using ChromaDB.
- **Query Interface**: Ask questions through Gradio UI and get responses from the documents.
- **FastAPI Backend**: Serves the RAG system via an API.

## Installation

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

### Install Dependencies

Run the following command to install all required dependencies:

```sh
pip install fastapi uvicorn gradio langchain langchain_community chromadb ollama python-dotenv
```

### Environment Variables

Create a `.env` file in the root directory and set any required environment variables (if needed for API keys or configurations).

## Running the Application

1. **Start the FastAPI server**

```sh
uvicorn main:app --reload
```

2. **Launch Gradio UI** Modify your Gradio script (`RAG_demo.py`) to include:

```python
import gradio as gr
import requests

API_UPLOAD_URL = "http://127.0.0.1:8000/upload/"
API_QUERY_URL = "http://127.0.0.1:8000/query/"

def upload_file(file):
    files = {"file": (file.name, file, "text/plain")}
    response = requests.post(API_UPLOAD_URL, files=files)
    return response.json()["message"]

def ask_question(question):
    response = requests.post(API_QUERY_URL, json={"question": question})
    return response.json()["answer"]

iface = gr.Interface(
    fn=ask_question,
    inputs="text",
    outputs="text",
    live=True,
    title="RAG with Ollama and Gradio"
)
iface.launch()
```

Run the UI using:

```sh
python RAG_demo.py
```

## API Endpoints

### Upload a Document

- **Endpoint:** `POST /upload/`
- **Description:** Uploads a document and processes it into vector embeddings.
- **Usage:**

```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/upload/' \
  -F 'file=@path/to/your/document.txt'
```

### Query the System

- **Endpoint:** `POST /query/`
- **Description:** Queries the system to retrieve answers based on stored documents.
- **Usage:**

```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/query/' \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is LangChain?"}'
```

## Troubleshooting

- If the API is not responding, ensure that FastAPI is running.
- If Gradio UI fails, check the API connection URL.
- Ensure all dependencies are installed correctly.

## UI - 
Upload a file here in text format, it will updated and saved in vector db(Chroma db)
![image](https://github.com/user-attachments/assets/189cfc0c-488a-4a02-8d19-68ad1c7fbb75)

Now ask any question related to that content, It will response you with the specific details.




## Contributing

Feel free to connect with me on sachinbhute23nov@gmail.com for improvements and suggestions.




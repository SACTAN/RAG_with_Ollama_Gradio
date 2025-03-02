from fastapi import Body, FastAPI, File, UploadFile, HTTPException, Query
import os
import shutil
import logging
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
DB_DIR = os.path.join(BASE_DIR, "db")
PERSISTENT_DB = os.path.join(DB_DIR, "chroma_db")

class QueryRequest(BaseModel):
    question: str

# Ensure necessary directories exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Define embedding model
EMBEDDING_MODEL = "llama3.1"
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Uploads a document and ensures it is added to the vector database properly."""
    file_path = os.path.join(DOCUMENTS_DIR, file.filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Load and process document
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    if not documents:
        return {"message": f"File '{file.filename}' uploaded but no content found."}

    # Ensure metadata is added
    for doc in documents:
        doc.metadata = {"source": file.filename}

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # ✅ Load existing ChromaDB (to keep old documents)
    db = Chroma(persist_directory=PERSISTENT_DB, embedding_function=embeddings)

    # ✅ Add new documents to the existing database
    db.add_documents(docs)
    db.persist()  # Ensure persistence

    return {"message": f"File '{file.filename}' uploaded and added to the database successfully."}



@app.post("/query/")
async def query_rag(question: str):
    """Handles user queries and retrieves answers based on stored documents."""
    try:
        # ✅ Load the existing Chroma database
        db = Chroma(persist_directory=PERSISTENT_DB, embedding_function=embeddings)

        # ✅ Check if the database contains any documents
        if db._collection.count() == 0:
            return {"answer": "No documents available in the database. Please upload files first."}

        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
        relevant_docs = retriever.invoke(question)

        if not relevant_docs:
            return {"answer": "I couldn't find relevant information in the stored documents."}

        # ✅ Extract document sources
        sources = list(set([doc.metadata["source"] for doc in relevant_docs]))

        # ✅ Prepare LLM Input
        combined_input = (
            f"Question: {question}\n\n"
            "Relevant Documents:\n"
            + "\n\n".join([doc.page_content for doc in relevant_docs])
            + "\n\nAnswer based only on provided documents."
        )

        model = OllamaLLM(model=EMBEDDING_MODEL)
        result = model.invoke(combined_input)

        return {"answer": result, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/status/")
def status():
    """API health check."""
    return {"message": "API is running."}

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Load environment variables
load_dotenv()

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Define the embedding model
embeddings = OllamaEmbeddings(model="llama3.1")

# Load the existing vector store
try:
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
except Exception as e:
    print(f"Error loading Chroma vector store: {e}")
    exit(1)

# Define the user's question
query = "What are the certification and achievements of Sachin Bhute?"

# Retrieve relevant documents
try:
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(query)
except Exception as e:
    print(f"Error retrieving documents: {e}")
    exit(1)

# Check if documents were retrieved
if not relevant_docs:
    print("\nNo relevant documents found. Exiting...")
    exit(0)

# Display relevant documents
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Prepare the input for the model
combined_input = (
    f"Here are some documents that might help answer the question: {query}\n\n"
    "Relevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. "
    "If the answer is not found in the documents, respond with 'I'm not sure'."
)

# Initialize the model
model = OllamaLLM(model="llama3.1")

# Define messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model
try:
    result = model.invoke(messages)
    print("\n--- Generated Response ---")
    print(f"Full result: {result}")
except Exception as e:
    print(f"Error invoking model: {e}")

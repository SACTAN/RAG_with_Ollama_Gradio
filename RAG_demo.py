import gradio as gr
import requests

API_UPLOAD_URL = "http://127.0.0.1:8000/upload/"
API_QUERY_URL = "http://127.0.0.1:8000/query/"

def upload_and_process(file):
    files = {'file': (file.name, file)}
    response = requests.post(API_UPLOAD_URL, files=files)
    # return response.json()["message"]

    response_json = response.json()
    print(response_json)  # Debugging: See the actual response
    return response_json.get("message", "Unexpected response format")


def ask_question(question):
    # print(f"Received question: {response}")
    # response = requests.post(API_QUERY_URL, json={"question": question})
    # print(f"Received response: {response}")
    # response_json = response.json()
    # print("API Response:", response_json)  # Debugging: See what is actually returned
    # return response.json()["answer"]

    print(f"Received question: {question}")  # Debugging
    response = requests.post(API_QUERY_URL, params={"question": question})  # Use query parameter
    print(f"Received response: {response.status_code}")  # Debugging

    try:
        response_json = response.json()
        print("API Response:", response_json)  # Debugging
        return response_json.get("answer", "No answer found.")  # Handle missing key safely
    except requests.exceptions.JSONDecodeError:
        print("Error: Response is not valid JSON")
        return "Error processing response"

with gr.Blocks() as demo:
    gr.Markdown("# RAG with Ollama - Upload & Query")
    
    with gr.Row():
        file_input = gr.File(label="Upload Document")
        file_button = gr.Button("Process")
        upload_status = gr.Textbox(label="Upload Status", interactive=False)
        
        file_button.click(upload_and_process, inputs=file_input, outputs=upload_status)

    question = gr.Textbox(label="Ask a question")
    answer = gr.Textbox(label="Answer", interactive=False)
    ask_button = gr.Button("Get Answer")

    ask_button.click(ask_question, inputs=question, outputs=answer)

demo.launch(server_name="0.0.0.0", server_port=8010, share=True)

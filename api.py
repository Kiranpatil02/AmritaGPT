import asyncio
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import torch
from transformers import pipeline, AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers
import os
from dotenv import load_dotenv
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request,File, UploadFile
from pydantic import BaseModel
from typing import List, Dict, Any
from huggingface_hub import InferenceClient
import whisper
import requests
from fastapi.testclient import TestClient
from gtts import gTTS
from fastapi.responses import StreamingResponse, JSONResponse
import io
import json

load_dotenv()
model_whisper = whisper.load_model("base")

# Load and split the text file
with open("general.txt", encoding="utf8") as f:
    raw_text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)
text_chunks = text_splitter.split_text(raw_text)

# Load embeddings for both methods
huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
google_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store_hf = FAISS.from_texts(text_chunks, embedding=huggingface_embeddings)
vector_store_google = FAISS.from_texts(text_chunks, embedding=google_embeddings)

vector_store_hf.save_local("faiss_index_hf")
vector_store_google.save_local("faiss_index_google")


client = InferenceClient(api_key=os.getenv("HF_API_TOKEN"))

# Generate conversational chain for both methods
def get_conversational_chain(use_google: bool):
    prompt_template = """
    You are an assistant for Amrita University. Use the following pieces of information to help answer the user's questions.
    Always maintain context from the previous conversation and combine it with new information from the knowledge base.

    Previous Conversation:
    {chat_history}

    Context from knowledge base:
    {context}
    
    Current Question: {question}
    
    If you previously provided information that's relevant to the current question, use that information along with any new context.
    If you cannot find specific information in the current context but you mentioned it in the chat history, you can refer to that.
    If you truly don't have enough information to answer, acknowledge what you know and what you don't know.
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "chat_history"]
    )

    if use_google:
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Format chat history
def format_chat_history(history):
    formatted_history = ""
    for entry in history:
        if "user" in entry:
            formatted_history += f"Human: {entry['user']}\n"
        if "bot" in entry:
            formatted_history += f"Assistant: {entry['bot']}\n"
    return formatted_history

# Answer question
async def answer_question(user_question, chat_history, use_google=False):
    # Load the appropriate FAISS vector store
    vector_store = FAISS.load_local(
        "faiss_index_google" if use_google else "faiss_index_hf",
        google_embeddings if use_google else huggingface_embeddings,
        allow_dangerous_deserialization=True
    )

    # Create a combined query using recent context
    context_query = user_question
    if chat_history:
        last_exchange = chat_history[-2:]  # Get last user question and bot response
        context_query = f"{' '.join([msg.get('user', msg.get('bot', '')) for msg in last_exchange])} {user_question}"

    # Retrieve relevant documents
    docs = vector_store.similarity_search(context_query)

    # Extract the text content from the retrieved documents
    context = "\n".join([doc.page_content for doc in docs])

    # Format conversation history for context
    formatted_history = format_chat_history(chat_history[-4:])  # Last 4 exchanges

    if use_google:
        chain = get_conversational_chain(use_google=use_google)
        response = chain(
            {
                "input_documents": docs,
                "question": user_question,
                "chat_history": formatted_history
            },
            return_only_outputs=True
        )
        return response["output_text"]
    else:
        print(formatted_history)
        # Construct messages for the Hugging Face model
        messages = [
            {"role": "system", "content": "Use the following context to answer the user's question."},
            {"role": "system", "content": context},
            {"role": "user", "content": user_question},
            {"role": "system", "content": formatted_history},
        ]
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=messages,
            max_tokens=500,
        )

        model_response = completion.choices[0].message["content"]

        return model_response


conversation_context: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

class QueryRequest(BaseModel):
    session_id: str | None = None
    input_text: str
    use_google: bool = False

app = FastAPI()
client = TestClient(app)

@app.get("/")
def index():
    return {"message": "Hello! Use the /get-response endpoint to chat."}

@app.post("/get-response/")
async def get_response(request: QueryRequest):
    print("HI")
    session_id = request.session_id if request.session_id else str(uuid.uuid4())

    user_input = request.input_text

    if session_id not in conversation_context:
        conversation_context[session_id] = {"history": []}

    conversation_history = conversation_context[session_id]["history"]
    conversation_history.append({"user": user_input})

    response = await answer_question(user_input, conversation_history, use_google=request.use_google)
    
    conversation_history.append({"bot": response})

    conversation_context[session_id]["history"] = conversation_history

    return {"session_id": session_id, "response": response, "history": conversation_history}


@app.post("/upload-audio")
async def upload_audio(audio: UploadFile = File(...)):
    path = "audio.webm"
    with open(path, "wb") as f:
        f.write(await audio.read())
        
    result = model_whisper.transcribe(path, fp16=False, language="en")
    print(result["text"])
    if result["text"]:
        transcription = result["text"]
        
        # return {"transcription": inp_text}
        
        data = {"input_text": transcription, "use_google": True}
        response = client.post("/get-response/", json=data)
        inp_text = response.json().get('response')
        print(inp_text)
        
        obj = gTTS(text=inp_text, lang='en', slow=False)
        audio_io = io.BytesIO()
        obj.write_to_fp(audio_io)
        audio_io.seek(0)
        headers = {
                'Transcription': transcription,
                'Response-Text': inp_text,
                'Content-Type': 'audio/mpeg'
            }
        os.remove(path)
        return StreamingResponse(
                audio_io,
                headers=headers,
                media_type="audio/mpeg"
            )
    else:
        def queryagain():
                with open("test.mp3", mode="rb") as file:
                    yield from file
        headers = {
                'Transcription': " ",
                'Response-Text': "Please enter your query again",
                'Content-Type': 'audio/mpeg'
            }
        return StreamingResponse(
                queryagain(),
                headers=headers,
                media_type="audio/mpeg"
            )

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Transcription", "Response-Text", "Content-Type"],
)

if __name__ == "__main__":
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())

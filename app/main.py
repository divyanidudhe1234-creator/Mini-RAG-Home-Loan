import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient

load_dotenv()

app = FastAPI(title="Mini RAG API")

api_key = os.getenv("NUGEN_API_KEY")

client = QdrantClient(path="qdrant_data")

collection_name = "home_loan_documents"

EMBEDDING_URL = "https://api.nugen.in/api/v3/inference/embeddings"
LLM_URL = "https://api.nugen.in/api/v3/inference/chat/completions"

MODEL_ID = "model_01m43tqnsrk7q9gb"


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Mini RAG API is running"}


@app.post("/ask")
def ask_question(data: Question):

    question = data.question

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 1. Create query embedding
    embedding_data = {
        "input": question,
        "model": MODEL_ID,
        "dimensions": 768
    }

    embedding_response = requests.post(
        EMBEDDING_URL,
        headers=headers,
        json=embedding_data
    )

    if embedding_response.status_code != 200:
        return {
            "error": "Embedding failed",
            "details": embedding_response.text
        }

    query_embedding = embedding_response.json()["data"][0]["embedding"]

    # 2. Retrieve Top 10 from Qdrant
    result = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=10,
        with_payload=True
    )

    top_10 = result.points

    # 3. Select Top 3
    top_3 = sorted(
        top_10,
        key=lambda x: x.score,
        reverse=True
    )[:3]

    # 4. Create context
    context = "\n\n".join(
        point.payload["text"]
        for point in top_3
    )

    # 5. Create prompt
    prompt = f"""
You are a helpful home loan assistant.

Answer the user's question using only the information provided in the context.

Context:
{context}

Question:
{question}

Give a clear and simple answer.
"""

    # 6. LLM inference
    llm_data = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 300,
        "temperature": 0.2,
        "stream": False
    }

    llm_response = requests.post(
        LLM_URL,
        headers=headers,
        json=llm_data
    )

    if llm_response.status_code != 200:
        return {
            "error": "LLM failed",
            "details": llm_response.text
        }

    llm_result = llm_response.json()

    answer = llm_result["choices"][0]["message"]["content"]

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": len(top_10),
        "top_3_pages": [
            point.payload["page"]
            for point in top_3
        ]
    }
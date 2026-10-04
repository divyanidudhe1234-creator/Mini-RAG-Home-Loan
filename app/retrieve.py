import os
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

api_key = os.getenv("NUGEN_API_KEY")

# Qdrant
client = QdrantClient(path="qdrant_data")

collection_name = "home_loan_documents"
print("Stored points:", client.count(collection_name).count)
# User question
question = input("Enter your question: ")

# -----------------------------
# 1. Create query embedding
# -----------------------------

url = "https://api.nugen.in/api/v3/inference/embeddings"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "input": question,
    "model": "model_01m43tqnsrk7q9gb",
    "dimensions": 768
}

response = requests.post(url, headers=headers, json=data)

if response.status_code != 200:
    print("Embedding error:", response.text)
    exit()

query_embedding = response.json()["data"][0]["embedding"]

print("\nQuery embedding created successfully!")


# -----------------------------
# 2. Qdrant Top 10 retrieval
# -----------------------------

result = client.query_points(
    collection_name=collection_name,
    query=query_embedding,
    limit=10,
    with_payload=True
)

top_10 = result.points
print("Retrieved points:", len(top_10))
print("\n==============================")
print("TOP 10 RETRIEVED CHUNKS")
print("==============================")

for i, point in enumerate(top_10):
    print(f"\n--- Result {i + 1} ---")
    print("Score:", point.score)
    print("Page:", point.payload["page"])
    print("Text:", point.payload["text"][:300])


# -----------------------------
# 3. Reranking - Top 3
# -----------------------------

top_3 = sorted(
    top_10,
    key=lambda x: x.score,
    reverse=True
)[:3]

print("\n==============================")
print("TOP 3 RERANKED CHUNKS")
print("==============================")

for i, point in enumerate(top_3):
    print(f"\n--- Reranked Result {i + 1} ---")
    print("Score:", point.score)
    print("Page:", point.payload["page"])
    print("Source:", point.payload["source"])
    print("Text:", point.payload["text"][:500])
    
    # -----------------------------
# 4. Generate final answer using LLM
# -----------------------------

context = "\n\n".join(
    point.payload["text"] for point in top_3
)

prompt = f"""
You are a helpful home loan assistant.

Answer the user's question using ONLY the information provided in the context.

Context:
{context}

Question:
{question}

Give a clear and simple answer.
"""

llm_url = "https://api.nugen.in/api/v3/inference/chat/completions"

llm_data = {
    "model": "model_01m43tqnsrk7q9gb",
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
    llm_url,
    headers=headers,
    json=llm_data
)

print("\n==============================")
print("FINAL ANSWER")
print("==============================")

if llm_response.status_code == 200:
    result = llm_response.json()

    answer = result["choices"][0]["message"]["content"]

    print("\n" + answer)
else:
    print("LLM Error:", llm_response.status_code)
    print(llm_response.text)

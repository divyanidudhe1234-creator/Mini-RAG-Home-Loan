import os
import requests
import fitz
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Load API key
load_dotenv()
api_key = os.getenv("NUGEN_API_KEY")

# -----------------------------
# 1. Extract chunks from PDF
# -----------------------------
pdf_path = "data/sample.pdf"

doc = fitz.open(pdf_path)

chunk_size = 500
overlap = 100

chunks = []

for page_number, page in enumerate(doc):
    text = page.get_text().strip()
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "page": page_number + 1,
                "source": "sample.pdf"
            })

        start = end - overlap


print("Total chunks:", len(chunks))


# -----------------------------
# 2. Nugen Embedding API
# -----------------------------
url = "https://api.nugen.in/api/v3/inference/embeddings"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}


def get_embedding(text):

    data = {
        "input": text,
        "model": "model_01m43tqnsrk7q9gb",
        "dimensions": 123
    }

    response = requests.post(
        url,
        headers=headers,
        json=data
    )

    if response.status_code != 200:
        print("Embedding error:", response.text)
        return None

    result = response.json()

    # Nugen/OpenAI-style response
    if "data" in result:
        return result["data"][0]["embedding"]

    print("Unexpected response:", result)
    return None


# -----------------------------
# 3. Create Qdrant database
# -----------------------------
client = QdrantClient(path="qdrant_data")

collection_name = "home_loan_documents"

# Delete old collection if it exists
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE
    )
)


# -----------------------------
# 4. Generate embeddings
#    and store in Qdrant
# -----------------------------
points = []

for i, chunk in enumerate(chunks):

    print(f"Embedding chunk {i + 1}/{len(chunks)}...")

    embedding = get_embedding(chunk["text"])

    if embedding is None:
        print("Skipping chunk", i + 1)
        continue

    points.append(
        PointStruct(
            id=i,
            vector=embedding,
            payload={
                "text": chunk["text"],
                "page": chunk["page"],
                "source": chunk["source"]
            }
        )
    )


# -----------------------------
# 5. Upload to Qdrant
# -----------------------------
client.upsert(
    collection_name=collection_name,
    points=points
)

print("\n==============================")
print("SUCCESS!")
print("==============================")
print("Stored points:", len(points))
print("Collection:", collection_name)
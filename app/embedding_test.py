import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NUGEN_API_KEY")

print("API key loaded:", bool(api_key))

url = "https://api.nugen.in/api/v3/inference/embeddings"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "input": "What percentage of the property's market value do lenders typically finance for a home loan?",
    "model": "model_01m43tqnsrk7q9gb",
    "dimensions": 768
}

response = requests.post(url, headers=headers, json=data)

print("Status Code:", response.status_code)
print("Response:")
print(response.text)
# Mini RAG Application – Home Loan Document QA

## Project Overview

This project is a Mini Retrieval-Augmented Generation (RAG) application for answering questions related to home loans.

The system extracts information from a PDF document, creates text chunks, generates embeddings, stores them in Qdrant, retrieves relevant chunks, and uses an LLM to generate the final answer.

## RAG Pipeline

PDF Document
↓
Text Extraction using PyMuPDF
↓
Chunking with Overlap
↓
Nugen Embeddings
↓
Qdrant Vector Database
↓
Top-K Retrieval (Top 10)
↓
Top 3 Relevant Chunks
↓
Prompt Construction
↓
Nugen LLM
↓
Final Answer

## Technologies Used

- Python
- FastAPI
- PyMuPDF
- Nugen API
- Qdrant
- Requests
- python-dotenv
- Uvicorn

## Project Structure

Mini RAG/
│
├── app/
│   ├── main.py
│   ├── extract_pdf.py
│   ├── chunking.py
│   ├── embedding_test.py
│   ├── embed_and_store.py
│   └── retrieve.py
│
├── data/
│   └── sample.pdf
│
├── qdrant_data/
│
├── requirements.txt
├── README.md
└── .env

## Features

- PDF text extraction
- Text chunking with overlap
- Page and source metadata
- Nugen embeddings
- Qdrant vector storage
- Top 10 semantic retrieval
- Top 3 relevant chunks
- LLM-based answer generation
- FastAPI REST API
- Swagger UI for testing

## API Endpoint

### POST /ask

Example request:

{
  "question": "What documents are required for a home loan?"
}

The API returns:

- User question
- Generated answer
- Number of retrieved chunks
- Pages of the top 3 retrieved chunks

## Example Question

Question:

What documents are required for a home loan?

Example Answer:

The required documents for a home loan generally include income proof, identity proof, address proof, employment or business documents, property-related documents, bank statements and credit-related documents.

## How to Run

### 1. Create virtual environment

python -m venv venv

### 2. Activate environment

Windows:

venv\Scripts\activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Configure API key

Create a `.env` file:

NUGEN_API_KEY=your_api_key_here

### 5. Run the FastAPI application

uvicorn app.main:app --reload

### 6. Open Swagger UI

Open:

http://127.0.0.1:8000/docs

Use the `/ask` endpoint to ask questions.

## Sample Q&A

### Q1. What documents are required for a home loan?

Answer: Income proof, identity proof, address proof, employment/business documents, property documents, bank statements and credit-related documents may be required.

### Q2. What factors affect home loan approval?

Answer: Loan approval can depend on income, credit score, employment or business stability, existing financial obligations and property-related factors.

### Q3. What is EMI?

Answer: EMI stands for Equated Monthly Instalment. It is the fixed periodic payment made by a borrower towards repayment of the home loan.

## Conclusion

This Mini RAG application demonstrates how Retrieval-Augmented Generation can be used to answer questions from a specific PDF knowledge base. It combines document processing, embeddings, vector search and LLM-based generation into a complete question-answering system.
import os
from dotenv import load_dotenv
# import openai
import chromadb
from sentence_transformers import SentenceTransformer
import cohere
import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.models import Batch
from typing import List
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor
from qdrant_client.models import VectorParams, Distance
from qdrant_client import models
import numpy as np
import pandas as pd


# Load environment variables from .env file
load_dotenv(override=True)

# Retrieve the API key
cohere_key = os.getenv("COHERE_API_KEY")

# Initialize clients
cohere_client = cohere.ClientV2(cohere_key)

qdrant_key = os.getenv("QDRANT_API_KEY")
qdrant_host = os.getenv("QDRANT_HOST")


def process_pdf_to_text(file_path: str, num_threads: int = 4) -> str:
    """
    Processes a PDF file and extracts its content as a single text string.

    Args:
        file_path: The path to the PDF file.
        num_threads: The number of threads to use for parallel page processing.

    Returns:
        The extracted text as a single string with no newlines. Returns an empty string on error.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"File is not a PDF: {file_path}")

    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            num_pages = len(reader.pages)

            # Use ThreadPoolExecutor to process pages in parallel
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                page_texts = list(executor.map(extract_page_text, [(reader, i) for i in range(num_pages)]))

            return " ".join(page_texts)  # Join all pages into a single string without newlines

    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")  # More specific error
        return ""

def extract_page_text(args):
    """Helper function for parallel PDF page extraction."""
    reader, page_num = args
    try:
        page = reader.pages[page_num]
        return " ".join(page.extract_text().split())  # Remove newlines and extra spaces
    except Exception as e:
        print(f"Error processing page {page_num}: {e}")  # Error handling
        return ""
    
def customize_chunking(text, chunk_size=150):
    list_of_chunks = []
    chunk = ""
    for i in range(0, len(text), chunk_size):
        chunk += text[i:i+chunk_size] + "\n"
        list_of_chunks.append(chunk)
    return list_of_chunks

def retrieve_top_chunks(query:str, collection_name, chunks, n=5):
    # Fetch all stored points
    stored_points = qdrant_client.scroll(collection_name="daves-rag", with_vectors=True, limit=1000)[0]
    query_chunks = query_chunking([query])
    query_embeddings = query_chunks.embeddings.float
    
    # Extract embeddings & IDs
    chunk_embeddings = [point.vector for point in stored_points]
    stored_ids = [point.id for point in stored_points]
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # --- Compute Similarity Scores ---
    similarities = []
    for chunk_embedding in chunk_embeddings:
        subquery_scores = [cosine_similarity(query_embedding, chunk_embedding) for query_embedding in query_embeddings]
        similarities.append(np.mean(subquery_scores))  # Average similarity if multiple subqueries

    print("Similarity scores:", similarities)

    # --- Retrieve Top `n` Chunks ---
    top_indices = np.argsort(similarities)[::-1][:n]  # Sort and get top `n`

    # Retrieve top similar document chunks
    top_chunks_after_retrieval = [chunks[i] for i in top_indices]

    return top_chunks_after_retrieval


def get_llm_output(top_chunks, ch, query):
    preamble = """
    ## Task & Context
    You give answers to user's questions with precision, based on chunked document string you receive.
    You should focus on serving the user's needs as best you can, which can be wide-ranging but always relevant to the document string.
    If you are not sure about the answer, you can ask for clarification or provide a general response saying you are not sure.
    
    ## Style Guide
    Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.
    """

    # retrieved documents
    documents = [
        {"data": {"title": f"chunk {i}", "snippet": top_chunks[i]}} for i in range(len(top_chunks))
    ]

    # get model response
    response = ch.chat(
        model="command-r-08-2024",
        messages=[{"role": "system", "content": preamble},
                  {"role": "user", "content": query}],
        documents=documents,  
        temperature=0.3
    )

    print("Final answer:")
    print(response.message.content[0].text)

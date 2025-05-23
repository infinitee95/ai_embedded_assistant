import spacy
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests
import time
import logging
import os

# Set up logging to print timestamps and step durations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize tools
nlp = spacy.load("en_core_web_sm")  # Load spaCy for semantic enrichment
model = SentenceTransformer('all-MiniLM-L6-v2')  # Load embedding model
client = chromadb.PersistentClient(path="./embeddings", settings=Settings())
collection = client.get_collection(name="documents")  # Access Chroma collection

# Get Hugging Face API key from environment variable
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY environment variable not set. Please set it before running the app.")

# Semantic enrichment function (same as in preprocess.py)
def semantic_enrichment(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "DATE", "EVENT", "PRODUCT", "LAW"]]
    key_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    semantic_terms = list(set(entities + key_phrases))
    enriched_text = f"{text} [Semantic Terms: {', '.join(semantic_terms)}]"
    return enriched_text

# Function to process a user query with progress updates
def process_query(query, k=1, status=None):
    start_time = time.time()

    # Step 1: Enrich the query
    if status:
        status.update(label="Step 1: Enriching query with semantic terms...")
    logging.info("Starting query enrichment...")
    step_start = time.time()
    enriched_query = semantic_enrichment(query)
    logging.info(f"Query enrichment completed in {time.time() - step_start:.2f} seconds")

    # Step 2: Embed the enriched query
    if status:
        status.update(label="Step 2: Embedding the query...")
    logging.info("Starting query embedding...")
    step_start = time.time()
    query_embedding = model.encode([enriched_query])[0]
    logging.info(f"Query embedding completed in {time.time() - step_start:.2f} seconds")

    # Step 3: Retrieve top-k relevant chunks from Chroma
    if status:
        status.update(label="Step 3: Retrieving relevant document chunks...")
    logging.info("Starting Chroma retrieval...")
    step_start = time.time()
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    retrieved_chunks = results["documents"][0]
    metadata = results["metadatas"][0]
    logging.info(f"Chroma retrieval completed in {time.time() - step_start:.2f} seconds")

    # Step 4: Build context from retrieved chunks
    if status:
        status.update(label="Step 4: Building context for answer generation...")
    logging.info("Building context...")
    step_start = time.time()
    context = "\n\n".join([f"Excerpt from {meta['document']}:\n{chunk}" for chunk, meta in zip(retrieved_chunks, metadata)])
    context = " ".join(context.split()[:100])  # Truncate to 100 words
    logging.info(f"Truncated context length: {len(context.split())} words")

    # Step 5: Generate answer using Hugging Face Inference API
    if status:
        status.update(label="Step 5: Generating answer with Hugging Face API...")
    logging.info("Sending request to Hugging Face API for answer generation...")
    step_start = time.time()
    prompt = f"""System: You are an expert assistant. Answer the question using only the provided document excerpts.
Document Excerpts:
{context}

User Query: {query}
Answer in detail:"""

    try:
        logging.info(f"Prompt length: {len(prompt.split())} words")
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        logging.info("Sending request to Hugging Face API at %s...", time.strftime('%Y-%m-%d %H:%M:%S'))
        response = requests.post(
            "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        answer = response.json()[0]["generated_text"].split("Answer in detail:")[1].strip()
        logging.info(f"Hugging Face API answer generation completed in {time.time() - step_start:.2f} seconds")
    except requests.exceptions.Timeout:
        answer = "Error: Hugging Face API request timed out after 30 seconds. Check your internet connection or API status."
        logging.error("Hugging Face API request timed out")
    except requests.exceptions.RequestException as e:
        answer = f"Error: Hugging Face API request failed. Ensure your API key is valid and the model is available. Details: {e}"
        logging.error(f"Hugging Face API request failed: {e}")
    except (KeyError, IndexError):
        answer = "Error: Unexpected response format from Hugging Face API."
        logging.error("Unexpected response format from Hugging Face API")

    # Log total processing time
    logging.info(f"Total query processing time: {time.time() - start_time:.2f} seconds")

    return answer, retrieved_chunks, metadata

if __name__ == "__main__":
    # Test the query function
    query = "What are the key trends in business analytics?"
    answer, chunks, metadata = process_query(query)
    print("Answer:", answer)
    print("\nSources:")
    for chunk, meta in zip(chunks, metadata):
        print(f"- {meta['document']}: {chunk[:500]}...")
import spacy
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests
import time
import logging

# Set up logging to print timestamps and step durations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize tools
nlp = spacy.load("en_core_web_sm")  # Load spaCy for semantic enrichment
model = SentenceTransformer('all-MiniLM-L6-v2')  # Load embedding model
client = chromadb.PersistentClient(path="./embeddings", settings=Settings())
collection = client.get_collection(name="documents")  # Access Chroma collection

# Semantic enrichment function (same as in preprocess.py)
def semantic_enrichment(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "DATE", "EVENT", "PRODUCT", "LAW"]]
    key_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    semantic_terms = list(set(entities + key_phrases))
    enriched_text = f"{text} [Semantic Terms: {', '.join(semantic_terms)}]"
    return enriched_text

# Function to process a user query with progress updates
def process_query(query, k=2, status=None):
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
    logging.info(f"Context building completed in {time.time() - step_start:.2f} seconds")
    logging.info(f"Context length: {len(context.split())} words")

    # Step 5: Generate answer using Ollama
    if status:
        status.update(label="Step 5: Generating answer with Mistral-7B (this may take a while)...")
    logging.info("Sending request to Ollama for answer generation...")
    step_start = time.time()
    prompt = f"""System: You are an expert assistant. Answer the question using only the provided document excerpts.
Document Excerpts:
{context}

User Query: {query}
Answer in detail:"""
    logging.info(f"Prompt length: {len(prompt.split())} words")
    logging.info(f"Request payload: {{\"model\": \"mistral\", \"prompt\": \"[truncated for logging]\", \"max_tokens\": 300}}")

    try:
        # Test Ollama connectivity before sending the main request
        logging.info("Testing Ollama server connectivity...")
        test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        test_response.raise_for_status()
        logging.info("Ollama server is reachable")

        # Send the main request
        logging.info("Sending main request to Ollama...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "max_tokens": 300},
            timeout=60
        )
        response.raise_for_status()
        answer = response.json()["response"]
        logging.info(f"Ollama answer generation completed in {time.time() - step_start:.2f} seconds")
    except requests.exceptions.Timeout:
        answer = "Error: Ollama request timed out after 60 seconds. Try a smaller model or reduce the context size."
        logging.error("Ollama request timed out")
    except requests.exceptions.ConnectionError as e:
        answer = f"Error: Could not connect to Ollama server. Ensure Ollama is running on localhost:11434. Details: {e}"
        logging.error(f"Ollama connection failed: {e}")
    except requests.exceptions.RequestException as e:
        answer = f"Error: Ollama request failed. Ensure Mistral-7B is loaded and Ollama is functioning. Details: {e}"
        logging.error(f"Ollama request failed: {e}")

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
        print(f"- {meta['document']}: {chunk[:100]}...")
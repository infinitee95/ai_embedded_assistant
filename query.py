import spacy
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests

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

# Function to process a user query
def process_query(query, k=5):
    # Enrich the query
    enriched_query = semantic_enrichment(query)
    # Embed the enriched query
    query_embedding = model.encode([enriched_query])[0]

    # Retrieve top-k relevant chunks from Chroma
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    retrieved_chunks = results["documents"][0]
    metadata = results["metadatas"][0]

    # Build context from retrieved chunks
    context = "\n\n".join([f"Excerpt from {meta['document']}:\n{chunk}" for chunk, meta in zip(retrieved_chunks, metadata)])

    # Create prompt for Mistral-7B
    prompt = f"""System: You are an expert assistant. Answer the question using only the provided document excerpts.
Document Excerpts:
{context}

User Query: {query}
Answer in detail:"""

    # Send request to Ollama API
    try:
        response = requests.post("http://localhost:11434/api/generate", json={"model": "mistral", "prompt": prompt})
        response.raise_for_status()
        answer = response.json()["response"]
    except requests.exceptions.RequestException as e:
        answer = f"Error: Could not connect to Ollama. Ensure Ollama is running and Mistral-7B is loaded. Details: {e}"

    return answer, retrieved_chunks, metadata

if __name__ == "__main__":
    # Test the query function
    query = "What are the key trends in business analytics?"
    answer, chunks, metadata = process_query(query)
    print("Answer:", answer)
    print("\nSources:")
    for chunk, meta in zip(chunks, metadata):
        print(f"- {meta['document']}: {chunk[:100]}...")
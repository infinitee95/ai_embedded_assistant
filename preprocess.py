import pdfplumber
import nltk
import os
import spacy
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from docx import Document  # For handling .doc and .docx files

# Initialize tools
nlp = spacy.load("en_core_web_sm")  # Load spaCy for semantic enrichment
model = SentenceTransformer('all-MiniLM-L6-v2')  # Load embedding model
client = chromadb.PersistentClient(path="./embeddings", settings=Settings())  # Chroma client
collection = client.get_or_create_collection(name="documents")  # Chroma collection for storing embeddings

# Semantic enrichment function: Adds entities and key phrases to text
def semantic_enrichment(text):
    doc = nlp(text)
    # Extract named entities (e.g., organizations, dates)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "DATE", "EVENT", "PRODUCT", "LAW"]]
    # Extract key noun phrases
    key_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    # Combine unique terms
    semantic_terms = list(set(entities + key_phrases))
    # Append semantic terms to original text
    enriched_text = f"{text} [Semantic Terms: {', '.join(semantic_terms)}]"
    return enriched_text

# Function to extract text from a PDF file
def process_pdf(pdf_path):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # Split text into sentences
                sentences = nltk.sent_tokenize(text)
                chunk = ""
                # Group sentences into chunks of ~150 tokens
                for sentence in sentences:
                    if len(chunk.split()) + len(sentence.split()) <= 150:
                        chunk += " " + sentence
                    else:
                        if chunk:
                            chunks.append(chunk.strip())
                        chunk = sentence
                if chunk:
                    chunks.append(chunk.strip())
    return chunks

# Function to extract text from a DOC or DOCX file
def process_docx(docx_path):
    chunks = []
    doc = Document(docx_path)
    text = ""
    # Extract text from all paragraphs
    for para in doc.paragraphs:
        if para.text.strip():
            text += para.text + "\n"
    # Split into chunks similar to PDF processing
    if text:
        sentences = nltk.sent_tokenize(text)
        chunk = ""
        for sentence in sentences:
            if len(chunk.split()) + len(sentence.split()) <= 150:
                chunk += " " + sentence
            else:
                if chunk:
                    chunks.append(chunk.strip())
                chunk = sentence
        if chunk:
            chunks.append(chunk.strip())
    return chunks

# Function to extract text from a TXT file
def process_txt(txt_path):
    chunks = []
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()
    if text:
        sentences = nltk.sent_tokenize(text)
        chunk = ""
        for sentence in sentences:
            if len(chunk.split()) + len(sentence.split()) <= 150:
                chunk += " " + sentence
            else:
                if chunk:
                    chunks.append(chunk.strip())
                chunk = sentence
        if chunk:
            chunks.append(chunk.strip())
    return chunks

# Main function to preprocess all supported files in the documents folder
def preprocess_documents():
    pdf_dir = "./documents"
    all_chunks = []  # Original chunks for retrieval
    enriched_chunks = []  # Enriched chunks for embedding
    chunk_ids = []  # Unique IDs for each chunk
    doc_ids = []  # Document names for metadata

    # Process each file based on its extension
    for idx, file_name in enumerate(os.listdir(pdf_dir)):
        file_path = os.path.join(pdf_dir, file_name)
        print(f"Processing {file_name}...")
        
        # Determine file type and process accordingly
        if file_name.endswith(".pdf"):
            chunks = process_pdf(file_path)
        elif file_name.endswith((".doc", ".docx")):
            chunks = process_docx(file_path)
        elif file_name.endswith(".txt"):
            chunks = process_txt(file_path)
        else:
            print(f"Skipping unsupported file: {file_name}")
            continue

        # Enrich and store each chunk
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{file_name}_{chunk_idx}"
            enriched_chunk = semantic_enrichment(chunk)
            all_chunks.append(chunk)
            enriched_chunks.append(enriched_chunk)
            chunk_ids.append(chunk_id)
            doc_ids.append(file_name)

    # Embed enriched chunks
    print("Embedding enriched chunks...")
    embeddings = model.encode(enriched_chunks, batch_size=32, show_progress_bar=True)

    # Store in Chroma with original chunks and metadata
    collection.add(
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=[{"document": doc_id} for doc_id in doc_ids],
        ids=chunk_ids
    )
    print(f"Stored {len(all_chunks)} chunks in Chroma.")

if __name__ == "__main__":
    preprocess_documents()
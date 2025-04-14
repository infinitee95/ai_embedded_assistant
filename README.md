"# ai_embedded_assistant" 
Platform Setup

Install Dependencies:
Create a virtual environment: python -m venv venv
Activate it: source venv/bin/activate (Linux/Mac) or venv\Scripts\activate (Windows)
Install packages: pip install pdfplumber nltk sentence-transformers chromadb streamlit spacy requests
Download NLTK data: Run import nltk; nltk.download('punkt') in Python
Download spaCy model: python -m spacy download en_core_web_sm

Install and Run Ollama:
Download Ollama from this website.
Follow installation instructions for your OS, then run ollama serve & in the background.
Start Mistral-7B: ollama run mistral (this may take time initially to download).

Prepare Documents:
Create a documents/ folder in your project directory and place your PDFs there.

Run Preprocessing:
Execute python preprocess.py to process documents, apply semantic enrichment, embed chunks, and store in Chroma.

Run the UI:
Start the Streamlit app with streamlit run app.py.
Open this website in your browser to interact with the chatbot.

=================================================================

Steps to Set Up

1. Create the documents/ folder in your project directory and add your PDF files.

2. Run preprocess.py to process the documents and generate embeddings (stored in embeddings/).

3. Ensure Ollama is running with Mistral-7B loaded. You can start it with:
ollama serve &
ollama run mistral

4. Run app.py to launch the Streamlit UI and start using the chatbot:
streamlit run app.py

=================================================================

Description

documents/: Place all your PDF files here (e.g., whitepaper1.pdf, ebook2.pdf). The preprocess.py script will read and process these files.
embeddings/: This folder is automatically created when you run preprocess.py. It stores the Chroma vector database with the embeddings of your document chunks, so you don’t need to create it manually.
preprocess.py: This script processes the PDFs, extracts text, chunks it, applies semantic enrichment, embeds the chunks, and stores them in Chroma.
query.py: This handles user queries, retrieves relevant document chunks from Chroma, and generates answers using the Mistral-7B model via Ollama.
app.py: This sets up the Streamlit web interface, letting you interact with the chatbot through a browser.
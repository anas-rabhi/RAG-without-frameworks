import os
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
import openai

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Create or get a collection
collection = chroma_client.get_or_create_collection("pdf_collection")

# Set up OpenAI API (make sure to set your API key as an environment variable)
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_path):
    """
    Extract all the text from a PDF
    """
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(text, words_per_chunk=500, overlap=50):
    """
    Split PDFs into smaller chunks => for a better embedding 
    """
    words = re.findall(r'\S+', text)
    chunks = []
    for i in range(0, len(words), words_per_chunk - overlap):
        chunk = ' '.join(words[i:i + words_per_chunk])
        chunks.append(chunk)
    return chunks

def get_embedding(text):
    """
    Call OpenAI to create embeddings for a given text
    """
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def process_pdfs(folder_path):
    """
    Process the PDFs: 
    1. Extract the text
    2. Convert the text into chunks
    3. Get embeddings for each chunk
    4. Load the chunks and the embeddings inside the chroma DB
    
    PS : We can also request OpenAI with batches...
    """

    doc_id = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            
            # Extract text from PDF
            text = extract_text_from_pdf(pdf_path)
            
            # Split text into chunks
            chunks = split_text_into_chunks(text)
            
            for i, chunk in enumerate(chunks):
                # Create embeddings using OpenAI
                embedding = get_embedding(chunk)
                
                # Add to Chroma
                collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"source": filename, "chunk": i}],
                    ids=[f"{filename}_chunk_{i}"]
                )
                doc_id += 1

    print(f"Processed and added {len(collection.get()['ids'])} chunks to Chroma.")

# Usage
If __name__== '__main__':
    data_folder = "./data"  # Replace with your PDF folder path
    process_pdfs(data_folder)

import os
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document

# ---------------------------
# Extract text
# ---------------------------
def extract_text_from_file(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"

    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        for p in doc.paragraphs:
            text += p.text + "\n"

    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    return text.strip()

# ---------------------------
# Chunking
# ---------------------------
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


# ---------------------------
# Ingestion
# ---------------------------
def build_index(folder="documents"):
    documents = []
    sources = []

    print("\nReading files...")
    for file in os.listdir(folder):
        if file.endswith((".pdf", ".docx", ".txt")):
            path = os.path.join(folder, file)
            print(f" → {file}")

            content = extract_text_from_file(path)
            chunks = chunk_text(content)

            documents.extend(chunks)
            sources.extend([file] * len(chunks))

    print(f"\nTotal chunks: {len(documents)}")

    # Save chunk data
    pickle.dump(documents, open("chunks.pkl", "wb"))
    pickle.dump(sources, open("sources.pkl", "wb"))

    # Create embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("\nCreating embeddings...")
    embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Save embeddings
    np.save("embeddings.npy", embeddings)

    # Build FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "index.faiss")
    print("\nFAISS index saved as index.faiss")


if __name__ == "__main__":
    build_index()

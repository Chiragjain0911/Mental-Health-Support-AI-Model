import faiss, numpy as np, os
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

PDF_FOLDER = "data"

all_text_chunks = []

print("Loading PDFs...")
for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        print("➡ Reading:", file)
        reader = PdfReader(os.path.join(PDF_FOLDER, file))
        text = ""
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                text += txt + " "
        # split into chunks
        while len(text) > 300:
            all_text_chunks.append(text[:300])
            text = text[200:]

print(" Total chunks:", len(all_text_chunks))

print(" Generating embeddings...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
emb = embedder.encode(all_text_chunks)

print(" Building FAISS index...")
index = faiss.IndexFlatL2(emb.shape[1])
index.add(emb)

print(" Saving vector.index + stored_docs.npy...")
faiss.write_index(index, "vector.index")
np.save("stored_docs.npy", np.array(all_text_chunks, dtype=object))

print(" DONE — FAST RAG DATABASE READY!")

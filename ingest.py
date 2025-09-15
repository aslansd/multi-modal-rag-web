import os
import faiss
import numpy as np
import pickle
from PIL import Image
from PyPDF2 import PdfReader
from utils.embeddings import get_text_embedding, get_image_embedding

docs_path = "data/documents/"
text_vectors = []
text_metadatas = []
image_vectors = []
image_metadatas = []

# Ensure the vectorstore directory exists
os.makedirs("vectorstore", exist_ok=True)

# Iterate through the documents directory
for fname in os.listdir(docs_path):
    fpath = os.path.join(docs_path, fname)
    
    # Skip directories
    if os.path.isdir(fpath):
        continue
    
    # Handle PDF files
    if fname.endswith(".pdf"):
        # Open the PDF in binary read mode ('rb')
        try:
            reader = PdfReader(fpath)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() or ""
            
            # Now we have the text, we can embed it
            if full_text.strip():
                vec = get_text_embedding(full_text)
                text_vectors.append(vec)
                text_metadatas.append({"type": "text", "source": fname, "content": full_text})
        except Exception as e:
            print(f"Error processing PDF file {fname}: {e}")
            continue

    # Handle image files
    elif fname.endswith((".png", ".jpg", ".jpeg")):
        try:
            img = Image.open(fpath)
            vec = get_image_embedding(img)
            image_vectors.append(vec)
            image_metadatas.append({"type": "image", "source": fname})
        except Exception as e:
            print(f"Error processing image file {fname}: {e}")
            continue
    
    # Ignore other file types
    else:
        print(f"Skipping unsupported file: {fname}")

# Create and save the text index
if text_vectors:
    text_vectors_np = np.stack(text_vectors)
    text_index = faiss.IndexFlatL2(text_vectors_np.shape[1])
    text_index.add(text_vectors_np)
    faiss.write_index(text_index, "vectorstore/text_index.faiss")
    with open("vectorstore/text_metadata.pkl", "wb") as f:
        pickle.dump(text_metadatas, f)

# Create and save the image index
if image_vectors:
    image_vectors_np = np.stack(image_vectors)
    image_index = faiss.IndexFlatL2(image_vectors_np.shape[1])
    image_index.add(image_vectors_np)
    faiss.write_index(image_index, "vectorstore/image_index.faiss")
    with open("vectorstore/image_metadata.pkl", "wb") as f:
        pickle.dump(image_metadatas, f)
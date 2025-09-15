from PIL import Image
from PyPDF2 import PdfReader
from utils.embeddings import get_text_embedding, get_image_embedding

def embed_uploaded_files(files):
    # Separate lists for text and image data
    text_vectors = []
    text_metadatas = []
    image_vectors = []
    image_metadatas = []

    for file in files:
        if file.name.endswith(".pdf"):
            try:
                reader = PdfReader(file)
                full_text = ""
                for page in reader.pages:
                    full_text += page.extract_text() or ""
                
                if full_text.strip():
                    vec = get_text_embedding(full_text)
                    text_vectors.append(vec)
                    text_metadatas.append({
                        "type": "text",
                        "content": full_text,
                        "source": file.name
                    })
            
            except Exception as e:
                print(f"Error processing uploaded PDF {file.name}: {e}")

        elif file.name.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                img = Image.open(file)
                vec = get_image_embedding(img)
                image_vectors.append(vec)
                image_metadatas.append({
                    "type": "image",
                    "source": file.name
                })
            
            except Exception as e:
                print(f"Error processing uploaded image {file.name}: {e}")
            
    return text_vectors, text_metadatas, image_vectors, image_metadatas
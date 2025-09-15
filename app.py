import faiss
import numpy as np
import os
import streamlit as st
import tempfile
from PIL import Image
from rag_pipeline import rag_pipeline
from utils.embed_uploaded_files import embed_uploaded_files
from utils.embeddings import get_text_embedding, get_image_embedding
from utils.model_wrapper import stream_ollama, stream_ollama_mm

st.set_page_config(page_title="📚 Multimodal Chat", layout="wide")
st.title("🧠 Multimodal RAG Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input area
with st.sidebar:
    st.markdown("### Upload Image (optional)")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    st.markdown("### Chat Settings")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
    
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs or images", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True)

    if "custom_docs" not in st.session_state:
        st.session_state.custom_docs = []

    for file in uploaded_files or []:
        st.session_state.custom_docs.append(file)

    if st.button("🗑️ Clear uploads"):
        st.session_state.custom_docs = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_response = ""
        msg_placeholder = st.empty()
        
        # NOTE: This is a corrected approach. The 'embed_uploaded_files' function
        # should now return separate lists for text and image embeddings.
        # This assumes the function signature has been updated to:
        # `embed_uploaded_files(files)` -> `(extra_text_vectors, extra_text_meta, extra_image_vectors, extra_image_meta)`
        extra_text_vectors, extra_text_meta, extra_image_vectors, extra_image_meta = embed_uploaded_files(st.session_state.custom_docs)

        # Rework the input dictionary to match the State in rag_pipeline.py
        inputs = {"input": {"text": prompt}}
        if uploaded_image is not None:
            # We open the uploaded file as an Image object for the RAG pipeline.
            inputs["input"]["image"] = Image.open(uploaded_image)

        # Run the RAG pipeline to get base documents from the pre-built vector store.
        # The pipeline will use the text or image index based on the input type.
        base_result = rag_pipeline.invoke(inputs)
        base_docs = base_result["retrieved"]
        
        user_docs = []
        
        # Perform search on user-uploaded text documents (e.g., PDFs)
        # This search uses the text prompt's embedding.
        if extra_text_vectors:
            query_vec = get_text_embedding(prompt)
            extra_text_vectors_np = np.stack(extra_text_vectors)
            temp_text_index = faiss.IndexFlatL2(extra_text_vectors_np.shape[1])
            temp_text_index.add(extra_text_vectors_np)
            D, I = temp_text_index.search(np.expand_dims(query_vec, 0), k=3)
            user_docs += [extra_text_meta[i] for i in I[0]]

        # Perform search on user-uploaded image documents
        # This search uses the uploaded image's embedding, if available.
        if uploaded_image is not None and extra_image_vectors:
            query_vec_image = get_image_embedding(Image.open(uploaded_image))
            extra_image_vectors_np = np.stack(extra_image_vectors)
            temp_image_index = faiss.IndexFlatL2(extra_image_vectors_np.shape[1])
            temp_image_index.add(extra_image_vectors_np)
            D_img, I_img = temp_image_index.search(np.expand_dims(query_vec_image, 0), k=3)
            user_docs += [extra_image_meta[i] for i in I_img[0]]
            
        # Merge all retrieved documents
        merged_docs = base_docs + user_docs

        prompt_text = "Use the following documents to answer:\n"
        for doc in merged_docs:
            if doc["type"] == "text":
                prompt_text += f"{doc['content']}\n"
            elif doc["type"] == "image":
                prompt_text += f"[Image: {doc['source']}]\n"
        prompt_text += f"\nQuestion: {prompt}\nAnswer:"

        # Stream the answer
        if uploaded_image is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            
            uploaded_image.seek(0)
            tmp.write(uploaded_image.read())
            
            gen = stream_ollama_mm(prompt_text, [tmp.name])
        else:
            gen = stream_ollama(prompt_text)

        for token in gen:
            full_response += token
            msg_placeholder.markdown(full_response + "▌")

        if uploaded_image is not None:
            os.unlink(tmp.name)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})
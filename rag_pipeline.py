import faiss
import numpy as np
import pickle
from PIL import Image
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from utils.embeddings import get_text_embedding, get_image_embedding

# Load vector stores
text_index = faiss.read_index("vectorstore/text_index.faiss")
with open("vectorstore/text_metadata.pkl", "rb") as f:
    text_metadata = pickle.load(f)

image_index = faiss.read_index("vectorstore/image_index.faiss")
with open("vectorstore/image_metadata.pkl", "rb") as f:
    image_metadata = pickle.load(f)

class InputState(TypedDict, total=False):
    text: str
    image: Image.Image

class State(TypedDict):
    input: InputState
    retrieved: List
    output: str

def retrieve(state: State):
    """Retrieves documents from the appropriate vector store."""
    
    input_dict = state["input"]
    if input_dict.get("text"):
        query_vec = get_text_embedding(input_dict["text"])
        D, I = text_index.search(np.expand_dims(query_vec, 0), k=3)
        retrieved_docs = [text_metadata[i] for i in I[0]]
    elif input_dict.get("image"):
        query_vec = get_image_embedding(input_dict["image"])
        D, I = image_index.search(np.expand_dims(query_vec, 0), k=3)
        retrieved_docs = [image_metadata[i] for i in I[0]]
    else:
        raise ValueError("No input provided in the state 'input' key")
    
    return {"retrieved": retrieved_docs}
    
# Build graph with the state schema
graph_builder = StateGraph(State)

graph_builder.add_node("retrieve", retrieve)
graph_builder.set_entry_point("retrieve")
graph_builder.set_finish_point("retrieve")

rag_pipeline = graph_builder.compile()
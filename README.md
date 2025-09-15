# A Multi-Modal RAG Application Built with Streamlit Using a Lightweight Ollama Model
This is a simple multi-modal RAG application built with Streamlit and powered by a lightweight Ollama model. Place your text and image documents in the data/documents/ folder, and the app will build a vector store from them.

# Installation Instruction
conda create -n multi-modal-rag-web python=3.10 -y
conda activate multi-modal-rag-web
cd multi-modal-rag-web
pip install -r requirements.txt
python ingest.py
streamlit run app.py
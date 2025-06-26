import os
import fitz  # PyMuPDF
import streamlit as st
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# Page setup
st.set_page_config(page_title="HR Policy Chatbot", layout="centered")
st.title("💼 HR Policy Chatbot")
st.markdown("Hey There! I am your HR assistant! If you have any queries regarding the HR policies, feel free to ask me!")

# Load API key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Set folder path
pdf_folder_path = "rag_pdf"

# Load PDFs
@st.cache_resource
def load_pdfs(folder_path):
    texts = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".pdf") or filename.endswith(".txt"):
            with fitz.open(os.path.join(folder_path, filename)) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                texts.append(text)
    return texts

# Semantic Chunking
@st.cache_resource
def semantic_chunking(texts, threshold=0.7):
    paras = []
    for text in texts:
        for para in text.split("\n\n"):
            if para.strip():
                paras.append(para.strip())

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(paras, convert_to_tensor=True)

    chunks = []
    current_chunk = [paras[0]]
    for i in range(1, len(paras)):
        sim = util.cos_sim(embeddings[i - 1], embeddings[i]).item()
        if sim > threshold:
            current_chunk.append(paras[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [paras[i]]
    chunks.append(" ".join(current_chunk))

    return chunks, model

# Retrieve relevant chunks
def get_relevant_chunks(query, chunks, model, top_k=2):
    query_emb = model.encode(query, convert_to_tensor=True)
    chunk_embs = model.encode(chunks, convert_to_tensor=True)
    sims = util.cos_sim(query_emb, chunk_embs)[0]
    top_indices = sims.argsort(descending=True)[:top_k]
    return [chunks[i] for i in top_indices]

# Build prompt
def build_prompt(query, context):
    return f"""
You are a helpful HR assistant. Answer strictly using the below policy context. Do not hallucinate or guess. 
If answer is not in the context, say: 'Sorry! The provided content doesn’t have the information you are looking for.'

Context:
{context}

Question: {query}

Answer:"""

# OpenRouter API Call
def ask_openrouter(prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    try:
        return response.json()['choices'][0]['message']['content'].strip()
    except:
        return "⚠️ Sorry! Couldn’t fetch an answer."

# Initialize and run
with st.spinner("Loading knowledge base..."):
    all_texts = load_pdfs(pdf_folder_path)
    chunks, model = semantic_chunking(all_texts)

query = st.text_input("Enter your HR policy question:")

if query:
    with st.spinner("Thinking..."):
        relevant = get_relevant_chunks(query, chunks, model)
        context = "\n\n".join(relevant)
        prompt = build_prompt(query, context)
        answer = ask_openrouter(prompt)
        st.markdown("**Answer:**")
        st.write(answer)

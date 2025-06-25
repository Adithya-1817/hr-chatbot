import os
import fitz
import faiss
import numpy as np
import streamlit as st
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="HR Policy Chatbot", layout="centered")
st.title("🤖 HR Policy Chatbot")
st.markdown("Hey There! I am your HR assistant! If you have any queries regarding the HR policies, Feel free to ask me!")

load_dotenv(".env")
api_key = os.getenv("OPENROUTER_API_KEY")

pdf_folder_path = "rag_pdf"

@st.cache_resource
def load_pdfs(folder_path):
    all_text = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            all_text.append(text)
    return all_text

@st.cache_resource
def semantic_chunking(texts, threshold=0.7):
    paragraphs = []
    for text in texts:
        for para in text.split("\n\n"):
            if para.strip():
                paragraphs.append(para.strip())

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(paragraphs)
    chunks = []
    current_chunk = [paragraphs[0]]

    for i in range(1, len(paragraphs)):
        sim = util.cos_sim(embeddings[i - 1], embeddings[i])
        if sim > threshold:
            current_chunk.append(paragraphs[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [paragraphs[i]]
    chunks.append(" ".join(current_chunk))

    return chunks, model

@st.cache_resource
def build_faiss_index(chunks, _model):
    embeddings = _model.encode(chunks)
    embeddings = np.array(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def retrieve_chunks(query, chunks, index, model, top_k=2):
    query_embedding = model.encode([query])
    scores, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

def build_prompt(query, context_chunks, max_chars=6000):
    context = "\n\n".join(context_chunks)
    if len(context) > max_chars:
        context = context[:max_chars]
    return f"""You are an expert HR policy assistant. Use only the following context from official HR documentation to answer the user's question. Do not guess or hallucinate. If the answer is not found, say 'Sorry, I couldn't find that in the documents.'

Context:
{context}

Question: {query}

Answer:"""

def query_openrouter(prompt):
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
    res_json = response.json()
    if response.status_code == 200 and 'choices' in res_json:
        return res_json['choices'][0]['message']['content']
    return "⚠️ Error: " + res_json.get('error', {}).get('message', 'Unknown error.')

with st.spinner("Loading HR document and building memory..."):
    all_texts = load_pdfs(pdf_folder_path)
    chunks, model = semantic_chunking(all_texts)
    index, _ = build_faiss_index(chunks, model)

query = st.text_input("Enter your HR policy question:")

if query:
    with st.spinner("Finding the best answer..."):
        context_chunks = retrieve_chunks(query, chunks, index, model)
        prompt = build_prompt(query, context_chunks)
        answer = query_openrouter(prompt)
        st.markdown("**Answer:**")
        st.write(answer.strip())

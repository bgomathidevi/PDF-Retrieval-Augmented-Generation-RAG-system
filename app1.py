import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 1: Extract text from NCERT PDF 
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

# Step 2: Split the text into chunks 
def split_into_chunks(text, chunk_size=100):
    sentences = text.split('.')
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + '.'
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '.'
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Step 3: Embed the chunks using SentenceTransformer
def embed_chunks(chunks):
    embeddings = model.encode(chunks)
    return np.array(embeddings).astype('float32')

# Step 4: Function to generate answer using LLM
def generate_answer(contexts, query):
    # Create a formatted string for the context to give more detailed responses
    context_string = "\n".join([f"- {context}" for context in contexts])
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or gpt-4 if available
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Using the context below, provide detailed points to answer the question:\n\nContext:\n{context_string}\n\nQuestion: {query}"}
        ]
    )
    answer = response.choices[0].message['content']
    return answer

# Streamlit app
st.title("NCERT PDF RAG System")

# File uploader for NCERT PDF
uploaded_file = st.file_uploader("Upload an PDF", type="pdf")

if uploaded_file:
    # Extract and display the PDF text
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.write("PDF successfully loaded!")

    # Split text into chunks
    chunks = split_into_chunks(pdf_text)
    
    # Embed the chunks and create FAISS index
    embedding_matrix = embed_chunks(chunks)
    d = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embedding_matrix)
    st.write("Text successfully split into chunks and embedded!")

    # Query input from the user
    query = st.text_input("Enter your query:")
    
    if query:
        # Embed the query and perform FAISS search
        query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
        k = 5  # Number of top results to return
        distances, indices = index.search(query_embedding, k)

        # Gather context from top results
        contexts = [chunks[i] for i in indices[0]]

        # Generate an answer using the LLM
        answer = generate_answer(contexts, query)

        # Display the answer
        st.write("Generated Answer:")
        st.write(answer)

        # Display the top 5 chunks
        st.write("Top 5 Relevant Chunks:")
        for i in range(k):
            st.write(f"{i+1}: {chunks[indices[0][i]]}")
else:
    st.write("Please upload a PDF file.")

# PDF-Retrieval-Augmented-Generation-RAG-system

Project Overview
This project is designed to extract and understand information from NCERT PDFs using a combination of Natural Language Processing (NLP) techniques. The system lets users upload an  PDF and then ask questions about the content. It returns relevant chunks of text from the PDF and generates a detailed, multi-point answer using a Large Language Model (LLM), specifically GPT-3.5 or GPT-4.

Key components of the project include:
PDF Text Extraction: Extracts text from NCERT PDFs.
Text Chunking: Splits the extracted text into smaller, meaningful chunks.
Text Embedding: Converts chunks of text into numerical embeddings using the SentenceTransformer model.
FAISS Index: A search index used to efficiently find the most relevant chunks based on the user's query.
LLM Integration: Uses GPT-3.5 or GPT-4 to generate a detailed, multi-point response based on the most relevant chunks of the document.

Detailed Explanation of Each Step:
1. PDF Text Extraction
The first step is to extract the raw text from the uploaded NCERT PDF file using the PyPDF2 library. Here’s how this works:
The extract_text_from_pdf function reads all pages from the PDF and extracts the text, storing it as a long string. This will later be split into smaller chunks for efficient processing and retrieval.

2. Text Chunking
Once the text is extracted, it’s often too long and unstructured to use directly. To overcome this, we split the text into smaller chunks (e.g., 100 characters or so) using the split_into_chunks function. This process involves:
Breaking the text into sentences.
Collecting these sentences into small, coherent chunks. This helps retain context when embedding the text.

3. Text Embedding
Embedding refers to converting text data into dense vector representations, which can be processed by the system. In this project:
The SentenceTransformer model (all-MiniLM-L6-v2) is used to convert text chunks into numerical vectors.
These vectors serve as the basis for similarity comparisons later in the project.
Once the text is embedded, the embeddings are stored in a numpy array for further processing.

4. FAISS Index for Retrieval
We use the FAISS library to create an index of these embeddings. FAISS is a high-performance tool for similarity search, allowing us to:
Store the embeddings: The dense vectors representing the text chunks are added to the FAISS index.
Efficiently search: When a user enters a query, it is also converted into an embedding, and FAISS searches the index for the most similar (i.e., relevant) chunks of text. This allows for quick, efficient retrieval of the relevant content.

5. Query Processing and Retrieval
When a user enters a query:
The system embeds the query using the same SentenceTransformer model.
The query embedding is then compared with the embeddings of the document chunks in the FAISS index.
The top 5 most relevant chunks of text are retrieved based on their similarity to the query.

6. Answer Generation using GPT (LLM Integration)
This is where RAG (Retrieval-Augmented Generation) comes into play:
The top 5 relevant chunks of text are passed to an LLM, such as GPT-3.5 or GPT-4, to generate a detailed answer.
The generate_answer function sends a prompt to the LLM, containing the user’s query and the retrieved chunks of text. This allows the model to generate an answer that is both accurate and contextual, based on the content of the document.
Here’s what happens during this step:
The relevant chunks are combined into a single prompt formatted as bullet points.
The system asks GPT to generate a multi-point answer using this context. The LLM is asked to provide at least three points, making the answer more comprehensive.

8. User Interface with Streamlit
The project uses Streamlit to provide a simple and interactive web interface. The main features of the UI include:
PDF File Upload: Users can upload anPDF to be processed.
Query Input: Users can type in a question related to the uploaded PDF.
Answer and Context Display: After processing the query, the system displays the generated answer along with the top 5 relevant chunks from the document.

Detailed Example of How the System Works:
A user uploads an NCERT PDF.
The system extracts the text from the PDF, splits it into chunks, and embeds those chunks into dense vectors.
When the user enters a query (e.g., "Explain photosynthesis"), the system embeds the query and uses FAISS to find the 5 most relevant chunks from the document.
These chunks are passed to GPT-3.5 (or GPT-4), which generates a detailed, multi-point answer.
The system returns both the answer and the top 5 chunks, allowing the user to see the most relevant parts of the document.

Why This Project is Valuable:
Combines NLP and LLMs: The project demonstrates practical usage of state-of-the-art NLP techniques (sentence embeddings, FAISS indexing) alongside powerful LLMs (GPT-3.5 or GPT-4).
Scalability: It can be extended to larger document collections or adapted to other types of PDFs (research papers, manuals, etc.).
Efficiency: FAISS ensures that document searches are fast and scalable, even for large documents.
Comprehensive Answers: By using GPT to generate multi-point responses, the system provides more thorough and understandable answers to user queries.

In Summary:
This project uses RAG (Retrieval-Augmented Generation) to allow users to ask questions about an NCERT PDF and get detailed, multi-point answers.
The system first breaks the PDF into chunks, embeds them using a transformer model, and then uses FAISS to retrieve relevant chunks based on the user’s query.
GPT-3.5 or GPT-4 is then used to generate a coherent, multi-point answer, providing a helpful, interactive way to explore educational content.
This explanation gives you a strong foundation to explain the technical and functional aspects of the project during an interview.







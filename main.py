import os
import streamlit as st
import pickle
import openai
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("PDF-Bot: PDF Research Tool 📈")
st.sidebar.title("Upload PDF Files")

# List existing embedding files
embedding_files = [f for f in os.listdir() if f.endswith(".pkl")]
use_existing = st.sidebar.radio(
    "Select an option:", ["Use Existing Embedding", "Create New Embedding"])

if use_existing == "Use Existing Embedding":
    selected_file = st.sidebar.selectbox(
        "Select an existing embedding file:", embedding_files)
else:
    embedding_filename = st.sidebar.text_input(
        "Enter embedding file name (without extension):")
    embedding_filename = f"{embedding_filename}.pkl" if embedding_filename else "faiss_store.pkl"

# File upload for new embeddings
pdf_files = st.sidebar.file_uploader(
    "Upload PDF files", type=["pdf"], accept_multiple_files=True)
process_pdf_clicked = st.sidebar.button("Process PDFs")

main_placeholder = st.empty()

if process_pdf_clicked and pdf_files:
    all_docs = []
    for pdf_file in pdf_files:
        temp_pdf_path = f"temp_{pdf_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        main_placeholder.text(f"Loading {pdf_file.name} ...")

        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()
        all_docs.extend(docs)
        os.remove(temp_pdf_path)

    main_placeholder.text("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    split_docs = [Document(page_content=chunk, metadata=doc.metadata)
                  for doc in all_docs for chunk in text_splitter.split_text(doc.page_content)]

    main_placeholder.text("Creating embeddings...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    with open(embedding_filename, "wb") as f:
        pickle.dump(vectorstore, f)

    main_placeholder.text(f"Embeddings saved as {embedding_filename}")

# Accept a user query
query = main_placeholder.text_input("Question: ")
if query:
    if use_existing == "Use Existing Embedding" and selected_file:
        embedding_filename = selected_file

    if os.path.exists(embedding_filename):
        with open(embedding_filename, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(query)

        # Organize extracted text by page
        page_contents = defaultdict(list)

        for doc in docs:
            page_number = doc.metadata.get("page", "Unknown")
            cleaned_text = doc.page_content.strip()
            page_contents[page_number].append(cleaned_text)

        # Display formatted output
        st.header("Sources: ")
        for page, contents in sorted(page_contents.items()):
            st.subheader(f"Page {page}")
            st.write("\n\n".join(contents))

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert research assistant."},
                {"role": "user", "content": f"Answer the following question using these documents: {docs}.\nQuestion: {query}"}
            ]
        )
        answer2 = response["choices"][0]["message"]["content"]
        st.header("My Analysis: ")
        st.write(answer2)
    else:
        st.error("No FAISS database found. Please process PDFs first.")

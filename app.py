import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load Groq API and GOOGLE_API_KEY from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-It")

prompt = ChatPromptTemplate.from_template(

    """
Answer the questions based on the proviede context only.
Please provide the most accurate response based on the question
<content>
{context}
<content>
Question: {input}

"""
)

def vector_embeddings():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        st.session_state.loader = PyPDFDirectoryLoader('./pdfdata') # Load PDFs from the pdfdata directory
        st.session_state.docs = st.session_state.loader.load() # Load the documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) # Split the documents into chunks
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings) # Create the vectors



prompt1 = st.text_input('Enter your text from Documents.')


if st.button('Documents Embeddings'):
    vector_embeddings()
    st.write('Documents Embeddings created successfully')


if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    response = retriever_chain.invoke({'input': prompt1})
    st.write(response['answer'])


    # With a streamlit expander 
    with st.expander('Document Similarity Search'):
        # Find the relevant chunk
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------------------')
import streamlit as st 
import os
from dotenv import load_dotenv
load_dotenv()
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document 
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS



my_api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=my_api_key)


gemini_model = genai.GenerativeModel('gemini-2.0-flash')
st.cache_data(show_spinner='Loading Embedding Model...')

def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

with st.spinner('Loading Embedding Model...'):
    embedding_model = load_embedding_model()   
    
st.header('RAG Assistant :blue[Using Embedding & Gemini LLM]')
st.subheader('Your Intelligent Document Assistant!')

st.write('Done')


uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])


if uploaded_file:
    st.write('Uploaded successfully!')

if uploaded_file:
    pdf = PdfReader(uploaded_file)
    raw_text = ''
    
    for page in pdf.pages:
        raw_text += page.extract_text()
        
    st.write('Extracted successfully!')
    
    if raw_text.strip():
        doc = Document(page_content=raw_text)
        CharacterTextSplitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        chunk_text = CharacterTextSplitter.split_documents([doc])

        text = [i.page_content for i in chunk_text]

        vector_database = FAISS.from_texts(texts=text, embedding=embedding_model)
        retrive = vector_database.as_retriever()
        st.success("Document loaded and processed successfull...Ask your question now!")
        
        query = st.text_input("Ask a query here:")   
        
        if query:
            with st.chat_message("human"):
                 with st.spinner("Thinking..."):
                     
                    retrive_docs = retrive.get_relevant_documents(query)
                    content = '\n\n'.join([i.page_content for i in retrive_docs])
                    
                    
                    prompt = f"""
                    You are an AI expert in answering questions based on the provided context.
                    Context: {content}
                    Question: {query}
                    Answer:
                    """
                    response = gemini_model.generate_content(prompt)
                    
                    st.markdown('### :green[Result]')
                    st.write(response.text)   
    else:
        st.warning('Drop the file with text content only! ')
import streamlit as st
import os
from dotenv import load_dotenv  
load_dotenv()  
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings # to get embedding model
from langchain_core.documents import Document # it store the text and its metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter # to split the raw text into smaller chunks
from langchain_community.vectorstores import FAISS # to store the emdedded -text chunks in a vector database for similarity search.

key=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)

gemini_model=genai.GenerativeModel('gemini-1.5-flash')


def load_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
with st.spinner('Loading embeddings model...'):
    embeddings_model = load_embeddings_model()
    
st.header('RAG Assistant :blue[Using Embeddings & Gemini LLM]')
st.subheader('Your Intelligent Document Assistant')

# st.write('DONE')

uploaded_file = st.file_uploader("Upload a PDF file",accept_multiple_files=True, type=["pdf"])


# if uploaded_file:
#     st.write("File uploaded successfully!")

if uploaded_file:
    raw_text = ""
    for file in uploaded_file:
        pdf = PdfReader(file)
        for page in pdf.pages:
            page_text = page.extract_text()
            # Handle cases where extract_text returns None
            if page_text:
                raw_text += page_text + "\n"

    
    if raw_text.strip():
        doc=Document(page_content=raw_text)
        spiltter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunck_text=spiltter.split_documents([doc])
        
        # text=[i.page_content for i in chunck_text]
        
        vector_db= FAISS.from_documents(chunck_text, embeddings_model)
        retrive=vector_db.as_retriever()
        
        st.success("Document Processed Successfully...Ask your questions")
        
        query=st.text_input('Enter your query here👇')
        
        if query:
            with st.chat_message("assistant"):
                with st.spinner('Analayzing the Document'):
                    relvent_doc = retrive.invoke(query)
                    
                    content = '\n\n'.join([i.page_content for i in relvent_doc])
                    
                    prompt = f'''
                    You are a expert in asnwering questions based on the attention is all you need documnet.
                    use the content and answer the query.If your unsure ypu should say "I don't know" and not make up an answer.                    
                    content: {content}
                    query: {query}
                    result :
                    '''
                    
                    response=gemini_model.generate_content(prompt)
                    st.markdown('### :green[result]')
                    st.write(response.text)
    else:
        st.warning("Drop the file in PDF format.")
        
        
# --- CONTACT INFORMATION ---
# st.write("---")
st.subheader("Contact Me")

# Replace with your actual LinkedIn URL and email address
linkedin_url = "https://www.linkedin.com/in/anubhav-v/"
email_address = "anubhavnaidu@gmail.com"

# HTML for the links
linkedin_link = f'<a href="{linkedin_url}" target="_blank">LinkedIn Profile</a>'
email_link = f'<a href="mailto:{email_address}">Send an Email</a>'

# Using columns to display them side-by-side
col1, col2  = st.columns(2)

with col1:
    st.markdown(f'[in] : {linkedin_link}', unsafe_allow_html=True)

with col2:
    st.markdown(f'✉️ : {email_link}', unsafe_allow_html=True)
    
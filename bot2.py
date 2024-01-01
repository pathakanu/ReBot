import streamlit as st
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader

load_dotenv()

def generate_response(uploaded_file, query):
    loader = PyPDFLoader(uploaded_file)
    documents = loader.load_and_split()
    
    #splitting documents
    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=10)
    docs = text_splitter.create_documents(documents)

    embedding_function = OpenAIEmbeddings()

    db = FAISS.from_documents(docs, embedding=embedding_function)
    db.save_local('./faiss/')

    db = FAISS.load_local(
        folder_path="./faiss/",
        index_name="index",
        embeddings=OpenAIEmbeddings(),
    )
    retriever = db.as_retriever(k=3)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    template = """You are an Resource Management chatbot designed to assist company resource team. 
            give the answer based on given client requirement documents and current market trends on tech stack.
            Below is the requirement given from which you need to find out tech stack required with their ratings and overall experience
            {context}
            Human: Tech stack for QA engineer
            Chatbot:{{
                            {{
                            TechStack: Qlik,
                            Rating: 4
                            }},
                            {{
                            TechStack: SQL,
                            Rating: 3
                            }},
                            {{
                            TechStack: NoSQL,
                            Rating: 3
                            }},
                            {{
                            TechStack: Jira,
                            Rating: 3
                            }},
                            {{
                            TechStack: Scrum/Agile,
                            Rating: 4
                            }}
                            experience:{{
                            year: 2+}}
                            }}

            Please give the answer in JSON format based on skills and experience which we mentioned in above example

            Human: {question}
            Chatbot:"""

    prompt = PromptTemplate(input_variables=["context","question"],
                        template=template)


    bot = qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='pdf')
if uploaded_file is not None:
    filename = uploaded_file.name
    
else:
    path_in = None
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted:
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, query_text)
            result.append(response)

if len(result):
    st.info(response)
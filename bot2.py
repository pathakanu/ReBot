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

def generate_response(file_path, query):
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split()
    
    #splitting documents
    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

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
                            Rating: 5
                            }},
                            {{
                            TechStack: SQL,
                            Rating: 4
                            }},
                            {{
                            TechStack: NoSQL,
                            Rating: 4
                            }},
                            {{
                            TechStack: Jira,
                            Rating: 5
                            }},
                            {{
                            TechStack: Scrum/Agile,
                            Rating: 4
                            }}
                            experience:{{
                            year: 5+}}
                            }}

            Please give the answer in JSON format based on skills and experience which we mentioned in above example. Make sure the above is only example, don't add this data with new query.
            Match the query tech stack with document tech stack, if those doesn't match return with an answer "I don't have information about this".
            If the given query doesn't match the content of context, just return with an answer "I don't have information about this". Do not give answer if the information is not available in provided document.
            Human: {question}
            Chatbot:"""

    prompt = PromptTemplate(input_variables=["context","question"],
                        template=template)


    bot = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )

    result = bot({'query': query})
    return result['result']

# Page title
st.set_page_config(page_title='🦜🔗 TFT Resource BOT')
st.title('🦜🔗 TFT Resource BOT')

file_path = "/home/anurag/Documents/langchain/ReBot/requirement"

# File upload
uploaded_file = st.file_uploader('Upload an article', type='pdf')
if uploaded_file is not None:
    filename = uploaded_file.name
    file_path = os.path.join(file_path,filename)
else:
    path_in = None

with st.sidebar:
    req = st.selectbox(
        "Choose a Query",
        ("Tech Stack", "Experience", "Qualification")
    )

    domain = st.selectbox(
        "Choose a Domain",
        ("QA Engineer", "Wordpress Developer", "UI/UX Developer")
    )

    template_1 = f"Tell me {req} for {domain}"

    st.write(template_1)

# Query text
# query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file, key='template_1')
query_text = st.text_input('Enter your question:', value=template_1 if uploaded_file else '', key='query')

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted:
        with st.spinner('Searching...'):
            response = generate_response(file_path, query_text)
            result.append(response)

if len(result):
    st.info(response)
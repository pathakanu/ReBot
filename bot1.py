from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

# #loading documents
# loader = DirectoryLoader("./requirement/")
# documents = loader.load()

# #Splitting text
# text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=10)
# docs = text_splitter.split_documents(documents)

# embedding_function = OpenAIEmbeddings()

# db = FAISS.from_documents(docs, embedding=embedding_function)
# db.save_local('./faiss/')

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

while True:
    query = input("Hi, Please enter your query (or 'exit' to quit): ")
    
    if query == "exit":
        break
    
    result = bot({'query': query})
    # json_output = parser.parse(result['result'])
    print(result['result'])
    # print(json_output)




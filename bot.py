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
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5})

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# # Define your desired data structure.
# class Tech(BaseModel):
#     TechStack: str = Field(description="It mentions the tech stack")
#     Rating: str = Field(description="It mentions the rating of the tech in scale of 1-5")

#     # You can add custom validation logic easily with Pydantic.
#     @validator("TechStack")
#     def question_ends_with_question_mark(cls, field):
#         if field[-1] != "?":
#             raise ValueError("Badly formed question!")
#         return field

# parser = PydanticOutputParser(pydantic_object=Tech)

# examples = [
#     {"techstack": "WordPress", "Rating": "3"},
#     {"techstack": "PHP", "Rating": "2"},
#     {"techstack": "JavaScript", "Rating": "4"},
#     {"techstack": "jQuery", "Rating": "3"},
#     {"techstack": "RESTful", "Rating": "5"},
#     {"techstack": "Git", "Rating": "5"},
#     {"techstack": "React.js", "Rating": "4"},
#     {"techstack": "SQL", "Rating": "3"},
#     {"techstack": "NoSQL", "Rating": "3"},
#     {"techstack": "JIRA", "Rating": "3"},
#     {"techstack": "Adobe CC", "Rating": "3"},  
#     {"techstack": "Adobe XD", "Rating": "3"},
#     {"techstack": "Figma", "Rating": "4"},
#     {"techstack": "Adobe CC", "Rating": "3"},
# ]

# example_formatter_template = """
# techstack: {techstack}
# Rating: {Rating}\n
# """

# example_prompt = PromptTemplate(
#     input_variables=["techstack","Rating"],
#     template=example_formatter_template,
# )

# few_shot_prompt = FewShotPromptTemplate(
#     examples=examples,
#     example_prompt=example_prompt,
#     prefix="Here are some examples of TechStacks and Rating associated with them:\n\n",
#     suffix="""\nYou are an Resource Management chatbot designed to assist company resource team. 
#             give the answer based on given client requirement documents and current market trends on tech stack.
#             Below is the requirement given from which you need to find out tech stack required with their ratings and overall experience
#             {context}
#             {format_instructions}
#             Human: {question}
#             Chatbot:""",
#     input_variables=["context","question"],
#     example_separator="\n",
#     partial_variables={"format_instructions": parser.get_format_instructions()},
# )

# formatted_prompt = few_shot_prompt.format(context="context",question="question")

# bot = qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=retriever,
#     chain_type_kwargs={"prompt": few_shot_prompt},
# )

# while True:
#     query = input("Hi, Please enter your query (or 'exit' to quit): ")
    
#     if query == "exit":
#         break
    
#     result = bot({'query': query})
#     # json_output = parser.parse(result['result'])
#     print(result['result'])
#     # print(json_output)


query = input("Hi, Please enter your query (or 'exit' to quit): ")
print(db.similarity_search(query))
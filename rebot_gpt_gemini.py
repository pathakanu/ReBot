from dotenv import load_dotenv
load_dotenv()
import PyPDF2
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import base64
import fitz
import streamlit as st
import os
import io
import json
from PIL import Image 
import pdf2image
import google.generativeai as genai

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

document = []

def get_gemini_response(input,pdf_cotent,prompt):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content([input,pdf_content,prompt])
    # print(response.text)
    return response

def get_gemini_response_vision(input,pdf_cotent,prompt):
    model=genai.GenerativeModel('gemini-pro-vision')
    response=model.generate_content([input,pdf_content[0],prompt])
    # print(response.text)
    return response

def summarize_document(pdf_content,prompt):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content([pdf_content,prompt])
    # print(response.text)
    return response.text

def get_gemini_list(pdf_content,prompt):
    model=genai.GenerativeModel('gemini-pro-vision')
    response=model.generate_content([pdf_content,prompt])
    # print(response.text)
    return response.text

def summarize_document_vision(pdf_content,prompt):
    model=genai.GenerativeModel('gemini-pro-vision')
    response=model.generate_content([pdf_content[0],prompt])
    # print(response.text)
    return response.text

def run_chain(job_description, cv_summarized, stack):
    template = """You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of {stack}, 
    your task is to evaluate the resume against the provided job description. State name of the candidate, candidate's total experience, Give the percentage of match based on the years of experience and tech stacks known by the candidate, comparing from the given job description. Mention the overall tech skills known by the candidate and add reason behind the percentage value. 

    Below is the Job Description:
    {job_description}
    \n
    Below is the Candidate's CV/Resume:
    {cv_summarized}
    \n
    Below is the provided sample format in which you need to give output.
    Example: {{
        "Candidate Name": "Rahul Robin",
        "Total Experience": "6+ years",
        "Match%": "58",
        "Known Skills": "MySQL, React.js"
        "Reason": "Could be better with more UI/UX knowledge"
    }}

    Please read the given summarized CV thoroughly
    Make sure you don't replicate the example. Based on the given resume, provide the output.
    """

    prompt = PromptTemplate(input_variables=["stack","job_description","cv_summarized"], template=template)

    chain = LLMChain(llm=llm, prompt=prompt)

    output = chain.invoke({"stack":stack, "job_description":job_description, "cv_summarized":cv_summarized})

    return output['text']

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        ## Convert the PDF to image
        # images = pdf2image.convert_from_path(uploaded_file)
        images=pdf2image.convert_from_bytes(uploaded_file.read())

        first_page=images[0]
        # first_page=merge_images(images)

        # first_page.save('output_image.png', 'PNG')

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

def document_load(file_path):
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()
    resume_dict = {}

    # Iterate through each document and store its content in the dictionary
    for i, document in enumerate(pages, start=1):
        page_key = f"Page_{i}"
        resume_dict[page_key] = document.page_content
    
    resume_json = json.dumps(resume_dict, indent=2)
    return resume_json

    # text = ""
    
    # try:
    #     # Open the PDF file in binary mode
    #     with open(file_path, 'rb') as file:
    #         # Create a PDF reader object
    #         pdf_reader = PyPDF2.PdfFileReader(file)

    #         # Iterate through all pages and extract text
    #         for page_num in range(pdf_reader.numPages):
    #             page = pdf_reader.getPage(page_num)
    #             text += page.extractText()

    # except Exception as e:
    #     print(f"Error: {e}")

    # return text


def merge_images(images):
    # Assuming all images have the same dimensions
    width, height = images[0].size

    # Create a new image with a height that accommodates all pages stacked vertically
    merged_image = Image.new('RGB', (width, height * len(images)))

    # Paste each page onto the merged image
    for i, image in enumerate(images):
        merged_image.paste(image, (0, i * height))

    return merged_image


## Streamlit App

st.set_page_config(page_title="ATS Resume EXpert")
st.header("TFT Resume Screening Bot")
# input_text=st.text_area("Job Description: ",key="input")
input_text = "The position of Python Developer (GraphQL) requires a minimum of 6 years of experience and expertise in Python, Django, and GraphQL. The role involves designing and building GraphQL schema, developing APIs, troubleshooting issues, and writing clean and reusable code. The candidate should have excellent communication skills, problem-solving abilities, and attention to detail. Think Future Technologies (TFT) is a technology services company that partners with clients to provide optimal solutions and delivery for their business needs."


summarize_prompt = """
Please summarize the given document mentioning total experience ,skillsets and qualification, don't include "About Us" Section which contains company details. Only mention the required data in JD, summarize within 100 words"""

summarize_resume_prompt = """
Please summarize the given resume mentioning name of the candidate, total working experience ,skillsets and qualification. Particulary summarize the projects section in very short and precise and in bullet points. If candidate experience is not mentioned, then try to calculate candidate's experience from date of projects ."""

uploaded_file=st.file_uploader("Upload your Job Description(PDF)...",type=["pdf"])
if uploaded_file is not None:
    with st.spinner('Uploading...'):
        filename = uploaded_file.name
        file_path = os.path.join("requirement/",filename)
        pdf_document = document_load(file_path)
        job_description = summarize_document(pdf_document,summarize_prompt)
        st.write("PDF Uploaded Successfully")


stack = st.selectbox("Choose JD Domain",("Data Science","Blockchain","Web Development","DevOps","Backend Developer","UI/UX","React-Native"))
submit = st.button("Submit")


if submit:
    with st.spinner('Searching...'):
        if document is not None:
            directory_contents = os.listdir("react-resume/")
            summarize_cv=[]
            for item in directory_contents:
                file_path = os.path.join("/home/anurag/Documents/langchain/ReBot/react-resume",item)
                pdf_content=document_load(file_path)
                cv_summarized = summarize_document(pdf_content,summarize_resume_prompt)
                summarize_cv.append(cv_summarized)
                response = run_chain(job_description, cv_summarized, stack)
                # response=get_gemini_response(prompt,cv_summarized,job_description)
                try:
                    json_object = json.loads(response)
                    print(json_object)
                    st.write(json_object)
                    document.append(json_object)

                except json.decoder.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")

            for summary in summarize_cv:
                with open("summarized_cv.txt", "a") as file:
                    file.write(summary)
                    file.write('\n\n\n\n')
            sorted_candidates = sorted(document, key=lambda x: int(x['Match%']), reverse=True)
            top_5_candidates = sorted_candidates[:5]
            st.subheader("Top 5 Candidate")
            for candidate in top_5_candidates:
                st.write(f"Candidate Name: {candidate['Candidate Name']}, Match%: {candidate['Match%']}")
        else:
            st.write("Please upload the resume")
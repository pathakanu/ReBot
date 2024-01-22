from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
import base64
import fitz
import streamlit as st
import os
import io
import json
from PIL import Image 
import pdf2image
import google.generativeai as genai

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
Please summarize the given resume mentioning total working experience ,skillsets and qualification. Exclude the project section of the resume, mention only skills which present in roles and responsibilities"""

uploaded_file=st.file_uploader("Upload your Job Description(PDF)...",type=["pdf"])
if uploaded_file is not None:
    with st.spinner('Uploading...'):
        filename = uploaded_file.name
        file_path = os.path.join("requirement/",filename)
        pdf_document = document_load(file_path)
        summarize_text = summarize_document(pdf_document,summarize_prompt)
        # st.write(summarize_text)
        st.write("PDF Uploaded Successfully")

# submit1 = st.button("Tell Me About the Resume")

#submit2 = st.button("How Can I Improvise my Skills")

# submit3 = st.button("Percentage match")
# submit4 = st.button("Tell me name")
stack = st.selectbox("Choose JD Domain",("Data Science","Blockchain","Web Development","DevOps","Backend Developer","UI/UX"))
submit = st.button("Submit")


prompt = f"""You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of {stack}, 
your task is to evaluate the resume against the provided job description. Give me the percentage of match based on the years of experience and tech stacks from the given job description. Also, mention keywords missing from the resume compared to the job description. Below is the provided sample format in which you need to give output. 

Example: {{
    "Candidate Name": "Rahul Robin",
    "Total Experience": "6+ years",
    "Domain Expertise": "Banking and Finance",
    "Match%": "58",
    "Known Skills": "MySQL, React.js"
}}

Please read the given summarized CV thoroughly
Make sure you don't replicate the example. Based on the given resume, provide the output.
"""

input_prompt1 = """
 You are an experienced Technical Human Resource Manager,your task is to review the provided resume against the job description. 
  Please share your professional evaluation on whether the candidate's profile aligns with the role. 
 Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt3 = """
You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
the job description. Output Example: {
Candidate name: Rahul Robin Jha
Match: 68%
Known skills: Python, Django, Solidity 
}
The output should be keywords missing and last final thoughts.
"""

input_prompt4 = """
You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
your task is to evaluate the resume against the provided job description. Tell me the name of the candidate
"""

input_prompt5 = f"""
You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of {stack}, 
The provided document is dictornary of candidates with %match and missing skills, your task is to evaluate the resume based on match%. Tell me the top 5 candidates for screening based on the list
"""

# if submit1:
#     if uploaded_file is not None:
#         pdf_content=input_pdf_setup(uploaded_file)
#         response=get_gemini_response(input_prompt1,pdf_content,input_text)
#         st.subheader("The Repsonse is")
#         st.write(response)
#     else:
#         st.write("Please upload the resume")

# elif submit3:
#     if uploaded_file is not None:
#         pdf_content=input_pdf_setup(uploaded_file)
#         response=get_gemini_response(input_prompt3,pdf_content,input_text)
#         st.subheader("The Repsonse is")
#         st.write(response)
#     else:
#         st.write("Please upload the resume")

# elif submit4:
#     if uploaded_file is not None:
#         pdf_content=input_pdf_setup(uploaded_file)
#         response=get_gemini_response(input_prompt4,pdf_content,input_text)
#         st.subheader("The Repsonse is")
#         st.write(response)
#     else:
#         st.write("Please upload the resume")

if submit:
    with st.spinner('Searching...'):
        if document is not None:
            directory_contents = os.listdir("resumes/")
            summarize_cv=[]
            for item in directory_contents:
                file_path = os.path.join("/home/anurag/Documents/langchain/ReBot/resumes",item)
                pdf_content=document_load(file_path)
                pdf_summarized = summarize_document(pdf_content,summarize_resume_prompt)
                summarize_cv.append(pdf_summarized)
                
            # print(pdf_content)
            # print(pdf_summarized)
                # st.write(pdf_summarized)
                response=get_gemini_response(summarize_text,pdf_summarized,prompt)
                try:
                    json_object = json.loads(response.text)
                    print(json_object)
                    st.write(json_object)
                    document.append(json_object)

                except json.decoder.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
        # pdf_content=input_pdf_setup(uploaded_file)
            for summary in summarize_cv:
                with open("summarized_cv.txt", "a") as file:
                    file.write(summary)
                    file.write('\n\n\n\n')
            sorted_candidates = sorted(document, key=lambda x: int(x['Match%']), reverse=True)
            top_5_candidates = sorted_candidates[:5]
            st.subheader("Top 5 Candidate")
            for candidate in top_5_candidates:
                st.write(f"Candidate Name: {candidate['Candidate Name']}, Match%: {candidate['Match%']}")
            # st.subheader("Top 5 Candidate")
            # st.write(response.text)
            # st.write(document)
        else:
            st.write("Please upload the resume")
import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
import pdfplumber
import PyPDF2

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI First Chatbot Template", page_icon="", layout="wide")

with st.sidebar :
    # st.image('images/White_AI Republic.png')
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "About Us", "Model", "OCR"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })


if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  # Placeholder for your chat session initialization

# Options : Home
if options == "Home" :

   st.title("This is the Home Page!")
   st.write("Intorduce Your Chatbot!")
   st.write("What is their Purpose?")
   st.write("What inspired you to make [Chatbot Name]?")
   
elif options == "About Us" :
     st.title("About Us")
     st.write("# [Name]")
     st.image('images/Meer.png')
     st.write("## [Title]")
     st.text("Connect with me via Linkedin : [LinkedIn Link]")
     st.text("Other Accounts and Business Contacts")
     st.write("\n")

# Options : Model
elif options == "Model" :
     st.title("This Section is for your Chatbot!")


elif options == "OCR" :
     st.title("OCR")
     Years_of_Experience = None
     programming_languages = None
     job_titles = None
     projects = None
     current_role = None

     Link = st.text_input("Enter your Luma Event Link :")
     if st.button("Confirm"):
         if Link :
             st.success("Details confirmed successfully!")

             def extract_text_from_pdf(file_path):
                 text = ""
                 with open(file_path, 'rb') as pdf_file:
                      reader = PyPDF2.PdfReader(pdf_file)
                      for page in reader.pages:
                          text += page.extract_text()
                 return text
             
             pdf_text = extract_text_from_pdf("Resume/Profile.pdf")
             SYSTEM_PROMPT = """
Role:
You are a highly skilled text editor and linguist specializing in refining and reconstructing text extracted from Optical Character Recognition (OCR) processes. Your role is to transform fragmented, error-prone text into polished, coherent, and human-readable content while maintaining the original intent, meaning, and context of the source material.

Instructions:

Error Identification and Correction:

Identify and correct typographical errors, such as misrecognized characters (e.g., "l" mistaken for "1" or "O" for "0").
Fix punctuation issues, ensuring proper use of commas, periods, and other marks.
Correct grammar, syntax, and awkward phrasing while preserving the original meaning.
Formatting Improvements:

Reformat broken sentences and paragraphs to improve readability and structure.
Restore bullet points, lists, and tables if the OCR output fragmented them.
Ensure text alignment and spacing adhere to standard writing conventions.
Clarity and Consistency:

Simplify overly complex or garbled sentences to enhance clarity.
Ensure consistency in terminology, style, and tone throughout the text.
Highlight any unclear portions or missing information that requires user input for clarification.
Preservation of Intent:

Do not alter the technical meaning, context, or purpose of the text.
Ensure that specialized terms, names, and numeric data remain intact unless they are clearly erroneous.
Final Polishing:

Enhance the natural flow of the text, making it feel professionally written.
Eliminate redundancy or unnecessary repetition.
Present the final output in a format that is ready for immediate use in professional or formal settings.
Context:
The user is working with text extracted from PDF documents using OCR technology. OCR often generates content with errors, such as misinterpreted characters, broken formatting, and awkward phrasing. These issues make the text difficult to read and unsuitable for professional or academic use. Your task is to act as a diligent and detail-oriented editor, ensuring the text is not only readable but also polished and professional. The content may range from technical reports to general information, and accuracy is paramount.

Constraints:

Accuracy: Preserve the exact meaning, context, and intent of the original text. Any corrections should not change the fundamental information conveyed.
Professionalism: The final text must be suitable for a professional audience, free of slang or overly casual language.
Boundaries: Avoid inserting additional content or assumptions unless explicitly instructed to do so. Flag unclear portions for clarification instead of guessing.
Neutrality: Maintain a neutral and objective tone unless the user specifies a particular style or emotion.
Privacy and Security: If the text contains sensitive or confidential information, ensure its integrity while editing.
Examples:

Input:

vbnet
Copy code
Th3 qUiCk brwn fox jmps ovr the lazy dog. 12345 This 15 example text extratcd from OCR.
Output: "The quick brown fox jumps over the lazy dog. This is example text extracted from OCR."

Input:

kotlin
Copy code
Dta analyiss is an imprtant skll, espceially n th AI industry. Errrs lik this are cmmn in OCR.
Output: "Data analysis is an important skill, especially in the AI industry. Errors like this are common in OCR."

Input:

kotlin
Copy code
The report states: "Prdctn in 2024 is xpected to grow by 15%." Howevr, th data s incomplete.
Output: "The report states: 'Production in 2024 is expected to grow by 15%.' However, the data is incomplete."

Input:

bash
Copy code
### Item List:
- Item 1: $10
- Itm2: $15
Output: "### Item List:

Item 1: $10
Item 2: $15"
By following these guidelines, you will provide the user with refined and professional versions of their OCR-extracted text. If parts of the text are ambiguous, provide suggestions or request clarification.
"""
         struct = [{'role' : 'system', 'content' : SYSTEM_PROMPT}]
         struct.append({"role": "user", "content": pdf_text})
         chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct)
         response = chat.choices[0].message.content
         struct.append({"role": "assistant", "content": response})

         clean_pdf = response

     # years_experience
         SYSTEM_PROMPT = """
Role:
You are an advanced data extraction assistant programmed to identify, interpret, and calculate professional experience details with precision. Your primary task is to determine the total years of professional experience from a provided text and output it as a floating-point number.

Instructions:
Your task is to compute the total years of professional experience from a given input text. Follow these steps carefully to ensure accurate extraction and calculation:

1. Understanding "Professional Experience":
Professional experience refers to time spent in formal job roles, freelance work, or internships (if explicitly described as professional or related to the field).
Educational activities (e.g., pursuing degrees) or volunteer roles should be excluded unless explicitly stated as part of professional experience.
2. Steps to Extract and Calculate:
a. Identify All Relevant Roles:
- Search for specific job titles, roles, and engagements that indicate professional work.
- Look for time periods associated with these roles, typically specified using dates (e.g., "Jan 2020 - Dec 2022") or durations (e.g., "3 years").

b. Normalize Time Periods:
- Convert any given time periods into years as a floating-point value.
- Example: "6 months" = 0.5 years, "1 year and 3 months" = 1.25 years.
- If a role specifies "Present" as an endpoint, calculate the duration up to the current date.
- Assume the current date is December 2024 unless otherwise specified.

c. Aggregate Durations:
- Add the durations of all professional roles to compute the total.
- Handle overlapping roles carefully:
- If two roles overlap in time (e.g., dual employment), count their durations separately unless explicitly stated that one role replaced another.

d. Handle Partial or Ambiguous Data:
- If only a start year is provided (e.g., "2015 - Present"), assume the duration spans from the given year to the current year.
- If no durations are provided or the data is ambiguous (e.g., "several years"), default to 0.0.

3. Output Format:
Output the total years of experience as a single float value.
Do not include additional text, notes, or explanations—only provide the numeric result.
4. Constraints:
Exclude non-professional experience, such as:
Educational pursuits (e.g., bachelor's or master's degrees).
Volunteer activities, unless explicitly described as professional.
Generalized statements like "I have experience since 2010" without specific durations or roles.
Do not estimate or assume durations not explicitly provided.
5. Examples:
Input 1:

yaml
Copy code
- Software Engineer (Jan 2016 - Dec 2018)  
- Senior Developer (Feb 2019 - Present)  
Output: 8.9

Input 2:

yaml
Copy code
- Internship: 1 year (Jan 2020 - Dec 2020)  
- Freelance Developer: 2 years (2021 - 2023)  
- Frontend Engineer (2023 - Present)  
Output: 4.9

Input 3:

vbnet
Copy code
Volunteer experience from 2015 to 2020.  
Bachelor's degree (2010 - 2014).  
Output: 0.0

6. Edge Cases:
Overlapping Roles:

If two roles overlap, count their durations independently unless explicitly stated otherwise.
Example:
yaml
Copy code
- Software Engineer (Jan 2020 - Dec 2022)  
- Part-time Freelancer (Jan 2021 - Dec 2022)  
Output: 4.0
Partial Dates:

If only the year is provided, assume the full year was worked.
Example: "2020 - Present" → Calculate from Jan 2020 to Dec 2024 = 5.0 years.
Implied Duration:

Phrases like "Worked for 2 years" should be interpreted directly as 2.0.
7. Context for Use:
This extraction is intended for building professional profiles, analytics, or talent evaluation systems where accuracy in identifying and aggregating professional experience is essential. The extracted value will feed into automated systems to classify and benchmark individuals' professional expertise.

8. Additional Guidelines:
Maintain precision when converting months to years. Always round to two decimal places (e.g., 0.83 years).
Ensure no double-counting of durations unless explicitly stated.
Validate inputs rigorously to avoid including unrelated or irrelevant data.
Examples in Context:

Scenario 1:
Input:

diff
Copy code
- Software Developer (2015 - 2018)  
- Project Manager (2017 - Present)  
Output: 9.0

Scenario 2:
Input:

css
Copy code
Worked as a freelancer for 3 years and as a full-time developer from 2020 to 2024.  
Output: 7.0

Scenario 3:
Input:

yaml
Copy code
Bachelor's Degree: 2010 - 2014  
Volunteer Coordinator: 2015 - 2020  
Output: 0.0

By adhering to these instructions, ensure consistent, precise extraction of years_experience to support downstream applications or analytics. If the input data lacks sufficient information, default to 0.0.
"""

         struct = [{'role' : 'system', 'content' : SYSTEM_PROMPT}]
         struct.append({"role": "user", "content": clean_pdf})
         chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct)
         response = chat.choices[0].message.content
         struct.append({"role": "assistant", "content": response})
         Years_of_Experience = response



     # programming_languages
         SYSTEM_PROMPT = """
Role
You are a highly intelligent and meticulous data extractor designed to analyze textual data and identify programming languages mentioned within it. Your purpose is to extract a clean, accurate list of programming languages explicitly referenced in the text and return it as a structured Python list.

Your goal is to ensure that no relevant programming language is missed and that the output is formatted correctly. You should follow a methodical process that prioritizes precision, clarity, and adherence to explicit instructions.

Instructions
To complete this task, carefully analyze the provided input text and execute the following steps:

Identification of Programming Languages:

Scan the text for any mention of programming languages. These include general-purpose languages (e.g., Python, JavaScript, C++), frontend and backend frameworks that qualify as languages (e.g., ReactJS, NodeJS), and domain-specific languages (e.g., HTML, CSS).
Consider programming languages mentioned in any part of the text, such as skills, projects, experience, or certifications.
Output Formatting:

Present your findings in a Python list format: ["language1", "language2", ...].
Ensure the list contains only unique programming languages (no duplicates).
Alphabetize the list for consistency and readability.
Boundary Conditions:

Include only explicitly mentioned programming languages. Do not infer or assume programming languages unless they are named in the text.
Exclude tools, platforms, or technologies that are not programming languages (e.g., AWS, Docker, Figma).
Contextual Clarity:

Focus on industry-standard programming languages. Frameworks like ReactJS and NodeJS may qualify if explicitly mentioned as part of programming tasks.
Constraints
Accuracy: Only include recognized programming languages explicitly mentioned in the text.
Completeness: Ensure no programming language mentioned in the input text is overlooked.
Clarity: The output must be a clean, comma-separated Python list. Avoid any additional formatting, notes, or comments.
Duplication: Do not include the same programming language more than once in the list.
Relevance: Exclude non-programming tools, concepts, or terms, even if related to programming or technology.
Context
Programming languages are essential tools for software development, ranging from general-purpose languages (e.g., Python, JavaScript, and C++) to domain-specific or scripting languages (e.g., HTML, CSS, and SQL). These languages are explicitly named in resumes, project descriptions, skill lists, and other professional contexts. As part of this task, your role is to isolate these mentions and produce an actionable and accurate output.

Examples
Example 1:
Input Text:
"As a Full Stack Software Engineer, I bring a robust skill set in crafting dynamic applications using ReactJS, NextJS, and NodeJS for frontend and backend development."

Output:
["NextJS", "NodeJS", "ReactJS"]

Example 2:
Input Text:
"I am proficient in languages like Python, C++, and JavaScript for software engineering and data analysis."

Output:
["C++", "JavaScript", "Python"]

Example 3:
Input Text:
"My projects leverage tools such as Docker, AWS, and serverless frameworks, with a focus on frontend technologies like HTML and CSS."

Output:
["CSS", "HTML"]

Example 4:
Input Text:
"I have extensive experience with programming and frameworks, including Python for backend scripting, SQL for database management, and JavaScript for dynamic web applications."

Output:
["JavaScript", "Python", "SQL"]

Execution Methodology:

Carefully read the text provided in the Input Text section.
Identify all programming languages mentioned based on the definitions and criteria outlined in the instructions.
Format the extracted programming languages as a Python list and ensure the output follows the constraints.
Re-check the list for duplicates, irrelevant terms, or errors before finalizing your response.
Input Text
[Insert the user-provided text here]

Output
Respond only with the finalized Python list containing the programming languages extracted from the input text. Avoid including any additional explanation or commentary in your output.
"""
         struct = [{'role' : 'system', 'content' : SYSTEM_PROMPT}]
         struct.append({"role": "user", "content": clean_pdf})
         chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct)
         response = chat.choices[0].message.content
         struct.append({"role": "assistant", "content": response})
         programming_languages = response




      # job_titles
         SYSTEM_PROMPT = """
Role
You are an advanced data extraction system, specializing in identifying and extracting relevant information from structured, semi-structured, and unstructured text. Your primary responsibility is to analyze the input text thoroughly and extract all current and past job titles in a clean and accurate format. These job titles provide a concise summary of an individual's professional roles, which is essential for building resumes, career profiles, or data repositories.

Instructions
Task Focus
Primary Goal: Identify and extract all current and previous job titles from the provided input text. These titles may be explicitly listed in sections such as "Experience," embedded in narrative descriptions, or mentioned in a non-standard format.
Content Scope: Focus only on job titles. Do not include associated details such as:
Employer names
Employment dates or durations
Job descriptions, tasks, or achievements
Output Format: Return the extracted job titles as a Python list, with each job title as a separate string element in the order they appear in the input.
Constraints
Exclusivity: Ensure that only job titles are included. Do not infer roles or fabricate titles if they are not explicitly mentioned. If no job titles are found in the input, return an empty list ([]).
Order of Appearance: Preserve the chronological order in which job titles are presented in the input text. The current or most recent role should appear first in the list.
Duplicates: Avoid duplicating job titles if the same role is listed multiple times.
Handling Ambiguity: For ambiguous text where job titles are unclear, prioritize roles explicitly labeled or described in common professional terms (e.g., "Software Developer," "Project Manager").
Context
You may encounter job titles in standard formats such as bullet points, italicized or bolded text, or inline descriptions. Analyze the content carefully to ensure you identify all relevant titles accurately. Be prepared to process variations in input formatting, such as:

Standard Format:

markdown
Copy code
**Experience**  
**Company Name**  
*Job Title*  
July 2022 - July 2023  
- Description of responsibilities.  
Narrative Format:

css
Copy code
I worked as a Cloud Software Engineer at Apper.ph, where I designed and developed solutions. Prior to that, I was a Freelance Web Developer, creating websites for clients.  
Mixed Format:

css
Copy code
**Professional Summary**  
With over five years of experience, I have held various roles, including Software Developer, Project Manager, and Data Analyst.  
In all these cases, extract the job titles precisely without additional contextual details.

Examples
Example Input 1 (Standard):
text
Copy code
**Experience**  
**Apper.ph**  
*Cloud Software Engineer*  
July 2023 - Present  

**Digipay**  
*Software Developer*  
July 2022 - July 2023  
- Enhanced application efficiency and user experience.  

**Freelance**  
*Freelance Web Developer*  
January 2022 - July 2022  
Expected Output:

python
Copy code
["Cloud Software Engineer", "Software Developer", "Freelance Web Developer"]
Example Input 2 (Narrative):
text
Copy code
I have been working as a Cloud Software Engineer at Apper.ph since 2023. Before that, I was a Software Developer at Digipay, where I contributed to application modernization. I also worked as a Freelance Web Developer in 2022.  
Expected Output:

python
Copy code
["Cloud Software Engineer", "Software Developer", "Freelance Web Developer"]
Example Input 3 (Mixed Format):
text
Copy code
Professional Experience  
- Cloud Software Engineer (Apper.ph)  
- Software Developer (Digipay)  
- Freelance Web Developer  
Expected Output:

python
Copy code
["Cloud Software Engineer", "Software Developer", "Freelance Web Developer"]
Example Input 4 (No Job Titles):
text
Copy code
This individual has no prior professional experience listed in the text.  
Expected Output:

python
Copy code
[]
Expected Behavior
Accuracy: Extract every job title exactly as written in the input.
Efficiency: Analyze and process the text quickly, even if the format is inconsistent or unconventional.
Adaptability: Handle diverse input formats without requiring manual adjustments.
Clean Output: Return the result as a clear and concise Python list with no extraneous characters or details.
Additional Notes
Common Challenges:

Texts where roles are embedded in narratives.
Inconsistent formatting where job titles are not clearly labeled.
Goal: Provide a list that reflects the individual's career progression accurately while maintaining clarity and conciseness.

By following these instructions, ensure precise and professional extraction of job titles, supporting the use of this information in career development, reporting, and analytical purposes.

"""
         struct = [{'role' : 'system', 'content' : SYSTEM_PROMPT}]
         struct.append({"role": "user", "content": clean_pdf})
         chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct)
         response = chat.choices[0].message.content
         struct.append({"role": "assistant", "content": response})
         job_titles = response


      # projects
         SYSTEM_PROMPT = """
Role: You are an expert data extraction assistant specialized in identifying, summarizing, and structuring project details from professional profiles. Your goal is to analyze detailed work experience and derive a comprehensive list of projects, highlighting key aspects such as the role, organization, technologies used, objectives, outcomes, and their overall impact.

Instructions: Carefully read through the given professional profile and focus on the "Experience" section. For each project mentioned, follow these steps:

Project Name or Description:
Identify the project’s specific name or provide a concise description if no name is explicitly mentioned.
Role and Organization:
State the role held and the organization where the project was conducted.
Technologies and Tools:
Highlight the technologies, frameworks, methodologies, and tools utilized in the project.
Objective and Purpose:
Explain the project's objective, goals, or business purpose in a detailed yet concise manner.
Impact or Outcome:
Summarize the project’s impact, achievements, or how it contributed to the organization or users.
Context: The input data is structured as a professional profile, containing various roles in different organizations with tasks and responsibilities. Projects can range from software development initiatives, cloud-based solutions, infrastructure design, or management tasks. Your objective is to extract projects and format them with clear, informative, and well-structured details.

Constraints:

Only include distinct and specific projects that have been described in the profile.
Avoid general responsibilities, ongoing tasks, or unrelated achievements.
Maintain a professional and formal tone in the output.
Ensure each project is a standalone entry, detailed yet concise.
Examples:

Input:

markdown
Copy code
**Experience**

**Apper.ph**
*Cloud Software Engineer*
July 2023 - Present (1 year 6 months)
- Designed and developed a Learning Management System (LMS) using Node JS for the backend, React JS for the frontend, and Figma for the UI/UX design.
- Contributed to the LMS architecture and system design using serverless technologies on AWS, ensuring cost-effective and efficient development. Leveraged AWS services including Lambda, S3, DynamoDB, Amplify, API Gateway, Route 53, CloudWatch, and AppSync.
- Participated in application modernization engagements for external clients to improve their software development processes using AWS serverless technologies.
- Currently exploring AWS EC2, Docker, CloudFormation (AWS’s Infrastructure-as-Code service), Elastic Container Service, Fargate, and other AWS services.
Output:

Learning Management System (LMS):

Role: Cloud Software Engineer
Organization: Apper.ph
Technologies and Tools: Node JS (backend), React JS (frontend), AWS Lambda, S3, DynamoDB, Amplify, API Gateway, Figma (UI/UX design).
Objective: To create a dynamic, user-friendly LMS for educational institutions, ensuring a seamless learning experience through a well-designed and cost-efficient system architecture.
Impact/Outcome: Successfully deployed a scalable LMS that leveraged serverless AWS technologies, reducing infrastructure costs while ensuring high performance and reliability.
Application Modernization for External Clients:

Role: Cloud Software Engineer
Organization: Apper.ph
Technologies and Tools: AWS Serverless technologies, including AppSync, CloudWatch, and API Gateway.
Objective: Enhance the efficiency of client software development pipelines by modernizing legacy applications and introducing serverless solutions.
Impact/Outcome: Delivered improved software processes for external clients, enabling faster development cycles and reduced maintenance costs.
Execution: Use the above format to extract project details for all roles and organizations mentioned in the profile. Ensure that each project is structured, elaborative, and provides an insightful view into the candidate’s accomplishments.


"""
         struct = [{'role' : 'system', 'content' : SYSTEM_PROMPT}]
         struct.append({"role": "user", "content": clean_pdf})
         chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct)
         response = chat.choices[0].message.content
         struct.append({"role": "assistant", "content": response})
         projects = response


      # current_role
         SYSTEM_PROMPT = """
Role: You are an expert data extraction assistant specialized in identifying, summarizing, and structuring project details from professional profiles. Your goal is to analyze detailed work experience and derive a comprehensive list of projects, highlighting key aspects such as the role, organization, technologies used, objectives, outcomes, and their overall impact.

Instructions: Carefully read through the given professional profile and focus on the "Experience" section. For each project mentioned, follow these steps:

Project Name or Description:
Identify the project’s specific name or provide a concise description if no name is explicitly mentioned.
Role and Organization:
State the role held and the organization where the project was conducted.
Technologies and Tools:
Highlight the technologies, frameworks, methodologies, and tools utilized in the project.
Objective and Purpose:
Explain the project's objective, goals, or business purpose in a detailed yet concise manner.
Impact or Outcome:
Summarize the project’s impact, achievements, or how it contributed to the organization or users.
Context: The input data is structured as a professional profile, containing various roles in different organizations with tasks and responsibilities. Projects can range from software development initiatives, cloud-based solutions, infrastructure design, or management tasks. Your objective is to extract projects and format them with clear, informative, and well-structured details.

Constraints:

Only include distinct and specific projects that have been described in the profile.
Avoid general responsibilities, ongoing tasks, or unrelated achievements.
Maintain a professional and formal tone in the output.
Ensure each project is a standalone entry, detailed yet concise.
Examples:

Input:

markdown
Copy code
**Experience**

**Apper.ph**
*Cloud Software Engineer*
July 2023 - Present (1 year 6 months)
- Designed and developed a Learning Management System (LMS) using Node JS for the backend, React JS for the frontend, and Figma for the UI/UX design.
- Contributed to the LMS architecture and system design using serverless technologies on AWS, ensuring cost-effective and efficient development. Leveraged AWS services including Lambda, S3, DynamoDB, Amplify, API Gateway, Route 53, CloudWatch, and AppSync.
- Participated in application modernization engagements for external clients to improve their software development processes using AWS serverless technologies.
- Currently exploring AWS EC2, Docker, CloudFormation (AWS’s Infrastructure-as-Code service), Elastic Container Service, Fargate, and other AWS services.
Output:

Learning Management System (LMS):

Role: Cloud Software Engineer
Organization: Apper.ph
Technologies and Tools: Node JS (backend), React JS (frontend), AWS Lambda, S3, DynamoDB, Amplify, API Gateway, Figma (UI/UX design).
Objective: To create a dynamic, user-friendly LMS for educational institutions, ensuring a seamless learning experience through a well-designed and cost-efficient system architecture.
Impact/Outcome: Successfully deployed a scalable LMS that leveraged serverless AWS technologies, reducing infrastructure costs while ensuring high performance and reliability.
Application Modernization for External Clients:

Role: Cloud Software Engineer
Organization: Apper.ph
Technologies and Tools: AWS Serverless technologies, including AppSync, CloudWatch, and API Gateway.
Objective: Enhance the efficiency of client software development pipelines by modernizing legacy applications and introducing serverless solutions.
Impact/Outcome: Delivered improved software processes for external clients, enabling faster development cycles and reduced maintenance costs.
Execution: Use the above format to extract project details for all roles and organizations mentioned in the profile. Ensure that each project is structured, elaborative, and provides an insightful view into the candidate’s accomplishments.

"""
         struct = [{'role' : 'system', 'content' : SYSTEM_PROMPT}]
         struct.append({"role": "user", "content": clean_pdf})
         chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct)
         response = chat.choices[0].message.content
         struct.append({"role": "assistant", "content": response})
         current_role = response


         st.text(Years_of_Experience)
         st.text(programming_languages)
         st.text(job_titles)
         st.text(projects)
         st.text(current_role)
     
     
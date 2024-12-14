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

     st.text(response)

    

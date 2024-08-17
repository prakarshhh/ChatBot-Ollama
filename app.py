import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv
import requests
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Enhanced Q&A Chatbot With Ollama"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to user queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, llm, temperature, max_tokens):
    try:
        llm = Ollama(model=llm)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({'question': question})
        return answer
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Title of the app
st.set_page_config(page_title="Ollama Chatbot", layout="wide")

st.title("Ollama Chatbot with Mistral Library")

# Sidebar
st.sidebar.header("Model and Settings")
llm = st.sidebar.selectbox("Select Open Source Model", ["mistral"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main Interface
st.write("Ask me anything and get responses from the Ollama model.")

user_input = st.text_input("You:", "")

if user_input:
    response = generate_response(user_input, llm, temperature, max_tokens)
    st.write(f"<div class='response'>{response}</div>", unsafe_allow_html=True)
else:
    st.write("Please provide your question in the input box.")

# Custom CSS for improved UI and animations
st.markdown("""
    <style>
    .css-1l02p8g {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .css-1f4i63v {
        background-color: #e0e0e0;
        color: #333;
    }
    .css-1e9vhu4 {
        border: 2px solid #007bff;
        border-radius: 5px;
        padding: 5px 10px;
        background-color: #007bff;
        color: white;
    }
    .css-1e9vhu4:hover {
        background-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)

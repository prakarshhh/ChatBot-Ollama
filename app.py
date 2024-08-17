import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

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
    llm = Ollama(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

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
    /* Main content styling */
    .streamlit-expanderHeader {
        font-size: 24px;
        font-weight: bold;
    }

    /* Animation for responses */
    .response {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #b3e0f3;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        animation: fadeIn 1s ease-out;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Hover effect for text input */
    .css-1uv7qhn {
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .css-1uv7qhn:hover {
        border-color: #007bff;
        box-shadow: 0 0 5px rgba(0, 123, 255, 0.5);
    }

    /* Button styles */
    .css-1e9vhu4 {
        border: 2px solid #007bff;
        border-radius: 5px;
        padding: 5px 10px;
        background-color: #007bff;
        color: white;
        transition: background-color 0.3s ease, transform 0.3s ease;
    }
    .css-1e9vhu4:hover {
        background-color: #0056b3;
        transform: scale(1.05);
    }
    .css-1e9vhu4:active {
        transform: scale(0.95);
    }

    /* Main layout styling */
    .main-content {
        padding: 20px;
        background-color: #f9f9f9;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .main-content:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }

    </style>
""", unsafe_allow_html=True)

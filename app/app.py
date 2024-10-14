import streamlit as st
import random
import time
import chromadb
from openai import OpenAI
import os
from utils import rag_pipeline
# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()


st.title("Simple chat")

# Initialize messages list if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful assistant. Use the following context to answer the user's question: {context}\n\n --------"

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    st.session_state.collection = chroma_client.get_collection("pdf_collection")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in rag_pipeline(prompt, 
                                     st.session_state.collection, 
                                     st.session_state.system_prompt, 
                                     st.session_state.messages):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

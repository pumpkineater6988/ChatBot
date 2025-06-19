# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 14:57:33 2025

@author: sahil
"""

import streamlit as st
import tempfile
import chardet
import pandas as pd
import os
import json
import re
import random
from openai import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from io import StringIO
from dotenv import load_dotenv

# ---- Load Environment Variables ----
load_dotenv()

# ---- OpenAI Client ----
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---- Streamlit UI ----
st.set_page_config(page_title="GenAI Financial Chatbot", layout="centered")
st.title("📊 GenAI Financial Question Generator")

# ---- Session State ----
for key in ["retriever", "dataframe", "generated_questions"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "generated_questions" else []

# ---- File Upload ----
uploaded_file = st.file_uploader("Upload PDF, CSV, or Excel", type=["pdf", "csv", "xlsx"])
if uploaded_file and st.button("📥 Upload & Generate Questions"):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        docs = []
        df = None

        if uploaded_file.name.endswith(".csv"):
            with open(tmp_path, 'rb') as f:
                encoding = chardet.detect(f.read())['encoding']
            df = pd.read_csv(tmp_path, encoding=encoding)
            docs = [Document(page_content=row.to_json() if row.to_json() is not None else "") for _, row in df.iterrows()]

        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(tmp_path)
            docs = [Document(page_content=row.to_json() if row.to_json() is not None else "") for _, row in df.iterrows()]

        elif uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

        st.session_state.dataframe = df

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        st.success("✅ File processed successfully!")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# ---- Question Generation ----
if st.session_state.retriever:
    num_questions = st.number_input("How many questions do you want to generate?", min_value=1, max_value=20, value=5)

    if st.button("🔍 Generate Questions"):
        try:
            sample_context = ""
            documents = st.session_state.retriever.get_relevant_documents("Generate insightful financial questions")
            for doc in documents:
                sample_context += doc.page_content + "\n"

            # Use OpenAI's message format
            from openai.types.chat import ChatCompletionMessageParam
            messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": "You are a financial data expert. Based on the provided context, generate insightful and relevant questions a user might ask."},
                {"role": "user", "content": f"Context:\n{sample_context}\n\nGenerate {num_questions} questions."}
            ]

            result = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )

            # Defensive: handle if result.choices[0].message.content is None
            content = getattr(result.choices[0].message, "content", None)
            if content is None:
                st.error("❌ No content returned from OpenAI.")
                st.session_state.generated_questions = []
            else:
                questions = content.strip()
                st.session_state.generated_questions = questions.split("\n") if "\n" in questions else [questions]
                st.success("✅ Questions generated successfully!")

        except Exception as e:
            st.error(f"❌ Error from OpenAI: {e}")

# ---- Display Generated Questions ----
if st.session_state.generated_questions:
    st.markdown("### 📋 Generated Questions")
    for i, q in enumerate(st.session_state.generated_questions, start=1):
        st.markdown(f"**Q{i}:** {q}") 
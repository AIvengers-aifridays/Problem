import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import httpx
import os
import requests

# ---- Disable SSL verification for internal endpoints ----
requests.packages.urllib3.disable_warnings()
session = requests.Session()
session.verify = False
requests.get = session.get

# Cache dir for tokens
tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
client = httpx.Client(verify=False)

# ---- LLM Setup ----
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-TEv_e2ELECztvdxNnQwz4A",   # replace with valid key
    http_client=client
)

embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key="sk-TEv_e2ELECztvdxNnQwz4A",   # replace with valid key
    http_client=client
)

# ---- Streamlit UI ----
st.set_page_config(page_title="Application Maintenance Issue Resolver")
st.title("🛠️ AI-Powered Application Maintenance Issue Resolver")

# ---- Step 1: Load Prebuilt VectorDB ----
try:
    vectordb_path = "data/vectordb.index"
    metadata_path = "data/vectordb_metadata.csv"

    # Load FAISS index with embeddings
    vectorstore = FAISS.load_local(
        vectordb_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True  # required for FAISS persistence
    )
    retriever = vectorstore.as_retriever()

    st.sidebar.success("✅ Loaded prebuilt VectorDB (faster queries).")
except Exception as e:
    st.error(f"❌ Failed to load VectorDB: {e}")
    st.stop()

# ---- Step 2: RetrievalQA ----
system_prompt = """
You are an AI Maintenance Assistant. 
You can ONLY help with application maintenance issues using the provided logs and troubleshooting manuals. 
If the user asks anything unrelated (like jokes, personal questions, general knowledge, etc.), 
reply with: "⚠️ I am designed only to help with application maintenance issues. Please describe your issue."
"""

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": None},
)

# ---- Step 3: Query UI ----
query = st.text_input("⚡ Describe the maintenance issue you are facing:")

if query:
    if any(word in query.lower() for word in ["joke", "weather", "who", "what", "when", "where"]):
        st.subheader("⚠️ Notice:")
        st.write("I am designed only to help with application maintenance issues. Please describe your issue.")
    else:
        answer = qa_chain.run(query)
        st.subheader("✅ Suggested Resolution Steps:")
        st.write(answer)

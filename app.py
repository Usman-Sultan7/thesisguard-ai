import os
import tempfile
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="ThesisGuard AI",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for Professional Centered Styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #4B5563;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. INITIALIZE SESSION STATE
# ==========================================
if "api_authenticated" not in st.session_state:
    st.session_state["api_authenticated"] = False
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "file_sig" not in st.session_state:
    st.session_state["file_sig"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ==========================================
# 3. SECURE ACCESS PORTAL (LOCK SCREEN)
# ==========================================
if not st.session_state["api_authenticated"]:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>🎓 ThesisGuard AI</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Secure Academic Peer Review Portal</p>", unsafe_allow_html=True)
        
        api_key_input = st.text_input("Enter OpenAI API Key", type="password")

        if st.button("Unlock Dashboard", use_container_width=True):
            if api_key_input.strip() != "":
                try:
                    # Test API key instantly
                    test_embeddings = OpenAIEmbeddings(openai_api_key=api_key_input.strip())
                    st.session_state["api_key"] = api_key_input.strip()
                    st.session_state["api_authenticated"] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Authentication failed! Please check your API key.")
            else:
                st.error("API key cannot be empty.")
    st.stop()

# ==========================================
# 4. SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.title("⚙️ ThesisGuard Control")
    st.markdown("---")
    
    st.subheader("📁 Document Selection")
    # Clean Radio Selector for PDF or Excel
    doc_type = st.radio("Choose Document Type:", ["PDF Research Paper", "Excel Metrics Sheet"])
    
    uploaded_pdf = None
    uploaded_excel = None
    
    if doc_type == "PDF Research Paper":
        uploaded_pdf = st.file_uploader("Upload Thesis (PDF)", type=["pdf"])
    else:
        uploaded_excel = st.file_uploader("Upload Data Sheet (Excel)", type=["xlsx", "xls"])
    
    st.markdown("---")
    st.subheader("🛠️ Chunking Parameters")
    chunk_size = st.slider("Chunk Size", min_value=200, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=300, value=200, step=25)
    
    st.markdown("---")
    st.subheader("🔍 Retrieval Settings")
    top_k = st.slider("Retrieved Chunks (top_k)", min_value=1, max_value=10, value=4, step=1)
    academic_rigor = st.selectbox("Academic Rigor Level", ["Master's Defense", "PhD Dissertation", "Undergraduate Review"])
    model_name = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o"])

    st.markdown("---")
    if st.button("🔒 Lock / Clear Session", use_container_width=True):
        st.session_state["api_authenticated"] = False
        st.session_state["vectorstore"] = None
        st.session_state["messages"] = []
        st.rerun()

# ==========================================
# 5. DOCUMENT PROCESSING & FAISS INDEXING
# ==========================================
@st.cache_resource
def build_vectorstore(pdf_file, excel_file, c_size, c_overlap, api_key):
    docs = []
    try:
        # Process PDF
        if pdf_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(pdf_file.getvalue())
                tmp_pdf_path = tmp_pdf.name
            
            loader = PyPDFLoader(tmp_pdf_path)
            docs.extend(loader.load())
            os.unlink(tmp_pdf_path)

        # Process Excel
        if excel_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_excel:
                tmp_excel.write(excel_file.getvalue())
                tmp_excel_path = tmp_excel.name
            
            excel_dfs = pd.read_excel(tmp_excel_path, sheet_name=None)
            from langchain_core.documents import Document
            for sheet_name, df in excel_dfs.items():
                csv_data = df.to_string(index=False)
                doc_text = f"Sheet Name: {sheet_name}\n\nDataset Summary:\n{csv_data}"
                docs.append(Document(page_content=doc_text, metadata={"source": excel_file.name, "page": sheet_name}))
            os.unlink(tmp_excel_path)

        if not docs:
            return None

        # Chunking & Embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
        split_docs = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error processing documents: {e}")
        return None

# Trigger Vector Store Build when files are uploaded
if uploaded_pdf is not None or uploaded_excel is not None:
    active_file = uploaded_pdf if uploaded_pdf else uploaded_excel
    file_sig = (active_file.name, active_file.size, chunk_size, chunk_overlap)

    if st.session_state.get("file_sig") != file_sig:
        with st.spinner("🔄 Indexing document chunks into FAISS vector space..."):
            vstore = build_vectorstore(
                uploaded_pdf, uploaded_excel, chunk_size, chunk_overlap, st.session_state["api_key"]
            )
            if vstore:
                st.session_state["vectorstore"] = vstore
                st.session_state["file_sig"] = file_sig
                st.session_state["messages"] = []
                st.success(f"✅ Successfully indexed **{active_file.name}**!")
            else:
                st.error("Failed to build vector store. Please check your uploaded file.")

# ==========================================
# 6. RAG CHAIN BUILDER
# ==========================================
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
     f"You are a strict and experienced academic peer reviewer and thesis committee head operating at a {academic_rigor} standard.\n"
     "Analyze the uploaded research documents and answer questions rigorously.\n"
     "Rules:\n"
     "1. Answer strictly using the provided context from the research documents.\n"
     "2. If information is missing, state: \"I couldn't find that in the provided thesis documents.\"\n"
     "3. Cite the exact page number or sheet source used, e.g., (p. 3).\n\n"
     "Context:\n{context}"
    ),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join([f"[Source: {d.metadata.get('source', 'Unknown')} | Page/Sheet: {d.metadata.get('page', 'N/A')}]\n{d.page_content}" for d in docs])

def get_chain(vectorstore, model, k):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model_name=model, temperature=0.1, openai_api_key=st.session_state["api_key"])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever

# ==========================================
# 7. MAIN CENTERED CHAT INTERFACE
# ==========================================
st.markdown("<h1 class='main-title'>🎓 ThesisGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Advanced Research Peer Review & Defense Preparation Suite</p>", unsafe_allow_html=True)

if st.session_state["vectorstore"] is not None:
    # Replay Message History
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Bottom-Center Chat Input
    if question := st.chat_input("Ask a question about your research paper or defense preparation..."):
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        try:
            chain, retriever = get_chain(st.session_state["vectorstore"], model_name, top_k)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing research context..."):
                    answer = chain.invoke(question)
                    st.markdown(answer)

                # Transparency: Retrieved Chunks Expander
                with st.expander("🔍 Sources (Retrieved Chunks Transparency)"):
                    retrieved_docs = retriever.invoke(question)
                    for i, doc in enumerate(retrieved_docs):
                        page_num = doc.metadata.get('page', 'N/A')
                        display_page = page_num + 1 if isinstance(page_num, int) else page_num
                        st.markdown(f"**Chunk {i+1} | Source: {doc.metadata.get('source', 'Unknown')} (Page/Sheet: {display_page})**")
                        st.text(doc.page_content[:500])
                        st.divider()

            st.session_state["messages"].append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"An error occurred during generation: {e}")
else:
    st.info("👆 Please select your file type (PDF or Excel) from the sidebar and upload your document to initialize ThesisGuard AI.")
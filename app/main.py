import streamlit as st
import os
from dotenv import load_dotenv
import io

from file_loader import get_raw_text
from qa_engine import build_qa_engine

# Load env
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_uploaded_file_bytes" not in st.session_state:
    st.session_state.last_uploaded_file_bytes = None
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "qa" not in st.session_state:
    st.session_state.qa = None

# Streamlit Page Config
st.set_page_config(page_title="Doc Chatbot", layout="wide")
st.title("📄 Chat with your Document")

# Two column layout
left, right = st.columns([1, 2])

# ---------------- LEFT: File Upload ----------------
with left:
    st.header("📂 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF, DOCX, Excel, or ZIP file",
        type=["pdf", "docx", "xls", "xlsx", "zip"],
        accept_multiple_files=False
    )

    if uploaded_file:
        uploaded_bytes = uploaded_file.read()
        uploaded_file_for_processing = io.BytesIO(uploaded_bytes)

        # Reset chat if file changed
        if st.session_state.last_uploaded_file_bytes != uploaded_bytes:
            st.session_state.last_uploaded_file_bytes = uploaded_bytes
            st.session_state.chat_history = []

            # Extract text
            st.session_state.raw_text = get_raw_text(uploaded_file_for_processing.getvalue(), uploaded_file.name)

            # Build QA engine
            st.session_state.qa = build_qa_engine(st.session_state.raw_text, openai_api_key)

        st.success(f"✅ {uploaded_file.name} uploaded successfully")

        with st.expander("Preview Extracted Text"):
            st.text_area("Extracted Content", st.session_state.raw_text[:3000], height=400)

# ---------------- RIGHT: Chat ----------------
with right:
    st.header("💬 Chat with Document")

    # Inject CSS for fixed input + chat bubbles
    st.markdown(
        """
        <style>
        /* Fix chat input at bottom */
        .stChatInputContainer {
            position: fixed;
            bottom: 0;
            width: 60%; /* adjust based on layout */
            background-color: white;
            padding: 10px;
            border-top: 1px solid #ddd;
            z-index: 100;
        }

        /* Chat bubble container */
        .chat-message {
            max-width: 75%;
            padding: 10px 15px;
            margin: 8px 0;
            border-radius: 15px;
            line-height: 1.4;
            font-size: 15px;
            word-wrap: break-word;
        }

        /* User bubble (right side) */
        .user-message {
            background-color: #DCF8C6;
            margin-left: auto;
            text-align: right;
        }

        /* Bot bubble (left side) */
        .bot-message {
            background-color: #F1F0F0;
            margin-right: auto;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    chat_container = st.container()

    # Chat input (fixed bottom)
    query = st.chat_input("Ask something about the document...")

    if query and st.session_state.qa:
        result = st.session_state.qa({"query": query})
        st.session_state.chat_history.append({
            "question": query,
            "answer": result["result"],
            "context": result["source_documents"]
        })

    # Show chat history (oldest → newest) with bubbles
    with chat_container:
        for chat in st.session_state.chat_history:
            st.markdown(f'<div class="chat-message user-message"><b>You:</b> {chat["question"]}</div>', unsafe_allow_html=True)
            formatted_answer = chat["answer"].replace("\n", "<br>")
            st.markdown(f'<div class="chat-message bot-message"><b>Answer:</b><br>{formatted_answer}</div>', unsafe_allow_html=True)

            # st.markdown(f'<div class="chat-message bot-message"><b>Answer:</b> {chat["answer"]}</div>', unsafe_allow_html=True)

            with st.expander("🔍 Relevant Context"):
                for doc in chat["context"]:
                    st.write(doc.page_content[:300] + "...")

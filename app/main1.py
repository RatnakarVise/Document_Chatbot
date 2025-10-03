import streamlit as st
import os
from dotenv import load_dotenv
import io
from urllib.parse import urlparse, unquote

from file_loader import get_raw_text
from qa_engine import build_qa_engine

# Optional: SharePoint SDK
from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.user_credential import UserCredential

# ---------------- Load Environment ----------------
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)
openai_api_key = os.getenv("OPENAI_API_KEY")

# ---------------- Session State ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_uploaded_file_bytes" not in st.session_state:
    st.session_state.last_uploaded_file_bytes = None
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "qa" not in st.session_state:
    st.session_state.qa = None

# ---------------- Streamlit Page ----------------
st.set_page_config(page_title="Doc Chatbot", layout="wide")
st.title("📄 Chat with your Document")

# Two-column layout
left, right = st.columns([1, 2])

# ---------------- LEFT: Upload / SharePoint ----------------
with left:
    st.header("📂 Upload or Load Document")

    option = st.radio("Choose Input Method:", ["Upload File", "SharePoint Link"])

    uploaded_bytes = None
    filename = ""

    if option == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload PDF, DOCX, Excel, or ZIP",
            type=["pdf", "docx", "xls", "xlsx", "zip"]
        )
        if uploaded_file:
            uploaded_bytes = uploaded_file.read()
            filename = uploaded_file.name

    elif option == "SharePoint Link":
        sharepoint_url = st.text_input("Enter SharePoint file URL")
        username = st.text_input("SharePoint Username")
        password = st.text_input("SharePoint Password", type="password")
        if st.button("Load from SharePoint"):
            if sharepoint_url and username and password:
                try:
                    parsed = urlparse(sharepoint_url)
                    site_url = f"{parsed.scheme}://{parsed.netloc}"
                    relative_url = unquote(parsed.path)

                    ctx = ClientContext(site_url).with_credentials(UserCredential(username, password))
                    uploaded_bytes = io.BytesIO()
                    file = ctx.web.get_file_by_server_relative_url(relative_url)
                    file.download(uploaded_bytes).execute_query()
                    uploaded_bytes.seek(0)
                    filename = os.path.basename(relative_url)
                    st.success(f"✅ {filename} loaded from SharePoint")

                except Exception as e:
                    st.error(f"Error loading SharePoint file: {e}")

    # ---------------- Process File ----------------
    if uploaded_bytes and filename:
        # Reset chat if file changed
        if st.session_state.last_uploaded_file_bytes != uploaded_bytes:
            st.session_state.last_uploaded_file_bytes = uploaded_bytes
            st.session_state.chat_history = []

            # Extract text
            st.session_state.raw_text = get_raw_text(uploaded_bytes if isinstance(uploaded_bytes, bytes) else uploaded_bytes.getvalue(), filename)

            # Build QA engine
            st.session_state.qa = build_qa_engine(st.session_state.raw_text, openai_api_key)

        with st.expander("Preview Extracted Text"):
            st.text_area("Extracted Content", st.session_state.raw_text[:5000], height=400)

# ---------------- RIGHT: Chat ----------------
with right:
    st.header("💬 Chat with Document")

    chat_container = st.container()

    query = st.chat_input("Ask something about the document...")

    if query and st.session_state.qa:
        result = st.session_state.qa({"query": query})
        st.session_state.chat_history.append({
            "question": query,
            "answer": result["result"],
            "context": result["source_documents"]
        })

    # Show chat history (oldest → newest)
    with chat_container:
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['question']}")
            st.markdown(f"**Bot:** {chat['answer']}")
            with st.expander("🔍 Relevant Context"):
                for doc in chat["context"]:
                    st.write(doc.page_content[:300] + "...")

import streamlit as st
import os
import io
import requests
import base64
import time
from urllib.parse import urlparse
from dotenv import load_dotenv

# local modules
from file_loader import get_raw_text
from qa_engine import build_qa_engine, save_vectorstore, load_vectorstore

# ---------------- Load Environment ----------------
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

# ---------------- Persistence ----------------
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "persisted_data")
os.makedirs(PERSIST_DIR, exist_ok=True)

# ---------------- Streamlit UI Init ----------------
st.set_page_config(page_title="Doc Chatbot", layout="wide")

# ---------------- Session State Defaults ----------------
for key, default in {
    "page_initialized": False,
    "chat_history": [],
    "raw_text": "",
    "qa": None,
    "current_cache_name": None,
    "page": "upload"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if not st.session_state.page_initialized:
    with st.spinner("🔧 Initializing app..."):
        time.sleep(1)
    st.session_state.page_initialized = True

# ---------------- AUTOLOAD (optional) ----------------
AUTOLOAD = False
if AUTOLOAD and OPENAI_API_KEY:
    try:
        vectorstore = load_vectorstore(OPENAI_API_KEY, PERSIST_DIR)
        if vectorstore:
            st.session_state.qa, _ = build_qa_engine("", OPENAI_API_KEY, load_vectorstore_obj=vectorstore)
            st.success("✅ Auto-loaded knowledge base from memory")
    except Exception:
        st.info("ℹ️ No auto-loadable memory found.")

# =====================================================================
#                            PAGE: Upload / Load
# =====================================================================
if st.session_state.page == "upload":
    st.title("📂 Upload or Load Document")
    left_col, right_col = st.columns([1, 2])

    with right_col:
        option = st.radio("Choose Input Method:", ["Upload File", "SharePoint Link"])
        uploaded_bytes = None
        filename = None

        # -------- FILE UPLOAD --------
        if option == "Upload File":
            uploaded_file = st.file_uploader("Upload PDF, DOCX, Excel, or ZIP", type=["pdf", "docx", "xls", "xlsx", "zip"])
            if uploaded_file:
                uploaded_bytes = uploaded_file.read()
                filename = uploaded_file.name

            cache_name = st.text_input("Cache name for this upload (unique)", value="")
            if st.button("Process & Save to Memory") and uploaded_bytes and filename and cache_name:
                with st.spinner("⏳ Extracting text and building QA engine..."):
                    try:
                        raw_text = get_raw_text(uploaded_bytes, filename)
                        if not raw_text.strip():
                            st.error("❌ No text extracted from the file.")
                        else:
                            qa, vectorstore = build_qa_engine(raw_text, OPENAI_API_KEY, cache_name=cache_name)
                            save_vectorstore(vectorstore, PERSIST_DIR, cache_name=cache_name)
                            st.session_state.raw_text = raw_text
                            st.session_state.qa = qa
                            st.session_state.current_cache_name = cache_name
                            st.success(f"✅ Saved knowledge base as '{cache_name}'")
                    except Exception as e:
                        st.error(f"❌ Failed to process file: {e}")

        # -------- SHAREPOINT LINK --------
        elif option == "SharePoint Link":
            st.markdown("Load documents from SharePoint (single file or folder).")
            sharepoint_url = st.text_input("Enter SharePoint File/Folder URL or Sharing Link")
            cache_name = st.text_input("Cache name for this SharePoint folder (unique)", value="")

            def load_sharepoint_folder(site_id, folder_path, access_token):
                all_files = []
                folder_api = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root:/{folder_path}:/children"
                res = requests.get(folder_api, headers={"Authorization": f"Bearer {access_token}"})
                res.raise_for_status()
                items = res.json().get("value", [])
                for item in items:
                    if item.get("file"):
                        download_url = item.get("@microsoft.graph.downloadUrl")
                        if download_url:
                            file_res = requests.get(download_url)
                            file_res.raise_for_status()
                            all_files.append({"name": item.get("name"), "bytes": io.BytesIO(file_res.content)})
                    elif item.get("folder"):
                        subfolder_path = f"{folder_path}/{item['name']}"
                        all_files.extend(load_sharepoint_folder(site_id, subfolder_path, access_token))
                return all_files

            if st.button("Load from SharePoint (Graph API)"):
                if not sharepoint_url:
                    st.warning("Please paste a SharePoint URL first.")
                elif not all([TENANT_ID, CLIENT_ID, CLIENT_SECRET]):
                    st.error("SharePoint credentials not set in .env.")
                elif not cache_name:
                    st.warning("Please provide a unique cache name.")
                else:
                    with st.spinner("🔄 Loading files from SharePoint..."):
                        try:
                            token_url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
                            token_data = {
                                "grant_type": "client_credentials",
                                "client_id": CLIENT_ID,
                                "client_secret": CLIENT_SECRET,
                                "scope": "https://graph.microsoft.com/.default"
                            }
                            token_response = requests.post(token_url, data=token_data)
                            token_response.raise_for_status()
                            access_token = token_response.json().get("access_token")

                            encoded_url = base64.urlsafe_b64encode(sharepoint_url.strip().encode("utf-8")).decode("utf-8").rstrip("=")
                            meta_url = f"https://graph.microsoft.com/v1.0/shares/u!{encoded_url}/driveItem"
                            meta_res = requests.get(meta_url, headers={"Authorization": f"Bearer {access_token}"})
                            meta_res.raise_for_status()
                            meta_json = meta_res.json()

                            if meta_json.get("folder"):
                                children_url = f"{meta_url}/children"
                                children_res = requests.get(children_url, headers={"Authorization": f"Bearer {access_token}"})
                                children_res.raise_for_status()
                                items = children_res.json().get("value", [])
                                all_text = ""
                                for item in items:
                                    if item.get("file"):
                                        download_url = item.get("@microsoft.graph.downloadUrl")
                                        if download_url:
                                            file_res = requests.get(download_url)
                                            file_res.raise_for_status()
                                            all_text += get_raw_text(file_res.content, item.get("name")) + "\n\n"
                                qa, vectorstore = build_qa_engine(all_text, OPENAI_API_KEY, cache_name=cache_name)
                                save_vectorstore(vectorstore, PERSIST_DIR, cache_name=cache_name)
                                st.session_state.qa = qa
                                st.session_state.current_cache_name = cache_name
                                st.success(f"✅ Loaded and saved folder as '{cache_name}'")
                            else:
                                filename = meta_json.get("name", "sharepoint_file")
                                download_url = meta_json.get("@microsoft.graph.downloadUrl")
                                if download_url:
                                    file_res = requests.get(download_url)
                                    file_res.raise_for_status()
                                    raw_text = get_raw_text(file_res.content, filename)
                                    qa, vectorstore = build_qa_engine(raw_text, OPENAI_API_KEY, cache_name=cache_name)
                                    save_vectorstore(vectorstore, PERSIST_DIR, cache_name=cache_name)
                                    st.session_state.qa = qa
                                    st.session_state.current_cache_name = cache_name
                                    st.success(f"✅ Loaded and saved file as '{cache_name}'")
                        except Exception as e:
                            st.error(f"Error loading SharePoint: {e}")

        # -------- MEMORY SECTION --------
        st.markdown("---")
        st.markdown("### 🧠 Persistent Memory")
        caches = [d for d in os.listdir(PERSIST_DIR) if os.path.isdir(os.path.join(PERSIST_DIR, d))]
        st.write("Available caches:", caches if caches else "No saved caches yet.")
        selected_cache = st.selectbox("Select cache to load", options=["-- select --"] + caches)

        if st.button("Load selected cache"):
            if selected_cache and selected_cache != "-- select --":
                try:
                    vectorstore = load_vectorstore(OPENAI_API_KEY, PERSIST_DIR, cache_name=selected_cache)
                    if vectorstore:
                        st.session_state.qa, _ = build_qa_engine("", OPENAI_API_KEY, load_vectorstore_obj=vectorstore)
                        st.session_state.current_cache_name = selected_cache
                        st.success(f"✅ Loaded '{selected_cache}' into memory.")
                except Exception as e:
                    st.error(f"Failed to load cache: {e}")

        if st.button("Clear memory selection"):
            st.session_state.qa = None
            st.session_state.current_cache_name = None
            st.success("Cleared in-memory QA engine.")

        if st.session_state.qa:
            st.markdown("---")
            st.success("✅ Knowledge base ready.")
            if st.button("➡️ Go to Chat"):
                st.session_state.page = "chat"

# =====================================================================
#                            PAGE: Chat
# =====================================================================
elif st.session_state.page == "chat":
    st.header("💬 Chat with Document")
    if not st.session_state.qa:
        st.warning("⚠️ No knowledge base loaded. Go back to upload/load first.")
    else:
        if st.button("⬅️ Back to Upload"):
            st.session_state.page = "upload"
            st.stop()

        chat_container = st.container()
        query = st.chat_input("Ask something about the document...")

        if query:
            with st.spinner("🤖 Thinking..."):
                try:
                    result = st.session_state.qa({"query": query})
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer": result["result"],
                        "context": result.get("source_documents", [])
                    })
                except Exception as e:
                    st.error(f"Error while querying QA engine: {e}")

        with chat_container:
            for chat in st.session_state.chat_history:
                st.markdown(f"**You:** {chat['question']}")
                st.markdown(f"**Bot:** {chat['answer']}")
                if chat.get("context"):
                    with st.expander("🔍 Relevant Context"):
                        for doc in chat["context"]:
                            st.write(doc.page_content[:400] + "...")

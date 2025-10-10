import streamlit as st
import os
import io
import requests
from dotenv import load_dotenv
from urllib.parse import urlparse
import base64
from file_loader import get_raw_text
from qa_engine import build_qa_engine

# ---------------- Load Environment ----------------
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)
openai_api_key = os.getenv("OPENAI_API_KEY")
tenant_id = os.getenv("TENANT_ID")
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

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

    # --- Local Upload ---
    if option == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload PDF, DOCX, Excel, or ZIP",
            type=["pdf", "docx", "xls", "xlsx", "zip"]
        )
        if uploaded_file:
            uploaded_bytes = uploaded_file.read()
            filename = uploaded_file.name

    # --- SharePoint via Graph API ---
    elif option == "SharePoint Link":
        st.markdown("Load documents from SharePoint (single file or folder).")
        sharepoint_url = st.text_input("Enter SharePoint File/Folder URL or Sharing Link")

        def load_sharepoint_folder(site_id, folder_path, access_token):
            """
            Download all files in a SharePoint folder via Graph API
            """
            folder_api = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root:/{folder_path}:/children"
            res = requests.get(folder_api, headers={"Authorization": f"Bearer {access_token}"})
            res.raise_for_status()
            items = res.json().get("value", [])

            all_files = []
            for item in items:
                if item.get("file"):  # skip subfolders
                    download_url = item.get("@microsoft.graph.downloadUrl")
                    filename = item.get("name")
                    if download_url:
                        file_res = requests.get(download_url)
                        file_res.raise_for_status()
                        all_files.append({"name": filename, "bytes": io.BytesIO(file_res.content)})
            return all_files

        if st.button("Load from SharePoint (Graph API)"):
            if sharepoint_url:
                try:
                    # ---------------- Step 1 — Get Access Token ----------------
                    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
                    token_data = {
                        "grant_type": "client_credentials",
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "scope": "https://graph.microsoft.com/.default"
                    }
                    token_response = requests.post(token_url, data=token_data)
                    token_response.raise_for_status()
                    access_token = token_response.json().get("access_token")

                    # ---------------- Step 2 — Detect URL Type ----------------
                    if "/:b:/" in sharepoint_url or "/:f:/" in sharepoint_url or "/:w:/" in sharepoint_url:
                        # Single file sharing link
                        st.write("Detected: Sharing Link (single file)")
                        encoded_url = base64.urlsafe_b64encode(sharepoint_url.strip().encode("utf-8")).decode("utf-8").rstrip("=")

                        # Get metadata
                        meta_url = f"https://graph.microsoft.com/v1.0/shares/u!{encoded_url}/driveItem"
                        meta_res = requests.get(meta_url, headers={"Authorization": f"Bearer {access_token}"})
                        meta_res.raise_for_status()
                        meta_json = meta_res.json()

                        filename = meta_json.get("name", "sharepoint_file")
                        download_url = meta_json.get("@microsoft.graph.downloadUrl")
                        if download_url:
                            res = requests.get(download_url)
                            res.raise_for_status()
                            uploaded_bytes = io.BytesIO(res.content)
                            # Process file
                            st.session_state.last_uploaded_file_bytes = None
                            st.session_state.chat_history = []
                            st.session_state.raw_text = get_raw_text(uploaded_bytes.getvalue(), filename)
                            st.session_state.qa = build_qa_engine(st.session_state.raw_text, openai_api_key)
                            st.success(f"✅ {filename} loaded successfully")
                        else:
                            st.error("⚠️ Could not get download URL from Graph metadata.")

                    else:
                        # Folder link / direct site path
                        parsed = urlparse(sharepoint_url)
                        site_hostname = parsed.netloc
                        path_parts = parsed.path.strip("/").split("/")
                        site_name = path_parts[1] if len(path_parts) > 1 else "sites"
                        relative_path = "/".join(path_parts[2:])

                        # Step 3 — Get Site ID
                        site_api = f"https://graph.microsoft.com/v1.0/sites/{site_hostname}:/sites/{site_name}"
                        site_res = requests.get(site_api, headers={"Authorization": f"Bearer {access_token}"})
                        site_res.raise_for_status()
                        site_id = site_res.json()["id"]

                        # Step 4 — Load all files in folder
                        all_files = load_sharepoint_folder(site_id, relative_path, access_token)
                        if not all_files:
                            st.warning("⚠️ No files found in the folder.")
                        else:
                            all_text = ""
                            for file in all_files:
                                all_text += get_raw_text(file["bytes"].getvalue(), file["name"]) + "\n\n"

                            st.session_state.last_uploaded_file_bytes = None
                            st.session_state.chat_history = []
                            st.session_state.raw_text = all_text
                            st.session_state.qa = build_qa_engine(all_text, openai_api_key)
                            st.success(f"✅ Loaded {len(all_files)} files from folder successfully")

                except requests.exceptions.HTTPError as e:
                    st.error(f"❌ HTTP Error: {e.response.status_code} - {e.response.text}")
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {e}")

# ---------------- Process Uploaded File ----------------
if uploaded_bytes and filename:
    if st.session_state.last_uploaded_file_bytes != uploaded_bytes:
        st.session_state.last_uploaded_file_bytes = uploaded_bytes
        st.session_state.chat_history = []

        st.session_state.raw_text = get_raw_text(
            uploaded_bytes if isinstance(uploaded_bytes, bytes) else uploaded_bytes.getvalue(),
            filename
        )
        if not st.session_state.raw_text.strip():
            st.error("❌ No text could be extracted from the uploaded file. Please check file format or content.")
            st.stop()

        st.write("📏 Extracted text length:", len(st.session_state.raw_text))
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

    # Show chat history
    with chat_container:
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['question']}")
            st.markdown(f"**Bot:** {chat['answer']}")
            with st.expander("🔍 Relevant Context"):
                for doc in chat["context"]:
                    st.write(doc.page_content[:300] + "...")

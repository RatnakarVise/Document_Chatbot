import streamlit as st
import os
import io
import requests
from dotenv import load_dotenv
from urllib.parse import urlparse, unquote
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
        st.markdown("Use Azure AD App Registration (Client ID / Secret).")
        sharepoint_url = st.text_input("Enter SharePoint File URL or Sharing Link")

        # Optional manual override
        tenant_id = st.text_input("Tenant ID", value=tenant_id or "")
        client_id = st.text_input("Client ID", value=client_id or "")
        client_secret = st.text_input("Client Secret", value=client_secret or "", type="password")

        if st.button("Load from SharePoint (Graph API)"):
            if sharepoint_url and client_id and client_secret and tenant_id:
                try:
                    # ---------------- Step 1️⃣ — Get Access Token ----------------
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

                    # ---------------- Step 2️⃣ — Detect URL Type ----------------
                    # if "/:b:/" in sharepoint_url or "/:f:/" in sharepoint_url:
                    #     # 📎 Sharing Link Mode
                    #     st.write("Detected: Sharing Link (base64 encoded method)")
                    #     encoded_url = base64.urlsafe_b64encode(sharepoint_url.encode()).decode().rstrip("=")
                    #     share_api = f"https://graph.microsoft.com/v1.0/shares/u!{encoded_url}/driveItem/content"

                    #     st.write("### 🔍 Debug Info")
                    #     st.write(f"**Sharing URL:** {sharepoint_url}")
                    #     st.write(f"**Encoded URL:** {encoded_url}")
                    #     st.write(f"**API Used:** {share_api}")

                    #     res = requests.get(share_api, headers={"Authorization": f"Bearer {access_token}"})
                    #     res.raise_for_status()

                    #     uploaded_bytes = io.BytesIO(res.content)
                    #     filename = "sharepoint_file"  # fallback name
                    #     st.write(f"**filename:** {filename}")
                    #     st.write(f"**uploaded_bytes:** {uploaded_bytes}")
                    #     st.success("✅ File loaded successfully via Sharing Link")
                    if "/:b:/" in sharepoint_url or "/:f:/" in sharepoint_url:
                        st.write("Detected: Sharing Link (base64 encoded method)")
                        encoded_url = base64.urlsafe_b64encode(sharepoint_url.strip().encode("utf-8")).decode("utf-8").rstrip("=")

                        # Step 1️⃣ — Get metadata
                        meta_url = f"https://graph.microsoft.com/v1.0/shares/u!{encoded_url}/driveItem"
                        meta_res = requests.get(meta_url, headers={"Authorization": f"Bearer {access_token}"})
                        meta_res.raise_for_status()
                        meta_json = meta_res.json()
                        
                        # Extract file name
                        filename = meta_json.get("name", "sharepoint_file")
                        st.write(f"**File Name:** {filename}")

                        # Step 2️⃣ — Get download URL
                        download_url = meta_json.get("@microsoft.graph.downloadUrl")
                        if download_url:
                            res = requests.get(download_url)
                            res.raise_for_status()
                            uploaded_bytes = io.BytesIO(res.content)
                            st.write(f"**uploaded_bytes:** {uploaded_bytes}")
                            st.success(f"✅ {filename} loaded successfully via Sharing Link")
                        else:
                            st.error("⚠️ Could not get download URL from Graph metadata.")


                    else:
                        # 🌐 Direct Site File Path Mode
                        parsed = urlparse(sharepoint_url)
                        site_hostname = parsed.netloc
                        path_parts = parsed.path.strip("/").split("/")
                        site_name = path_parts[1] if len(path_parts) > 1 else "sites"
                        relative_file_path = "/".join(path_parts[2:])

                        st.write("### 🔍 Debug Info")
                        st.write(f"**Original URL:** {sharepoint_url}")
                        st.write(f"**Site Hostname:** {site_hostname}")
                        st.write(f"**Site Name:** {site_name}")
                        st.write(f"**Relative File Path:** {relative_file_path}")

                        # Step 3️⃣ — Get Site ID
                        site_api = f"https://graph.microsoft.com/v1.0/sites/{site_hostname}:/sites/{site_name}"
                        site_res = requests.get(site_api, headers={"Authorization": f"Bearer {access_token}"})
                        site_res.raise_for_status()
                        site_id = site_res.json()["id"]
                        st.write(f"✅ Site ID: {site_id}")

                        # Step 4️⃣ — Get File Content
                        drive_api = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root:/{relative_file_path}:/content"
                        file_res = requests.get(drive_api, headers={"Authorization": f"Bearer {access_token}"})
                        file_res.raise_for_status()

                        uploaded_bytes = io.BytesIO(file_res.content)
                        filename = os.path.basename(relative_file_path)
                        st.success(f"✅ {filename} loaded successfully from SharePoint (Graph API)")

                except requests.exceptions.HTTPError as e:
                    st.error(f"❌ HTTP Error: {e.response.status_code} - {e.response.text}")
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {e}")
            else:
                st.warning("⚠️ Please fill Tenant ID, Client ID, Client Secret, and URL.")

    # ---------------- Process File ----------------
    if uploaded_bytes and filename:
        # Reset chat if file changed
        if st.session_state.last_uploaded_file_bytes != uploaded_bytes:
            st.session_state.last_uploaded_file_bytes = uploaded_bytes
            st.session_state.chat_history = []

            # Extract text
            st.session_state.raw_text = get_raw_text(
                uploaded_bytes if isinstance(uploaded_bytes, bytes) else uploaded_bytes.getvalue(),
                filename
            )
            if not st.session_state.raw_text.strip():
                st.error("❌ No text could be extracted from the uploaded file. Please check file format or content.")
                st.stop()

            st.write("📏 Extracted text length:", len(st.session_state.raw_text))
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

import os
import re
from typing import Optional, Tuple
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ----------------------- CONFIG -----------------------
DEFAULT_MODEL = "gpt-4o"      # or "gpt-4-turbo", adjust if needed
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# ----------------------- TEXT SANITIZATION -----------------------
def sanitize_text(text: str) -> str:
    """
    Cleans sensitive or client-specific data before vectorization.
    Removes/masks: emails, domains, phone numbers, orgs, persons, locations.
    """
    # Replace structured patterns
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[EMAIL]", text)
    text = re.sub(r'\b(?:https?://)?(?:www\.)?[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[DOMAIN]", text)
    text = re.sub(r'\b\d{3,4}[- ]?\d{6,10}\b', "[PHONE]", text)

    # Named entity removal using spaCy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON", "GPE", "LOC"]:
            text = text.replace(ent.text, f"[{ent.label_}]")

    return text

# ----------------------- EMBEDDINGS -----------------------
def _get_embeddings(api_key: str):
    """Helper to get OpenAI embedding model."""
    return OpenAIEmbeddings(openai_api_key=api_key)

# ----------------------- QA ENGINE BUILDER -----------------------
def build_qa_engine(
    raw_text: Optional[str],
    openai_api_key: str,
    model_name: Optional[str] = DEFAULT_MODEL,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    cache_name: Optional[str] = None,
    load_vectorstore_obj: Optional[FAISS] = None
) -> Tuple[RetrievalQA, Optional[FAISS]]:
    """
    Builds or reloads a QA engine.
    - If `load_vectorstore_obj` is provided → uses existing FAISS.
    - Else → sanitizes, chunks, embeds, and creates new FAISS index.
    """
    if load_vectorstore_obj:
        vectorstore = load_vectorstore_obj
    else:
        if not raw_text:
            raise ValueError("Cannot build vectorstore — no raw_text provided.")

        # 1️⃣ Sanitize text before embeddings
        clean_text = sanitize_text(raw_text)

        # 2️⃣ Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(clean_text)

        # 3️⃣ Embed & build FAISS index
        embeddings = _get_embeddings(openai_api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings)

    # 4️⃣ Retriever configuration — controls recall level
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # 5️⃣ LLM setup
    llm = ChatOpenAI(model=model_name, temperature=0.3, openai_api_key=openai_api_key)

    # 6️⃣ Context-based prompt
    prompt_template = """Use the context below to answer accurately and completely.

    - Always use the provided context.
    - Infer missing details logically if partially available.
    - Never reply “I don’t know”; reason based on context.
    - If question is generic, answer in the context domain.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # 7️⃣ QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return qa_chain, vectorstore

# ----------------------- SAVE / LOAD HANDLERS -----------------------
def save_vectorstore(vectorstore: FAISS, persist_dir: str, cache_name: Optional[str] = None):
    """
    Persists FAISS vectorstore locally.
    Works on any server (Render, Azure VM, local, etc.).
    On Azure VM, this persists unless the VM disk resets.
    """
    cache_name = cache_name or "default"
    target_dir = os.path.join(persist_dir, cache_name)
    os.makedirs(target_dir, exist_ok=True)
    vectorstore.save_local(target_dir)

def load_vectorstore(openai_api_key: str, persist_dir: str, cache_name: Optional[str] = None) -> Optional[FAISS]:
    """
    Loads existing FAISS vectorstore if available.
    Returns None if not found.
    """
    cache_name = cache_name or "default"
    target_dir = os.path.join(persist_dir, cache_name)

    if not os.path.isdir(target_dir):
        return None

    embeddings = _get_embeddings(openai_api_key)
    return FAISS.load_local(target_dir, embeddings, allow_dangerous_deserialization=True)

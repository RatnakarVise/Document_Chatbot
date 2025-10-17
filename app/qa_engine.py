import os
import re
import logging
from typing import Optional, Tuple
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ----------------------- LOGGER CONFIG -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("qa_engine")

# ----------------------- CONFIG -----------------------
DEFAULT_MODEL = "gpt-4o"      # or "gpt-4-turbo"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# ----------------------- TEXT SANITIZATION -----------------------
def sanitize_text(text: str) -> str:
    """
    Cleans sensitive or client-specific data before vectorization.
    Removes/masks: emails, domains, phone numbers, orgs, persons, locations.
    """
    logger.info("Starting text sanitization...")

    # Structured patterns first
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[EMAIL]", text)
    text = re.sub(r'\b(?:https?://)?(?:www\.)?[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[DOMAIN]", text)
    text = re.sub(r'\b\d{3,4}[- ]?\d{6,10}\b', "[PHONE]", text)

    try:
        nlp = spacy.load("en_core_web_sm")  # Keep parser/tagger for stability
    except OSError:
        from spacy.cli import download
        logger.warning("spaCy model not found. Downloading 'en_core_web_sm'...")
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    sanitized_text = text

    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON", "GPE", "LOC"]:
            sanitized_text = sanitized_text.replace(ent.text, f"[{ent.label_}]")

    logger.info(f"Sanitization complete. Text length before: {len(text)}, after: {len(sanitized_text)}")
    return sanitized_text

# ----------------------- EMBEDDINGS -----------------------
def _get_embeddings(api_key: str):
    """Helper to get OpenAI embedding model."""
    logger.info("Initializing OpenAI embeddings...")
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
    logger.info("Initializing QA engine...")

    if load_vectorstore_obj:
        logger.info("Using existing vectorstore object.")
        vectorstore = load_vectorstore_obj
    else:
        if not raw_text or not raw_text.strip():
            logger.error("No raw_text provided. Cannot build vectorstore.")
            raise ValueError("Cannot build vectorstore — no raw_text provided.")

        # 1️⃣ Sanitize text before embeddings
        clean_text = sanitize_text(raw_text)
        if not clean_text.strip():
            logger.error("Sanitized text is empty — possible data loss during sanitization.")
            raise ValueError("Sanitized text is empty — check file decoding or sanitization rules.")

        # 2️⃣ Split text into chunks
        logger.info("Splitting text into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(clean_text)
        logger.info(f"Created {len(chunks)} chunks (avg length ≈ {chunk_size}).")

        # 3️⃣ Embed & build FAISS index
        logger.info("Generating embeddings and building FAISS vectorstore...")
        embeddings = _get_embeddings(openai_api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        logger.info("FAISS vectorstore built successfully.")

    # 4️⃣ Retriever configuration
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    logger.info("Retriever configured with top_k = 6.")

    # 5️⃣ LLM setup
    logger.info(f"Initializing LLM model: {model_name}")
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
    logger.info("Building RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    logger.info("QA engine successfully built.")
    return qa_chain, vectorstore

# ----------------------- SAVE / LOAD HANDLERS -----------------------
def save_vectorstore(vectorstore: FAISS, persist_dir: str, cache_name: Optional[str] = None):
    cache_name = cache_name or "default"
    target_dir = os.path.join(persist_dir, cache_name)
    os.makedirs(target_dir, exist_ok=True)
    logger.info(f"Saving FAISS vectorstore at: {target_dir}")
    vectorstore.save_local(target_dir)
    logger.info("Vectorstore saved successfully.")

def load_vectorstore(openai_api_key: str, persist_dir: str, cache_name: Optional[str] = None) -> Optional[FAISS]:
    cache_name = cache_name or "default"
    target_dir = os.path.join(persist_dir, cache_name)

    if not os.path.isdir(target_dir):
        logger.warning(f"No existing FAISS vectorstore found at {target_dir}. Returning None.")
        return None

    logger.info(f"Loading FAISS vectorstore from: {target_dir}")
    embeddings = _get_embeddings(openai_api_key)
    vectorstore = FAISS.load_local(target_dir, embeddings, allow_dangerous_deserialization=True)
    logger.info("Vectorstore loaded successfully.")
    return vectorstore

import os
import re
from typing import Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import spacy

# default model names and chunk config
DEFAULT_MODEL = "gpt-4o"  # adjust to your available model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# ----------------------- SANITIZATION LOGIC -----------------------
def sanitize_text(text: str) -> str:
    """
    Removes or masks client-specific identifiers before vectorization.
    Covers: Emails, URLs/domains, phone numbers, organizations, person names, and locations.
    """
    # Regex-based anonymization for structured identifiers
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[EMAIL]", text)  # Emails
    text = re.sub(r'\b(?:https?://)?(?:www\.)?[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[DOMAIN]", text)  # Domains/URLs
    text = re.sub(r'\b\d{3,4}[- ]?\d{6,10}\b', "[PHONE]", text)  # Phone numbers

    # Load lightweight spaCy model for entity detection
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # fallback if model not downloaded
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    sanitized_text = text

    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON", "GPE", "LOC"]:  # orgs, people, locations
            sanitized_text = sanitized_text.replace(ent.text, f"[{ent.label_}]")

    return sanitized_text

# ------------------------------------------------------------------

def _get_embeddings(api_key: str):
    return OpenAIEmbeddings(openai_api_key=api_key)


def build_qa_engine(raw_text: str,
                    openai_api_key: str,
                    model_name: Optional[str] = DEFAULT_MODEL,
                    chunk_size: int = CHUNK_SIZE,
                    chunk_overlap: int = CHUNK_OVERLAP,
                    cache_name: Optional[str] = None,
                    load_vectorstore_obj=None) -> Tuple[RetrievalQA, Optional[FAISS]]:
    """
    Build or reuse a QA engine. Returns (qa_chain, vectorstore).
    If load_vectorstore_obj is provided, that vectorstore is used directly.
    If cache_name is provided, vectorstore saving/loading functions will use it.
    """

    if load_vectorstore_obj:
        vectorstore = load_vectorstore_obj
    else:
        if not raw_text:
            raise ValueError("No raw_text provided to build new vectorstore.")

        # 🔒 Step 1: Sanitize text before chunking and embedding
        raw_text = sanitize_text(raw_text)

        # 🔹 Step 2: Chunk and embed sanitized text
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(raw_text)
        embeddings = _get_embeddings(openai_api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    llm = ChatOpenAI(model=model_name, temperature=0.3, openai_api_key=openai_api_key)

    prompt_template = """Use the following context to answer the question.
    You are a highly intelligent assistant that answers questions based on the provided context.
    Use the context to provide the **most relevant, informative, and complete** answer possible.
    - If the context partially contains the answer, infer the missing parts logically.
    - If the context does not contain an exact answer, use reasoning or related details from context.
    - Never reply "I don't know." — give the best possible answer using available context.
    - If the question is generic, answer based on common understanding related to the context domain.

    Context:
    {context}

    Question: {question}
    Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )

    return qa, vectorstore


def save_vectorstore(vectorstore: FAISS, persist_dir: str, cache_name: Optional[str] = None):
    if not cache_name:
        cache_name = "default"
    target = os.path.join(persist_dir, cache_name)
    os.makedirs(target, exist_ok=True)
    vectorstore.save_local(target)


def load_vectorstore(openai_api_key: str, persist_dir: str, cache_name: Optional[str] = None) -> Optional[FAISS]:
    if not cache_name:
        cache_name = "default"
    target = os.path.join(persist_dir, cache_name)
    if not os.path.isdir(target):
        return None
    embeddings = _get_embeddings(openai_api_key)
    return FAISS.load_local(target, embeddings, allow_dangerous_deserialization=True)

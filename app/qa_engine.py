# qa_engine.py
import os
from typing import Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# default model names and chunk config
DEFAULT_MODEL = "gpt-4o"  # adjust to your available model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def _get_embeddings(api_key: str):
    return OpenAIEmbeddings(openai_api_key=api_key)

def build_qa_engine(raw_text: str,
                    openai_api_key: str,
                    model_name: Optional[str] = DEFAULT_MODEL,
                    chunk_size: int = CHUNK_SIZE,
                    chunk_overlap: int = CHUNK_OVERLAP,
                    cache_name: Optional[str] = None,
                    load_vectorstore_obj = None) -> Tuple[RetrievalQA, Optional[FAISS]]:
    """
    Build or reuse a QA engine. Returns (qa_chain, vectorstore).
    If load_vectorstore_obj is provided, that vectorstore is used directly.
    If cache_name is provided, vectorstore saving/loading functions will use it.
    """

    # If they passed an already-loaded vectorstore (from load_vectorstore), use it
    if load_vectorstore_obj:
        vectorstore = load_vectorstore_obj
    else:
        # Need raw_text to create embeddings; ensure we have it
        if not raw_text:
            raise ValueError("No raw_text provided to build new vectorstore.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(raw_text)
        embeddings = _get_embeddings(openai_api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings)

    retriever = vectorstore.as_retriever()

    # LLM configuration - pass api_key; if OPENAI_API_BASE is set in env, langchain-openai should use it
    llm = ChatOpenAI(model=model_name, temperature=0.1, openai_api_key=openai_api_key)

    prompt_template = """Use the following context to answer the question.
If the answer is not in the document, say "I don't know."

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
    """
    Save a FAISS vectorstore to disk under persist_dir/cache_name (if provided) or a default directory.
    """
    if not cache_name:
        cache_name = "default"
    target = os.path.join(persist_dir, cache_name)
    os.makedirs(target, exist_ok=True)
    vectorstore.save_local(target)

def load_vectorstore(openai_api_key: str, persist_dir: str, cache_name: Optional[str] = None) -> Optional[FAISS]:
    """
    Load FAISS vectorstore from disk. Returns FAISS object or None if missing.
    """
    if not cache_name:
        # if no cache requested, try to load 'default' if available
        cache_name = "default"
    target = os.path.join(persist_dir, cache_name)
    if not os.path.isdir(target):
        return None
    embeddings = _get_embeddings(openai_api_key)
    return FAISS.load_local(target, embeddings, allow_dangerous_deserialization=True)

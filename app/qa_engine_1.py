from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def build_qa_engine(raw_text, openai_api_key, model_name="gpt-5", chunk_size=1000, chunk_overlap=100):
    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(raw_text)

    # Embeddings & Vector DB
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # LLM
    llm = ChatOpenAI(model=model_name, temperature=0.3, openai_api_key=openai_api_key)

    # Prompt
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

    return qa

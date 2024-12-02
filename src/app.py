import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from prompts import qa_template  
from MyVectorStoreRetriever import MyVectorStoreRetriever  
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configuração da API do Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    raise ValueError("Chave de API do Google Gemini não configurada!")


def get_pdf_text(pdf_docs):
    """Extrair texto dos documentos PDF."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """Dividir o texto em partes menores e gerenciáveis."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    """Criar um banco de vetores usando os embeddings do Google Gemini."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_retrieval_chain(search_type="similarity", score_threshold=None):
    """
    Configurar uma cadeia de recuperação de perguntas e respostas (QA) com retriever customizado.
    """
    # Carregar o template do prompt
    prompt = PromptTemplate(template=qa_template, input_variables=["context", "question"])

    # Configurar os embeddings do Google Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Carregar o índice FAISS
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Configurar o retriever customizado
    retriever = MyVectorStoreRetriever(
        vectorstore=vectorstore,
        search_type=search_type,
        search_kwargs={"k": 10, "score_threshold": score_threshold} if score_threshold else {"k": 10},
    )

    # Configurar o modelo de linguagem do Google Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    # Criar a cadeia de recuperação
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain


def main():
    st.set_page_config("Chat com PDFs")
    st.header("Chat com PDFs usando LangChain 📄")

    # Menu lateral
    with st.sidebar:
        st.title("Configurações")
        search_type = st.selectbox("Tipo de Busca", ["similarity", "similarity_score_threshold", "mmr"])
        score_threshold = (
            float(
                st.number_input(
                    "Threshold de Similaridade (para similarity_score_threshold)",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.1,
                    value=0.1,
                )
            )
            if search_type == "similarity_score_threshold"
            else None
        )


        if search_type == "similarity_score_threshold" and score_threshold == 0.0:
            st.error(
                "O valor de threshold 0.0 é inválido. Por favor, insira um número entre 0.1 e 1.0 para evitar resultados irrelevantes."
            )
            st.stop() 

        pdf_docs = st.file_uploader("Envie seus PDFs", accept_multiple_files=True)
        if st.button("Processar PDFs"):
            with st.spinner("Processando..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs processados com sucesso!")

    # Conteúdo principal
    user_question = st.text_input("Faça uma pergunta sobre seus PDFs")
    if user_question:
        with st.spinner("Buscando resposta..."):
            chain = get_retrieval_chain(search_type=search_type, score_threshold=score_threshold)
            response = chain.run(user_question)
            st.write("Resposta:", response) 


if __name__ == "__main__":
    main()

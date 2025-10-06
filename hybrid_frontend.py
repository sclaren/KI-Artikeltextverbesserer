import streamlit as st
import os
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# Wir brauchen jetzt nur noch DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

# ======================= ⚙️ KONFIGURATION =======================
# (Keine Änderungen hier)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "Demo1234")

# ======================= 🔐 PASSWORT-SCHUTZ =======================
# (Keine Änderungen hier)
def check_password():
    try:
        if st.session_state["password_correct"]: return True
    except KeyError: pass
    st.header("Login")
    password = st.text_input("Bitte gib das Passwort für die Demo ein:", type="password")
    if password == APP_PASSWORD:
        st.session_state["password_correct"] = True
        st.rerun()
    elif password != "": st.error("Das eingegebene Passwort ist falsch.")
    return False

# ======================= ⛓️ RAG-SETUP MIT ORDNER-TRENNUNG (FINAL) =======================
@st.cache_resource
def setup_retrievers():
    st.info("Initialisiere RAG-System mit getrennten Wissensdatenbanken...")
    
    # Gemeinsame Komponenten
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    # --- 1. Wissensbasis für die Analyse ---
    # NEU: Lädt jetzt mehrere Dateitypen
    analysis_loader = DirectoryLoader('./data/analyse/', glob="**/*[.txt,.md,.html]", show_progress=True, recursive=True)
    analysis_docs = analysis_loader.load()
    analysis_splits = text_splitter.split_documents(analysis_docs)
    analysis_vectorstore = FAISS.from_documents(documents=analysis_splits, embedding=embeddings)
    analysis_retriever = analysis_vectorstore.as_retriever()
    
    # --- 2. Wissensbasis für die kreative Erstellung ---
    # NEU: Lädt jetzt mehrere Dateitypen
    creative_loader_erstellung = DirectoryLoader('./data/erstellung/', glob="**/*[.txt,.md,.html]", show_progress=True, recursive=True)
    creative_loader_analyse = DirectoryLoader('./data/analyse/', glob="**/*[.txt,.md,.html]", show_progress=True, recursive=True)
    creative_docs = creative_loader_erstellung.load() + creative_loader_analyse.load() # Kombiniert das Wissen
    creative_splits = text_splitter.split_documents(creative_docs)
    creative_vectorstore = FAISS.from_documents(documents=creative_splits, embedding=embeddings)
    creative_retriever = creative_vectorstore.as_retriever()

    st.success("RAG-System ist bereit!")
    return analysis_retriever, creative_retriever

def setup_chain(retriever, prompt_file, temperature=0.1):
    # (Keine Änderungen hier)
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except FileNotFoundError:
        st.error(f"Fehler: Die Datei '{prompt_file}' wurde nicht gefunden.")
        return None
    
    prompt = PromptTemplate.from_template(prompt_text)
    llm = AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=temperature,
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    return rag_chain

# ======================= 🎨 STREAMLIT OBERFLÄCHE (VEREINFACHT) =======================
st.set_page_config(page_title="Article-QA v9 (Robustes RAG-System)", layout="wide")

if check_password():
    analysis_retriever, creative_retriever = setup_retrievers()
    st.title("🛍️ Article-QA v9 (Robustes RAG-System)")

    article_text = st.text_area(
        "Füge hier den originalen Artikeltext ein:",
        height=250,
        placeholder="z.B. Leckerer Schinken. Jetzt kaufen..."
    )
    st.markdown("---")

    # --- BEREICH 1: ANALYSE ---
    st.header("1. Text-Analyse & Bewertung")
    st.markdown("Analysiere den oben eingefügten Text anhand des universellen Bewertungs-Regelwerks.")
    
    if st.button("Analyse starten"):
        if article_text and analysis_retriever:
            analysis_chain = setup_chain(analysis_retriever, "systemprompt_bewertung.txt", temperature=0.1)
            if analysis_chain:
                with st.spinner("Das Analyse-System arbeitet..."):
                    try:
                        response = analysis_chain.invoke({"input": article_text})
                        answer = response.get("answer", "")
                        st.markdown("### Ergebnis der Analyse")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Ein Fehler ist aufgetreten: {e}")
        else:
            st.warning("Bitte gib einen Text ein oder warte, bis das RAG-System bereit ist.")

    st.markdown("---")

    # --- BEREICH 2: TEXT-ERSTELLUNG ---
    st.header("2. KI-gestützte Texterstellung")
    st.markdown("Lass die KI einen neuen, optimierten Artikeltext basierend auf dem Original erstellen.")

    style = st.select_slider(
        "Stil:",
        options=["Sachlich", "Moderat", "Kreativ"],
        value="Moderat"
    )
    do_rezept = st.checkbox("Rezeptvorschlag generieren", value=False)
    
    if st.button("✨ Artikeltext mit KI erstellen"):
        if article_text and creative_retriever:
            temp_map = {"Sachlich": 0.2, "Moderat": 0.5, "Kreativ": 0.8}
            temperature = temp_map.get(style, 0.5)
            
            creative_chain = setup_chain(creative_retriever, "systemprompt_kreativ.txt", temperature=temperature)
            if creative_chain:
                command_string = "(Vorschlag)"
                if do_rezept: command_string += "(Rezept)"
                final_input = f"{command_string}{article_text}"
                with st.spinner(f"Das Kreativ-System generiert den Text im Stil '{style}'..."):
                    try:
                        response = creative_chain.invoke({"input": final_input})
                        answer = response.get("answer", "")
                        st.markdown("### Ergebnis der KI-Texterstellung")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Ein Fehler ist aufgetreten: {e}")
        else:
            st.warning("Bitte gib einen Text ein oder warte, bis das RAG-System bereit ist.")
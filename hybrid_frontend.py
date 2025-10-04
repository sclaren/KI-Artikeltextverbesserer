import streamlit as st
import os
import re # Importieren der Regular Expressions Bibliothek
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

# ======================= ⚙️ KONFIGURATION =======================
# Liest Werte aus der .env
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "Demo1234")

# ======================= 🔐 PASSWORT-SCHUTZ & RAG-SETUP =======================
# (Die Funktionen check_password() und setup_rag_chain() bleiben unverändert)
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

@st.cache_resource
def setup_rag_chain():
    st.info("Initialisiere RAG-System...")
    loader = DirectoryLoader('./data/', glob="**/*.txt", show_progress=True)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    llm = AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0.1,
    )
    try:
        with open("systemprompt.txt", "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except FileNotFoundError:
        st.error("Fehler: Die Datei 'systemprompt.txt' wurde nicht gefunden.")
        return None
    prompt = PromptTemplate.from_template(prompt_text)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    rag_chain = create_retrieval_chain(retriever, document_chain)
    st.success("RAG-System ist bereit!")
    return rag_chain

# ======================= 🎨 STREAMLIT OBERFLÄCHE (FINALE VERSION) =======================
st.set_page_config(page_title="Article-QA v5", layout="wide")

if check_password():
    rag_chain = setup_rag_chain()
    st.title("🛍️ Article-QA v5 (Intelligente RAG-Demo)")

    article_text = st.text_area(
        "Füge hier den zu analysierenden Artikeltext ein:",
        height=250,
        placeholder="z.B. Verschimmeltes Brot. Jetzt kaufen..."
    )

    st.markdown("---")
    st.markdown("**Analyse-Optionen auswählen:**")
    
    # Checkboxen in Spalten anordnen
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        do_wert = st.checkbox("Kompakte Wertung (JSON)", value=True)
    with col2:
        do_protokoll = st.checkbox("Detailliertes Protokoll", value=False)
    with col3:
        do_vorschlag = st.checkbox("Optimierungsvorschlag", value=False)
    with col4:
        # NEUE CHECKBOX
        do_rezept = st.checkbox("Rezeptvorschlag generieren", value=False)

    if st.button("Analyse starten"):
        if article_text and rag_chain:
            
            command_string = ""
            if do_wert:
                command_string += "(Bewertung_Wert)"
            if do_protokoll:
                command_string += "(Bewertung_Protokoll)"
            if do_vorschlag:
                command_string += "(Vorschlag)"
            # NEUE LOGIK FÜR REZEPT
            if do_rezept:
                command_string += "(Rezept)"
            
            if not command_string:
                command_string = "(Bewertung_Protokoll)"
                st.info("Keine Option ausgewählt, führe detailliertes Protokoll als Standard aus.")

            final_input = f"{command_string}{article_text}"
            
            with st.spinner("Das System analysiert den Text..."):
                try:
                    response = rag_chain.invoke({"input": final_input})
                    answer = response.get("answer", "")

                    st.markdown("---")
                    st.header("Ergebnis der Analyse")
                    st.markdown(answer) # Zeige die komplette, formatierte Antwort an

                except Exception as e:
                    st.error(f"Ein Fehler ist aufgetreten: {e}")
        else:
            st.warning("Bitte gib einen Text ein oder warte, bis das RAG-System bereit ist.")
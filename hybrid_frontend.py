import streamlit as st
import os
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
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "Demo1234")

# ======================= 🔐 PASSWORT-SCHUTZ =======================
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

# ======================= ⛓️ RAG-KETTEN SETUP =======================
@st.cache_resource
def setup_rag_chains():
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    # --- Kette 1: Analyse ---
    try:
        with open("systemprompt_bewertung.txt", "r", encoding="utf-8") as f:
            analysis_prompt_text = f.read()
    except FileNotFoundError:
        st.error("Fehler: Die Datei 'systemprompt_bewertung.txt' wurde nicht gefunden.")
        return None, None
    
    analysis_prompt = PromptTemplate.from_template(analysis_prompt_text)
    analysis_llm = AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0.0,
    )
    analysis_document_chain = create_stuff_documents_chain(analysis_llm, analysis_prompt)
    analysis_chain = create_retrieval_chain(retriever, analysis_document_chain)

    # --- Kette 2: Kreativ ---
    try:
        with open("systemprompt_kreativ.txt", "r", encoding="utf-8") as f:
            creative_prompt_text = f.read()
    except FileNotFoundError:
        st.error("Fehler: Die Datei 'systemprompt_kreativ.txt' wurde nicht gefunden.")
        return None, None

    creative_prompt = PromptTemplate.from_template(creative_prompt_text)
    # LLM für die kreative Kette wird dynamisch erstellt
    
    st.success("RAG-System ist bereit!")
    return analysis_chain, creative_prompt, retriever

def get_creative_chain(style: str, prompt: PromptTemplate, retriever):
    temp_map = {"Sachlich": 0.0, "Moderat": 0.5, "Kreativ": 0.9}
    temperature = temp_map.get(style, 0.5)
    
    creative_llm = AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=temperature,
    )
    creative_document_chain = create_stuff_documents_chain(creative_llm, prompt)
    creative_chain = create_retrieval_chain(retriever, creative_document_chain)
    return creative_chain

# ======================= 🎨 STREAMLIT OBERFLÄCHE =======================
st.set_page_config(page_title="Article-QA v6", layout="wide")

if check_password():
    analysis_chain, creative_prompt, retriever = setup_rag_chains()
    st.title("🛍️ Article-QA v6 (Dual-RAG-System)")

    article_text = st.text_area(
        "Füge hier den originalen Artikeltext ein:",
        height=250,
        placeholder="z.B. Verschimmeltes Brot. Jetzt kaufen..."
    )

    st.markdown("---")

    # --- BEREICH 1: ANALYSE ---
    st.header("1. Text-Analyse & Bewertung")
    st.markdown("Analysiere den oben eingefügten Text anhand des hinterlegten Regelwerks.")
    
    col1, col2 = st.columns(2)
    with col1:
        do_wert = st.checkbox("Kompakte Wertung (JSON)", value=True)
    with col2:
        do_protokoll = st.checkbox("Detailliertes Protokoll", value=False)

    if st.button("Analyse starten"):
        if article_text and analysis_chain:
            command_string = ""
            if do_wert: command_string += "(Bewertung_Wert)"
            if do_protokoll: command_string += "(Bewertung_Protokoll)"
            if not command_string:
                command_string = "(Bewertung_Protokoll)"
                st.info("Keine Option ausgewählt, führe detailliertes Protokoll als Standard aus.")
            
            final_input = f"{command_string}{article_text}"
            
            with st.spinner("Das Analyse-System arbeitet..."):
                try:
                    response = analysis_chain.invoke({"input": final_input})
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
        if article_text and creative_prompt and retriever:
            command_string = "(Vorschlag)" # Immer aktiv für diesen Button
            if do_rezept:
                command_string += "(Rezept)"
            
            final_input = f"{command_string}{article_text}"
            
            with st.spinner(f"Das Kreativ-System generiert den Text im Stil '{style}'..."):
                try:
                    # Dynamisch die kreative Kette mit der richtigen Temperatur holen
                    creative_chain = get_creative_chain(style, creative_prompt, retriever)
                    response = creative_chain.invoke({"input": final_input})
                    answer = response.get("answer", "")
                    st.markdown("### Ergebnis der KI-Texterstellung")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"Ein Fehler ist aufgetreten: {e}")
        else:
            st.warning("Bitte gib einen Text ein oder warte, bis das RAG-System bereit ist.")
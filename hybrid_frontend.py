import streamlit as st
import os
import glob # NEU: Importiert, um nach Dateien zu suchen
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders import TextLoader, DirectoryLoader
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

# ======================= ⛓️ DYNAMISCHES RAG-SETUP (IHR VORSCHLAG) =======================
@st.cache_resource
def setup_base_retrievers():
    st.info("Initialisiere RAG-System...")
    
    # Gemeinsame Komponenten
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    # --- 1. Wissensbasis NUR für die Analyse (immer gleich) ---
    analysis_loader = DirectoryLoader('./data/analyse/', glob="**/*.txt")
    analysis_docs = analysis_loader.load()
    analysis_splits = text_splitter.split_documents(analysis_docs)
    analysis_vectorstore = FAISS.from_documents(documents=analysis_splits, embedding=embeddings)
    analysis_retriever = analysis_vectorstore.as_retriever()
    
    st.success("RAG-System ist bereit!")
    return analysis_retriever, embeddings, text_splitter

def get_creative_retriever(article_type, embeddings, text_splitter):
    # Diese Funktion baut jetzt DYNAMISCH die Wissensbasis für die Erstellung
    st.info(f"Lade Regeln für Artikelart: {article_type}...")
    
    # Lädt die allgemeinen Analyse-Regeln
    base_loader = DirectoryLoader('./data/analyse/', glob="**/*.txt")
    docs = base_loader.load()

    # Lädt die spezifischen Struktur-Regeln für den gewählten Typ
    structure_file_path = f'./data/erstellung/regelwerk_struktur_{article_type}.txt'
    if os.path.exists(structure_file_path):
        structure_loader = TextLoader(structure_file_path, encoding='utf-8')
        docs.extend(structure_loader.load())
    else:
        st.warning(f"Keine spezifischen Struktur-Regeln für '{article_type}' gefunden. Nutze nur allgemeine Regeln.")

    creative_splits = text_splitter.split_documents(docs)
    creative_vectorstore = FAISS.from_documents(documents=creative_splits, embedding=embeddings)
    return creative_vectorstore.as_retriever()

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

# ======================= 🎨 STREAMLIT OBERFLÄCHE =======================
st.set_page_config(page_title="Article-QA v8 (Flexibles RAG-System)", layout="wide")

if check_password():
    analysis_retriever_base, embeddings, text_splitter = setup_base_retrievers()
    st.title("🛍️ Article-QA v8 (Flexibles RAG-System)")

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
        if article_text and analysis_retriever_base:
            analysis_chain = setup_chain(analysis_retriever_base, "systemprompt.txt", temperature=0.1)
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

    # NEU: Dynamische Auswahl der Artikelart basierend auf den verfügbaren Regel-Dateien
    try:
        rule_files = glob.glob('./data/erstellung/regelwerk_struktur_*.txt')
        # Extrahiert den "Typ" aus dem Dateinamen (z.B. "lebensmittel")
        article_types = [os.path.basename(f).replace('regelwerk_struktur_', '').replace('.txt', '') for f in rule_files]
        if not article_types:
            st.error("Keine Struktur-Regelwerke im Ordner 'data/erstellung/' gefunden.")
            article_types = ["default"]
    except Exception:
        article_types = ["default"]

    selected_article_type = st.selectbox("Wähle die Artikelart (Regelwerk):", options=article_types)

    style = st.select_slider(
        "Stil:",
        options=["Sachlich", "Moderat", "Kreativ"],
        value="Moderat"
    )
    do_rezept = st.checkbox("Rezeptvorschlag generieren", value=False)
    
    if st.button("✨ Artikeltext mit KI erstellen"):
        if article_text and selected_article_type:
            # Holt den passenden Retriever für die gewählte Artikelart
            creative_retriever = get_creative_retriever(selected_article_type, embeddings, text_splitter)
            
            temp_map = {"Sachlich": 0.2, "Moderat": 0.5, "Kreativ": 0.8}
            temperature = temp_map.get(style, 0.5)
            
            # (Der Rest der Logik bleibt gleich)
            creative_chain = setup_chain(creative_retriever, "systemprompt_creative.txt", temperature=temperature)
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
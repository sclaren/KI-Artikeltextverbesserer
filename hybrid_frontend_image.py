import streamlit as st
import os
import base64
import io
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage
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

# ======================= INITIALISIERUNG DES SESSION STATE=======================
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "main_text_area" not in st.session_state:
    st.session_state.main_text_area = ""
if "expander_state" not in st.session_state:
    st.session_state.expander_state = False

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

# ======================= ⛓️ RAG-SETUP =======================
@st.cache_resource
def setup_retrievers():
    st.info("Initialisiere RAG-System mit getrennten Wissensdatenbanken...")
    
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    analysis_loader = DirectoryLoader('./data/analyse/', glob="**/*[.txt,.md,.html]", show_progress=True, recursive=True)
    analysis_docs = analysis_loader.load()
    analysis_splits = text_splitter.split_documents(analysis_docs)
    analysis_vectorstore = FAISS.from_documents(documents=analysis_splits, embedding=embeddings)
    analysis_retriever = analysis_vectorstore.as_retriever()
    
    creative_loader_erstellung = DirectoryLoader('./data/erstellung/', glob="**/*[.txt,.md,.html]", show_progress=True, recursive=True)
    creative_loader_analyse = DirectoryLoader('./data/analyse/', glob="**/*[.txt,.md,.html]", show_progress=True, recursive=True)
    creative_docs = creative_loader_erstellung.load() + creative_loader_analyse.load()
    creative_splits = text_splitter.split_documents(creative_docs)
    creative_vectorstore = FAISS.from_documents(documents=creative_splits, embedding=embeddings)
    creative_retriever = creative_vectorstore.as_retriever()

    st.success("RAG-System ist bereit!")
    return analysis_retriever, creative_retriever


def setup_chain(retriever, prompt_file, temperature=0.1):
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


# ======================= 🖼️ BILD-ZU-TEXT FUNKTION =======================
def extract_text_from_images(image_files):
    if not image_files:
        st.warning("Bitte laden Sie zuerst Bilder hoch.")
        return ""

    with st.spinner("Analysiere Bilder und extrahiere Text..."):
        try:
            chat = AzureChatOpenAI(
                azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
                api_version=AZURE_OPENAI_API_VERSION,
                temperature=0.1
            )

            message_content = [{"type": "text", "text": "Extrahiere den gesamten Text von den folgenden Bildern. Kombiniere den Text zu einem einzigen, zusammenhängenden Textblock. Gib nur den reinen Text als Antwort zurück."}]
            
            for image_file in image_files:
                img_bytes = image_file.getvalue()
                base64_image = base64.b64encode(img_bytes).decode('utf-8')
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            
            response = chat.invoke([HumanMessage(content=message_content)])
            
            st.success("Text erfolgreich aus Bildern extrahiert!")
            return response.content

        except Exception as e:
            st.error(f"Ein Fehler bei der Bilderkennung ist aufgetreten: {e}")
            return ""

# ======================= 🎨 STREAMLIT OBERFLÄCHE =======================
st.set_page_config(page_title="AI Artikelverbesserer", layout="wide")

if check_password():
    analysis_retriever, creative_retriever = setup_retrievers()
    st.title("🛍️ AI Artikelverbesserer")

    # KORREKTUR: Der Expander für den Bildupload wird jetzt VOR dem Textfeld gezeichnet.
    with st.expander("Optional: Text aus Produktbildern extrahieren", expanded=st.session_state.expander_state):
        uploaded_files_new = st.file_uploader(
            "Laden Sie hier Produktbilder hoch (Vorder- und Rückseite)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            on_change=lambda: setattr(st.session_state, 'expander_state', True)
        )

        if uploaded_files_new:
            st.session_state.uploaded_files = uploaded_files_new

        if st.session_state.uploaded_files:
            st.write("Hochgeladene Bilder:")
            cols = st.columns(len(st.session_state.uploaded_files))
            for i, uploaded_file in enumerate(st.session_state.uploaded_files):
                with cols[i]:
                    st.image(uploaded_file, width=150)
                    if st.button(f"Entfernen", key=f"remove_{uploaded_file.name}"):
                        st.session_state.uploaded_files.pop(i)
                        st.session_state.expander_state = True
                        st.rerun()

        if st.button("Text aus Bildern extrahieren"):
            extracted_text = extract_text_from_images(st.session_state.uploaded_files)
            st.session_state.main_text_area = extracted_text
            st.session_state.uploaded_files = []
            st.session_state.expander_state = False
            st.rerun()
    
    st.markdown("---")

    st.header("Artikeltext")
    
    # Das Textfeld wird jetzt NACH dem Code gezeichnet, der seinen Zustand ändern könnte.
    article_text = st.text_area(
        "Fügen Sie hier den originalen Artikeltext ein (oder nutzen Sie den Bildupload oben):",
        height=300,
        key="main_text_area"
    )

    st.markdown("---")

    # --- BEREICH 1: ANALYSE ---
    st.header("1. Text-Analyse & Bewertung")
    if st.button("Analyse starten"):
        current_text = st.session_state.main_text_area
        if current_text and analysis_retriever:
            analysis_chain = setup_chain(analysis_retriever, "prompts/systemprompt_bewertung.txt", temperature=0.1)
            if analysis_chain:
                with st.spinner("Das Analyse-System arbeitet..."):
                    try:
                        response = analysis_chain.invoke({"input": current_text})
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
    style = st.select_slider("Stil:", options=["Sachlich", "Moderat", "Kreativ"], value="Moderat")
    do_rezept = st.checkbox("Rezeptvorschlag generieren", value=False)
    do_kategorien = st.checkbox("Kategoriervorschlag generieren", value=False)
    
    if st.button("✨ Artikeltext mit KI erstellen"):
        current_text = st.session_state.main_text_area
        if current_text and creative_retriever:
            prompt_file_map = {
                "Sachlich": "prompts/systemprompt_sachlich.txt",
                "Moderat": "prompts/systemprompt_moderat.txt",
                "Kreativ": "prompts/systemprompt_kreativ.txt"
            }
            prompt_file = prompt_file_map.get(style)

            temp_map = {"Sachlich": 0.2, "Moderat": 0.5, "Kreativ": 0.8}
            temperature = temp_map.get(style, 0.5)
            
            creative_chain = setup_chain(creative_retriever, prompt_file, temperature=temperature)
            
            if creative_chain:
                command_string = "(Vorschlag)"
                if do_rezept: command_string += "(Rezept)"
                if do_kategorien: command_string += "(Kategorien)"
                final_input = f"{command_string}{current_text}"
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

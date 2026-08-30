from pathlib import Path

import streamlit as st
try:
    # Nom de package actuel (langchain >= 0.1) : le text splitter vit dans son
    # propre paquet, installé comme dépendance de langchain-community.
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Compat avec de très vieilles versions de langchain.
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Local",
    page_icon="📚",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Configuration du pipeline d'ingestion
# ---------------------------------------------------------------------------

UPLOAD_DIR = Path("uploaded_docs")       # fichiers uploadés, réécrits sur disque pour les Loaders
PERSIST_DIR = Path("chroma_db")          # base vectorielle persistée
COLLECTION_NAME = "rag_local"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Stratégie de chunking :
# - chunk_size=1000 caractères (~200-250 tokens) : reste proche de la fenêtre native
#   du modèle d'embedding (all-MiniLM-L6-v2, max_seq_length=256 tokens), pour éviter
#   qu'un chunk soit tronqué silencieusement lors de la vectorisation.
# - chunk_overlap=150 caractères (~15%) : préserve le contexte à cheval sur deux chunks
#   (une phrase coupée en fin de chunk reste compréhensible grâce au chevauchement),
#   ce qui limite la perte de pertinence au moment de la recherche.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Nombre de chunks remontés par la recherche sémantique (top-k similarité vectorielle).
TOP_K = 4

UPLOAD_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# État de session
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialise les variables persistantes de la session Streamlit."""
    if "messages" not in st.session_state:
        # Historique de la conversation : liste de {"role": "user"|"assistant", "content": str}
        st.session_state.messages = []

    if "documents_indexed" not in st.session_state:
        # Passera à True une fois l'indexation (Étape 2/3) effectuée.
        st.session_state.documents_indexed = False

    if "llm_enabled" not in st.session_state:
        # True  -> Mode "Assistant RAG complet" (retrieval + génération LLM)
        # False -> Mode "Recherche Sémantique pure" (retrieval seul)
        st.session_state.llm_enabled = True

    if "indexed_chunk_count" not in st.session_state:
        st.session_state.indexed_chunk_count = 0


# ---------------------------------------------------------------------------
# Pipeline d'ingestion (Étape 2 : extraction -> chunking -> vectorisation)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embedding_function() -> HuggingFaceEmbeddings:
    """Charge le modèle d'embeddings local une seule fois (coûteux à instancier)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_vectorstore() -> Chroma:
    """Client Chroma persistant unique pour tout le process serveur.

    Important sur Windows : Chroma garde ses fichiers d'index (HNSW) ouverts/
    mappés en mémoire tant que le client Python vit. En créer un nouveau à
    chaque indexation (et vouloir supprimer l'ancien dossier sur disque)
    provoque une erreur "file in use". En réutilisant toujours le même client
    (mis en cache), on peut vider puis repeupler la collection sans jamais
    toucher aux fichiers directement.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=str(PERSIST_DIR),
    )


def save_uploaded_file(uploaded_file) -> Path:
    """Écrit un fichier uploadé (en mémoire) sur disque pour que les DocumentLoaders
    de LangChain (qui attendent un chemin) puissent le lire."""
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def load_document(file_path: Path) -> list:
    """Extrait le texte d'un fichier en un ou plusieurs Document LangChain,
    en choisissant le loader adapté à son extension."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyMuPDFLoader(str(file_path))
    elif suffix in (".txt", ".md"):
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError(f"Format non supporté : {suffix}")

    docs = loader.load()

    # On force la métadonnée "source" au nom de fichier original (et non le chemin
    # temporaire sur disque), pour un affichage propre des sources en Étape 3.
    for doc in docs:
        doc.metadata["source"] = file_path.name

    return docs


def split_documents(documents: list) -> list:
    """Découpe les documents en chunks (voir justification de la stratégie plus haut)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def index_documents(uploaded_files) -> None:
    """Pipeline complet d'ingestion : extraction -> chunking -> embeddings -> ChromaDB.

    Réindexe entièrement la base à chaque appel à partir des fichiers actuellement
    présents dans le file_uploader (approche simple, adaptée à l'échelle du projet).
    """
    # Extraction : un ou plusieurs Document par fichier, métadonnées (source) conservées.
    all_documents = []
    for uploaded_file in uploaded_files:
        file_path = save_uploaded_file(uploaded_file)
        all_documents.extend(load_document(file_path))

    # Chunking
    chunks = split_documents(all_documents)

    # Vectorisation + stockage. On repart d'une collection vide à chaque indexation
    # (pour éviter les doublons entre deux clics successifs), en vidant son contenu
    # via l'API Chroma plutôt qu'en supprimant les fichiers sur disque (cf. docstring
    # de get_vectorstore).
    vectorstore = get_vectorstore()
    existing_ids = vectorstore.get()["ids"]
    if existing_ids:
        vectorstore.delete(ids=existing_ids)
    vectorstore.add_documents(chunks)

    st.session_state.documents_indexed = True
    st.session_state.indexed_chunk_count = len(chunks)


# ---------------------------------------------------------------------------
# Génération de réponse (à implémenter dans les étapes suivantes)
# ---------------------------------------------------------------------------

def retrieve_relevant_chunks(query: str) -> list:
    """
    Recherche sémantique dans le vector store (peuplé par index_documents — Étape 2).

    Aucun modèle génératif n'est appelé ici : on se contente d'encoder `query`
    avec le même modèle d'embeddings que celui utilisé à l'indexation, puis de
    comparer par similarité vectorielle aux chunks stockés dans ChromaDB.

    Retourne les TOP_K Document LangChain les plus proches (chunk.page_content
    = texte du fragment, chunk.metadata["source"] = nom du fichier d'origine).
    """
    if not st.session_state.documents_indexed:
        return []
    return get_vectorstore().similarity_search(query, k=TOP_K)


def format_semantic_results(chunks: list) -> str:
    """Formate les résultats du mode 'Recherche Sémantique pure' : le contenu
    brut de chaque chunk, précédé du nom exact du fichier source (métadonnées)."""
    if not chunks:
        return "Aucun passage pertinent trouvé dans les documents indexés."

    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "source inconnue")
        blocks.append(f"**Extrait {i}** — 📄 `{source}`\n\n{chunk.page_content}")
    return "\n\n---\n\n".join(blocks)


def generate_llm_answer(query: str, context_chunks: list) -> str:
    """
    Génère une réponse via le LLM local (Ollama), contrainte par `context_chunks`.

    TODO (Étape 4) : construire le prompt (contexte + question) et appeler Ollama
    (ex: via langchain_community.llms.Ollama ou l'API Ollama directement).
    """
    return "*(réponse du LLM à implémenter — Étape suivante)*"


def build_response(query: str) -> str:
    """Construit la réponse à afficher selon le mode actif (toggle)."""
    if not st.session_state.documents_indexed:
        return "⚠️ Merci d'indexer au moins un document avant de poser une question."

    chunks = retrieve_relevant_chunks(query)

    if st.session_state.llm_enabled:
        # Mode "Assistant RAG complet" : retrieval + génération LLM.
        return generate_llm_answer(query, chunks)
    else:
        # Mode "Recherche Sémantique pure" : aucun appel au LLM, on renvoie les
        # passages bruts trouvés par similarité vectorielle.
        return format_semantic_results(chunks)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.header("📁 Documents")

        uploaded_files = st.file_uploader(
            label="Charger des documents",
            type=["pdf", "md", "txt"],
            accept_multiple_files=True,
            help="Formats acceptés : PDF, Markdown, TXT",
        )

        if st.button("🔍 Indexer les documents", use_container_width=True):
            if uploaded_files:
                with st.spinner("Indexation en cours... (le premier lancement télécharge le modèle d'embeddings)"):
                    try:
                        index_documents(uploaded_files)
                    except Exception as exc:
                        st.error(f"Erreur pendant l'indexation : {exc}")
                    else:
                        st.success(
                            f"{len(uploaded_files)} document(s) indexé(s) "
                            f"— {st.session_state.indexed_chunk_count} chunks générés."
                        )
            else:
                st.warning("Aucun fichier sélectionné.")

        if st.session_state.documents_indexed:
            st.caption("✅ Base documentaire prête")
        else:
            st.caption("⏳ Aucun document indexé")

        st.divider()

        st.header("⚙️ Mode de fonctionnement")

        st.session_state.llm_enabled = st.toggle(
            "Assistant RAG complet (LLM activé)",
            value=st.session_state.llm_enabled,
            help=(
                "Activé : recherche + génération de réponse par le LLM local.\n"
                "Désactivé : recherche sémantique pure (passages bruts, sans LLM)."
            ),
        )

        mode_label = "🤖 Assistant RAG complet" if st.session_state.llm_enabled else "🔎 Recherche Sémantique pure"
        st.caption(f"Mode actif : **{mode_label}**")


# ---------------------------------------------------------------------------
# Zone principale (chat)
# ---------------------------------------------------------------------------

def render_chat() -> None:
    st.title("📚 RAG Local")
    st.caption("Posez une question sur vos documents — 100% local, aucune donnée envoyée à l'extérieur.")

    # Affichage de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Saisie utilisateur
    prompt = st.chat_input("Votre question...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):
                answer = build_response(prompt)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()

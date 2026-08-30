import os
from pathlib import Path

# Désactive la télémétrie anonyme de ChromaDB (appel réseau externe à la création
# du client). Doit être fait AVANT tout import de chromadb : sur un réseau qui
# bloque cet appel sortant au lieu de le refuser tout de suite, celui-ci attend
# le timeout et peut ajouter 30-60s au démarrage — sans compter que ça va à
# l'encontre du principe "100% local" du projet.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import streamlit as st
try:
    # Nom de package actuel (langchain >= 0.1) : le text splitter vit dans son
    # propre paquet, installé comme dépendance de langchain-community.
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Compat avec de très vieilles versions de langchain.
    from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
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
# Configuration du LLM local (Étape 4)
# ---------------------------------------------------------------------------

# Nom du modèle tel que chargé dans Ollama (`ollama pull mistral` / `ollama pull
# qwen2.5-coder`, etc.). Modifiable dans la sidebar sans toucher au code.
DEFAULT_OLLAMA_MODEL = "mistral"

# URL du serveur Ollama local (API REST exposée par `ollama serve`, lancé en
# tâche de fond quand tu fais `ollama run ...`). Port par défaut : 11434.
# Explicité ici plutôt que de laisser la valeur par défaut implicite de la
# classe Ollama, pour bien montrer qu'il s'agit d'un appel réseau 100% local.
OLLAMA_BASE_URL = "http://localhost:11434"

# Prompt système strict : le LLM ne doit répondre qu'à partir du {context} fourni
# (les chunks retrouvés à l'Étape 3), jamais à partir de connaissances externes,
# et doit explicitement dire quand l'information est absente du contexte plutôt
# que d'halluciner une réponse plausible.
RAG_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Tu es un assistant documentaire. Réponds à la question UNIQUEMENT à partir "
        "du contexte ci-dessous, extrait des documents fournis par l'utilisateur.\n\n"
        "Règles strictes :\n"
        "- N'utilise aucune connaissance en dehors de ce contexte.\n"
        "- Si le contexte ne contient pas la réponse, dis exactement : "
        "\"Je ne trouve pas cette information dans les documents fournis.\"\n"
        "- Ne fais aucune supposition, n'invente rien.\n"
        "- Réponds de façon claire et concise, en français.\n\n"
        "--- Contexte ---\n"
        "{context}\n"
        "--- Fin du contexte ---\n\n"
        "Question : {question}\n"
        "Réponse :"
    ),
)


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

    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = DEFAULT_OLLAMA_MODEL


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


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str) -> Ollama:
    """Client vers le modèle Ollama local (mis en cache par nom de modèle).

    C'est ici que l'URL du serveur Ollama (OLLAMA_BASE_URL) est branchée :
    chaque appel `.invoke()` sur l'objet retourné fait un POST vers
    `{OLLAMA_BASE_URL}/api/generate`.
    """
    return Ollama(model=model_name, base_url=OLLAMA_BASE_URL)


def build_context_text(chunks: list) -> str:
    """Assemble les chunks retrouvés en un seul bloc de texte pour le prompt,
    en taguant chacun avec son fichier source."""
    blocks = [f"[Source : {chunk.metadata.get('source', 'inconnue')}]\n{chunk.page_content}" for chunk in chunks]
    return "\n\n".join(blocks)


def generate_llm_answer(query: str, context_chunks: list) -> str:
    """Génère une réponse via le LLM local (Ollama), contrainte par `context_chunks`
    grâce au prompt strict RAG_PROMPT_TEMPLATE."""
    if not context_chunks:
        return "Je ne trouve pas cette information dans les documents fournis (aucun extrait pertinent retrouvé)."

    prompt = RAG_PROMPT_TEMPLATE.format(
        context=build_context_text(context_chunks),
        question=query,
    )

    try:
        llm = get_llm(st.session_state.ollama_model)
        return llm.invoke(prompt)
    except Exception as exc:
        return (
            f"⚠️ Impossible de contacter le modèle Ollama `{st.session_state.ollama_model}`.\n\n"
            f"Vérifie qu'Ollama tourne (`ollama serve`) et que le modèle est bien chargé "
            f"(`ollama pull {st.session_state.ollama_model}`).\n\nErreur : {exc}"
        )


def build_response(query: str) -> tuple:
    """Construit la réponse à afficher selon le mode actif (toggle).

    Retourne (texte_affiché, chunks_sources) : `chunks_sources` n'est renseigné
    qu'en mode "Assistant RAG complet" (pour l'expander de transparence) — en
    mode "Recherche Sémantique pure", les extraits sont déjà le contenu principal
    de la réponse, un expander séparé serait redondant.
    """
    if not st.session_state.documents_indexed:
        return "⚠️ Merci d'indexer au moins un document avant de poser une question.", []

    chunks = retrieve_relevant_chunks(query)

    if st.session_state.llm_enabled:
        # Mode "Assistant RAG complet" : retrieval + génération LLM.
        answer = generate_llm_answer(query, chunks)
        return answer, chunks
    else:
        # Mode "Recherche Sémantique pure" : aucun appel au LLM, on renvoie les
        # passages bruts trouvés par similarité vectorielle.
        return format_semantic_results(chunks), []


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

        st.session_state.ollama_model = st.text_input(
            "Modèle Ollama",
            value=st.session_state.ollama_model,
            help="Doit correspondre à un modèle déjà tiré via `ollama pull` (ex: mistral, qwen2.5-coder).",
            disabled=not st.session_state.llm_enabled,
        )


# ---------------------------------------------------------------------------
# Zone principale (chat)
# ---------------------------------------------------------------------------

def render_sources_expander(sources: list) -> None:
    """Élément de transparence (Étape 4) : permet de déplier les extraits de
    texte ayant servi de contexte à la réponse du LLM."""
    if not sources:
        return
    with st.expander("📎 Vérifier les sources utilisées"):
        st.markdown(format_semantic_results(sources))


def render_chat() -> None:
    st.title("📚 RAG Local")
    st.caption("Posez une question sur vos documents — 100% local, aucune donnée envoyée à l'extérieur.")

    # Affichage de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources_expander(message.get("sources"))

    # Saisie utilisateur
    prompt = st.chat_input("Votre question...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):
                answer, sources = build_response(prompt)
            st.markdown(answer)
            render_sources_expander(sources)

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()

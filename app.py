"""RAG local (PDF/Markdown/TXT) avec Streamlit, ChromaDB et Ollama.

Deux modes, activables via un toggle dans la sidebar :
- Recherche Sémantique pure : renvoie les chunks les plus pertinents, sans LLM.
- Assistant RAG complet : ajoute une génération par un LLM local (Ollama),
  contrainte à ne répondre qu'à partir du contexte retrouvé.
"""

import json
import os
import uuid
from pathlib import Path

# Doit être fait avant l'import de chromadb : désactive un appel réseau de
# télémétrie qui peut bloquer 30-60s sur un réseau qui le filtre en silence.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import streamlit as st
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma

st.set_page_config(
    page_title="RAG Local",
    page_icon="📚",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

UPLOAD_DIR = Path("uploaded_docs")
PERSIST_DIR = Path("chroma_db")
COLLECTION_NAME = "rag_local"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHATS_FILE = Path("chats.json")  # historique des discussions, survit à un redémarrage

# Stratégie de chunking :
# - 1000 caractères (~200-250 tokens) : reste sous la fenêtre du modèle
#   d'embedding (max_seq_length=256 tokens) pour éviter une troncature silencieuse.
# - 150 caractères de chevauchement (~15%) : évite qu'une phrase coupée en
#   bord de chunk perde son sens lors de la recherche.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

TOP_K = 4  # nombre de chunks remontés par recherche

UPLOAD_DIR.mkdir(exist_ok=True)

DEFAULT_OLLAMA_MODEL = "mistral"  # doit correspondre à un modèle déjà tiré via `ollama pull`
OLLAMA_BASE_URL = "http://localhost:11434"

# Prompt strict : le LLM ne doit répondre qu'à partir du contexte fourni.
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

NEW_CHAT_TITLE = "Nouvelle discussion"


def init_session_state() -> None:
    """Initialise les variables de session au premier chargement de la page."""
    if "chats" not in st.session_state:
        if not load_chats_from_disk():
            first_id = str(uuid.uuid4())
            st.session_state.chats = {first_id: {"title": NEW_CHAT_TITLE, "messages": []}}
            st.session_state.current_chat_id = first_id

    if "documents_indexed" not in st.session_state:
        st.session_state.documents_indexed = False

    if "llm_enabled" not in st.session_state:
        st.session_state.llm_enabled = True  # False = Recherche Sémantique pure

    if "indexed_chunk_count" not in st.session_state:
        st.session_state.indexed_chunk_count = 0

    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = DEFAULT_OLLAMA_MODEL


# ---------------------------------------------------------------------------
# Ingestion : extraction -> chunking -> embeddings -> ChromaDB
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embedding_function() -> HuggingFaceEmbeddings:
    """Modèle d'embeddings local, chargé une seule fois (coûteux à instancier)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        # normalize_embeddings=True : requis pour que la distance cosinus soit valide.
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def get_vectorstore() -> Chroma:
    """Client Chroma unique et persistant pour tout le process serveur.

    Sur Windows, Chroma garde son index (HNSW) ouvert en mémoire tant que le
    client Python vit : en recréer un à chaque indexation empêcherait de
    supprimer l'ancien dossier ("file in use"). D'où le cache_resource.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=str(PERSIST_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def save_uploaded_file(uploaded_file) -> Path:
    """Écrit un fichier uploadé sur disque (les DocumentLoaders attendent un chemin)."""
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def load_document(file_path: Path) -> list:
    """Extrait le texte d'un fichier en Document(s) LangChain, selon son extension."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyMuPDFLoader(str(file_path))
    elif suffix in (".txt", ".md"):
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError(f"Format non supporté : {suffix}")

    docs = loader.load()
    for doc in docs:
        # Nom de fichier original plutôt que le chemin temporaire sur disque.
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
    """Pipeline complet d'ingestion : extraction -> chunking -> embeddings -> stockage.

    Réindexe entièrement la base à chaque appel, à partir des fichiers
    actuellement présents dans le file_uploader.
    """
    all_documents = []
    for uploaded_file in uploaded_files:
        file_path = save_uploaded_file(uploaded_file)
        all_documents.extend(load_document(file_path))

    chunks = split_documents(all_documents)
    if not chunks:
        raise ValueError(
            "Aucun texte exploitable extrait des fichiers sélectionnés "
            "(PDF scanné/composé d'images, ou fichier vide ?)."
        )

    # On repart d'une collection vide pour éviter les doublons entre deux indexations.
    vectorstore = get_vectorstore()
    existing_ids = vectorstore.get()["ids"]
    if existing_ids:
        vectorstore.delete(ids=existing_ids)
    vectorstore.add_documents(chunks)

    st.session_state.documents_indexed = True
    st.session_state.indexed_chunk_count = len(chunks)


# ---------------------------------------------------------------------------
# Génération de réponse : retrieval + (optionnel) LLM
# ---------------------------------------------------------------------------

def retrieve_relevant_chunks(query: str) -> list:
    """Renvoie les TOP_K chunks les plus proches de `query` par similarité vectorielle."""
    if not st.session_state.documents_indexed:
        return []

    return get_vectorstore().similarity_search(query, k=TOP_K)


def format_semantic_results(chunks: list) -> str:
    """Formate les chunks bruts pour l'affichage : contenu + fichier source."""
    if not chunks:
        return "Aucun passage pertinent trouvé dans les documents indexés."

    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "source inconnue")
        blocks.append(f"**Extrait {i}** — `{source}`\n\n{chunk.page_content}")
    return "\n\n---\n\n".join(blocks)


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str) -> Ollama:
    """Client vers le modèle Ollama local (mis en cache par nom de modèle)."""
    return Ollama(model=model_name, base_url=OLLAMA_BASE_URL)


def build_context_text(chunks: list) -> str:
    """Assemble les chunks en un bloc de texte pour le prompt, tagué par fichier source."""
    blocks = [f"[Source : {chunk.metadata.get('source', 'inconnue')}]\n{chunk.page_content}" for chunk in chunks]
    return "\n\n".join(blocks)


def generate_llm_answer(query: str, context_chunks: list) -> str:
    """Génère une réponse via Ollama, contrainte au contexte par RAG_PROMPT_TEMPLATE."""
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
            f"Impossible de contacter le modèle Ollama `{st.session_state.ollama_model}`.\n\n"
            f"Vérifie qu'Ollama tourne (`ollama serve`) et que le modèle est bien chargé "
            f"(`ollama pull {st.session_state.ollama_model}`).\n\nErreur : {exc}"
        )


def build_response(query: str) -> tuple:
    """Construit la réponse selon le mode actif.

    Retourne (texte_affiché, chunks_sources) : `chunks_sources` n'est rempli
    qu'en mode Assistant RAG complet, pour l'expander de transparence (en mode
    Recherche Sémantique, les extraits sont déjà le contenu principal affiché).
    """
    if not st.session_state.documents_indexed:
        return "Merci d'indexer au moins un document avant de poser une question.", []

    chunks = retrieve_relevant_chunks(query)

    if st.session_state.llm_enabled:
        answer = generate_llm_answer(query, chunks)
        return answer, chunks
    else:
        return format_semantic_results(chunks), []


# ---------------------------------------------------------------------------
# Discussions (plusieurs conversations en parallèle, façon ChatGPT)
# Persistées sur disque dans CHATS_FILE : survivent à un redémarrage du serveur.
# ---------------------------------------------------------------------------

def _message_to_json(message: dict) -> dict:
    """Un message en dict JSON-sérialisable (les sources sont des Document LangChain)."""
    return {
        "role": message["role"],
        "content": message["content"],
        "sources": [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in message.get("sources") or []
        ],
    }


def _message_from_json(data: dict) -> dict:
    """Reconstruit un message (et ses Document sources) depuis le JSON sauvegardé."""
    return {
        "role": data["role"],
        "content": data["content"],
        "sources": [Document(page_content=s["page_content"], metadata=s["metadata"]) for s in data["sources"]],
    }


def save_chats_to_disk() -> None:
    """Écrit toutes les discussions dans CHATS_FILE."""
    payload = {
        "current_chat_id": st.session_state.current_chat_id,
        "chats": {
            chat_id: {
                "title": chat["title"],
                "messages": [_message_to_json(m) for m in chat["messages"]],
            }
            for chat_id, chat in st.session_state.chats.items()
        },
    }
    CHATS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chats_from_disk() -> bool:
    """Recharge les discussions depuis CHATS_FILE dans st.session_state. False si rien à charger."""
    if not CHATS_FILE.exists():
        return False

    try:
        payload = json.loads(CHATS_FILE.read_text(encoding="utf-8"))
        chats = {
            chat_id: {
                "title": chat["title"],
                "messages": [_message_from_json(m) for m in chat["messages"]],
            }
            for chat_id, chat in payload["chats"].items()
        }
    except (json.JSONDecodeError, KeyError, OSError):
        return False

    if not chats:
        return False

    st.session_state.chats = chats
    current_id = payload.get("current_chat_id")
    st.session_state.current_chat_id = current_id if current_id in chats else next(iter(chats))
    return True


def get_current_chat() -> dict:
    """Discussion actuellement affichée : {'title': str, 'messages': list}."""
    return st.session_state.chats[st.session_state.current_chat_id]


def find_empty_chat_id() -> str | None:
    """Id d'une discussion sans aucun message, s'il en existe une."""
    for chat_id, chat in st.session_state.chats.items():
        if not chat["messages"]:
            return chat_id
    return None


def start_new_chat() -> None:
    """Crée une discussion vide et la rend active — ou réutilise celle déjà vide s'il y en a une."""
    existing_empty_id = find_empty_chat_id()
    if existing_empty_id is not None:
        st.session_state.current_chat_id = existing_empty_id
    else:
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": NEW_CHAT_TITLE, "messages": []}
        st.session_state.current_chat_id = new_id
    save_chats_to_disk()


def delete_chat(chat_id: str) -> None:
    """Supprime une discussion. Bascule sur une autre si c'était la discussion active."""
    del st.session_state.chats[chat_id]

    if not st.session_state.chats:
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": NEW_CHAT_TITLE, "messages": []}
        st.session_state.current_chat_id = new_id
    elif st.session_state.current_chat_id == chat_id:
        st.session_state.current_chat_id = next(iter(st.session_state.chats))

    save_chats_to_disk()


def maybe_set_chat_title(chat: dict, first_message: str) -> None:
    """Donne un titre à la discussion à partir de la première question posée."""
    if chat["title"] == NEW_CHAT_TITLE:
        chat["title"] = first_message if len(first_message) <= 40 else first_message[:40] + "…"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_chat_history_sidebar() -> None:
    """Bouton "Nouveau chat" + liste des discussions précédentes (plus récente en premier)."""
    with st.container(border=True):
        st.subheader("Discussions", anchor=False)

        if st.button("Nouveau chat", type="primary", use_container_width=True):
            start_new_chat()
            st.rerun()

        for chat_id in reversed(list(st.session_state.chats.keys())):
            chat = st.session_state.chats[chat_id]
            is_active = chat_id == st.session_state.current_chat_id
            label = f"● {chat['title']}" if is_active else f"○ {chat['title']}"

            col_select, col_delete = st.columns([5, 1])
            with col_select:
                if st.button(label, key=f"chat_btn_{chat_id}", use_container_width=True, disabled=is_active):
                    st.session_state.current_chat_id = chat_id
                    save_chats_to_disk()
                    st.rerun()
            with col_delete:
                if st.button("×", key=f"chat_del_{chat_id}", use_container_width=True):
                    delete_chat(chat_id)
                    st.rerun()


def render_sidebar() -> None:
    """Sidebar complète : discussions, upload/indexation des documents, réglages du mode."""
    with st.sidebar:
        render_chat_history_sidebar()

        with st.container(border=True):
            st.subheader("Documents", anchor=False)

            uploaded_files = st.file_uploader(
                label="Charger des documents",
                type=["pdf", "md", "txt"],
                accept_multiple_files=True,
                help="Formats acceptés : PDF, Markdown, TXT",
                label_visibility="collapsed",
            )

            if st.button("Indexer les documents", type="primary", use_container_width=True):
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

            st.caption("Base documentaire prête" if st.session_state.documents_indexed else "Aucun document indexé")

        with st.container(border=True):
            st.subheader("Mode de fonctionnement", anchor=False)

            st.session_state.llm_enabled = st.toggle(
                "Assistant RAG complet (LLM activé)",
                value=st.session_state.llm_enabled,
                help=(
                    "Activé : recherche + génération de réponse par le LLM local.\n"
                    "Désactivé : recherche sémantique pure (passages bruts, sans LLM)."
                ),
            )

            mode_label = "Assistant RAG complet" if st.session_state.llm_enabled else "Recherche Sémantique pure"
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
    """Élément de transparence : dépliant listant les extraits ayant servi de contexte au LLM."""
    if not sources:
        return
    with st.expander("Vérifier les sources utilisées"):
        st.markdown(format_semantic_results(sources))


def render_chat() -> None:
    """Zone principale : historique de la discussion active + saisie utilisateur."""
    st.title("RAG Local", anchor=False)
    st.caption("Posez une question sur vos documents — 100% local, aucune donnée envoyée à l'extérieur.")

    chat = get_current_chat()

    for message in chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources_expander(message.get("sources"))

    prompt = st.chat_input("Votre question...")
    if prompt:
        chat["messages"].append({"role": "user", "content": prompt})
        maybe_set_chat_title(chat, prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):
                answer, sources = build_response(prompt)
            st.markdown(answer)
            render_sources_expander(sources)

        chat["messages"].append({"role": "assistant", "content": answer, "sources": sources})
        save_chats_to_disk()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    """Initialise l'état de session puis affiche la sidebar et le chat."""
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()

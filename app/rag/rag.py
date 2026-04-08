
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.prompts import PromptTemplate

# BM25 + fusion may live in optional modules depending on version
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

from llama_index.llms.openai import OpenAI

from app.ui.config import PERSIST_DIR


# -----------------------------
# 1) LLM for RAG synthesis (NEUTRAL)
# -----------------------------
# This LLM is used ONLY to synthesize an answer from retrieved nodes.
# Style/mode/verbosity will be handled in api.py rewrite step (Option A).
llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0,
    system_prompt=(
        "You are a retrieval-augmented assistant.\n"
        "Follow these rules strictly:\n"
        "1) Answer in the SAME LANGUAGE as the user's question.\n"
        "2) Use ONLY the provided context as factual basis. Do not add new facts.\n"
        "3) If the context is insufficient, say so clearly.\n"
        "4) Be concise and factual.\n"
    ),
)
Settings.llm = llm


# -----------------------------
# 2) Load index + build hybrid retriever
# -----------------------------
storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
index = load_index_from_storage(storage_context)

vector_retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=40, # default k=20
    vector_store_query_mode="mmr",
)

bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore,
    similarity_top_k=40, # default k=20
)

hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=20,       # final top_k used for answering/citations default k=8
    num_queries=1,
    mode="reciprocal_rerank",
)


# -----------------------------
# 3) Neutral RAG prompt template
# -----------------------------
"""TEXT_QA_PROMPT = PromptTemplate(
    "You are a helpful assistant.\n"
    "IMPORTANT: Answer in the SAME LANGUAGE as the user's question.\n"
    "Use ONLY the provided context as factual basis. Do not add new facts.\n"
    "If the context is insufficient to answer, say so explicitly.\n"
    "\n"
    "Context:\n"
    "{context_str}\n"
    "\n"
    "Question:\n"
    "{query_str}\n"
    "\n"
    "Answer:"
)"""

TEXT_QA_PROMPT = PromptTemplate(
    "You are a helpful assistant.\n"
    "IMPORTANT: Answer in the SAME LANGUAGE as the user's question.\n"
    "Use the provided context as the factual basis.\n"
    "If the context is insufficient, do TWO parts:\n"
    "A) 'From sources' (only what is supported by context).\n"
    "B) 'General explanation (not in sources)' (helpful background, clearly marked).\n"
    "\n"
    "Context:\n"
    "{context_str}\n"
    "\n"
    "Question:\n"
    "{query_str}\n"
    "\n"
    "Answer:"
)


# -----------------------------
# 4) Query engine
# -----------------------------
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    response_mode="compact",
    text_qa_template=TEXT_QA_PROMPT,
)


# -----------------------------
# 5) Public function used by API
# -----------------------------
def query_rag(question: str) -> dict:
    resp = query_engine.query(question)

    sources = []
    for i, sn in enumerate(getattr(resp, "source_nodes", []) or [], start=1):
        node = sn.node
        meta = dict(getattr(node, "metadata", {}) or {})
        sources.append({
            "id": i,
            "score": float(getattr(sn, "score", 0.0) or 0.0),
            "source_type": meta.get("source_type") or "unknown",
            "file_name": meta.get("file_name") or meta.get("clean_file") or meta.get("file_name"),
            "source_path": meta.get("source_path"),
            "url": meta.get("url"),
            "domain": meta.get("domain"),
            "clean_file": meta.get("clean_file") or meta.get("file_name"),
            "snippet": (node.get_text()[:240] + "…") if hasattr(node, "get_text") else None,
            "label": meta.get("url") or meta.get("file_name") or meta.get("clean_file") or "unknown"
        })

    return {"answer": str(resp), "sources": sources}

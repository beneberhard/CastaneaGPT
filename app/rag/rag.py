import os
from functools import lru_cache

from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever

from app.ui.config import PERSIST_DIR

load_dotenv()


TEXT_QA_PROMPT = PromptTemplate(
    "You are a helpful assistant.\n"
    "IMPORTANT: Answer in the SAME LANGUAGE as the user's question.\n"
    "Use ONLY the provided context as the factual basis.\n"
    "Do NOT add new facts.\n"
    "If the context is insufficient, say so explicitly.\n"
    "\n"
    "Context:\n"
    "{context_str}\n"
    "\n"
    "Question:\n"
    "{query_str}\n"
    "\n"
    "Answer:"
)


@lru_cache(maxsize=1)
def get_llm():
    llm = OpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
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
    return llm


@lru_cache(maxsize=1)
def get_index():
    storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
    return load_index_from_storage(storage_context)


@lru_cache(maxsize=1)
def get_hybrid_retriever():
    index = get_index()
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=40,
        vector_store_query_mode="mmr",
    )
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore,
        similarity_top_k=40,
    )
    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=20,
        num_queries=1,
        mode="reciprocal_rerank",
    )


@lru_cache(maxsize=1)
def get_query_engine():
    get_llm()
    return RetrieverQueryEngine.from_args(
        retriever=get_hybrid_retriever(),
        response_mode="compact",
        text_qa_template=TEXT_QA_PROMPT,
    )


def query_rag(question: str) -> dict:
    resp = get_query_engine().query(question)

    sources = []
    for i, sn in enumerate(getattr(resp, "source_nodes", []) or [], start=1):
        node = sn.node
        meta = dict(getattr(node, "metadata", {}) or {})
        sources.append(
            {
                "id": i,
                "score": float(getattr(sn, "score", 0.0) or 0.0),
                "source_type": meta.get("source_type") or "unknown",
                "file_name": meta.get("file_name") or meta.get("clean_file") or meta.get("file_name"),
                "source_path": meta.get("source_path"),
                "url": meta.get("url"),
                "domain": meta.get("domain"),
                "clean_file": meta.get("clean_file") or meta.get("file_name"),
                "snippet": (node.get_text()[:240] + "…") if hasattr(node, "get_text") else None,
                "label": meta.get("url") or meta.get("file_name") or meta.get("clean_file") or "unknown",
            }
        )

    return {"answer": str(resp), "sources": sources}

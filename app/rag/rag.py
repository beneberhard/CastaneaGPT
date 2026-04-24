import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever

from app.api.prompting import build_response_policy
from app.ui.config import PERSIST_DIR
from app.utils.glossary import (
    build_glossary_context_block,
    find_glossary_matches,
    normalize_language,
)

load_dotenv()

@lru_cache(maxsize=1)
def get_llm():
    llm = OpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        system_prompt=(
            "You are CastaneaGPT, a retrieval-augmented assistant.\n"
            "Follow these rules strictly:\n"
            "1) Follow the response policy and task instructions exactly.\n"
            "2) Use ONLY the provided context as factual basis. Do not add new facts.\n"
            "3) If the context is insufficient, say so clearly.\n"
            "4) Do not let generic assistant style override the requested mode, verbosity, language, or citation behavior.\n"
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


def build_text_qa_prompt(
    *,
    context_str: str,
    question: str,
    mode: str,
    verbosity: str,
    language: str,
    cite_style: str,
    glossary_block: str = "",
) -> str:
    policy = build_response_policy(
        mode=mode,
        verbosity=verbosity,
        language=language,
        cite_style=cite_style,
    )
    glossary_section = f"\n\n{glossary_block}" if glossary_block else ""
    return (
        f"{policy}\n\n"
        "TASK:\n"
        "- Answer the user's question directly from the provided context.\n"
        "- If the context is insufficient for a requested detail, state that clearly.\n"
        "- Keep the requested mode behavior while remaining grounded in the context.\n"
        f"{glossary_section}\n"
        "\n"
        "Context:\n"
        f"{context_str}\n"
        "\n"
        "Question:\n"
        f"{question}\n"
        "\n"
        "Answer:"
    )


def retrieve_source_nodes(question: str) -> list[Any]:
    return list(get_hybrid_retriever().retrieve(question))


def format_retrieved_context(source_nodes: list[Any], limit: int = 8) -> str:
    chunks = []

    for i, sn in enumerate(source_nodes[:limit], start=1):
        node = getattr(sn, "node", sn)
        meta = dict(getattr(node, "metadata", {}) or {})
        label = meta.get("url") or meta.get("file_name") or meta.get("clean_file") or f"source {i}"
        text = node.get_text() if hasattr(node, "get_text") else str(node)
        chunks.append(f"[Source {i}: {label}]\n{text.strip()}")

    return "\n\n".join(chunks)


def extract_sources(source_nodes: list[Any]) -> list[dict]:
    sources = []
    for i, sn in enumerate(source_nodes or [], start=1):
        node = getattr(sn, "node", sn)
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
    return sources


def query_rag(
    question: str,
    *,
    mode: str,
    verbosity: str,
    language: str,
    cite_style: str,
    language_hint_text: str = "",
) -> dict:
    source_nodes = retrieve_source_nodes(question)
    context_str = format_retrieved_context(source_nodes)
    target_language = normalize_language(language, fallback_text=language_hint_text or question)

    glossary_matches = find_glossary_matches(
        [getattr(getattr(sn, "node", sn), "get_text", lambda: "")() for sn in source_nodes[:8]],
        target_language=target_language,
    )
    glossary_block = build_glossary_context_block(glossary_matches, target_language)

    prompt = build_text_qa_prompt(
        context_str=context_str,
        question=question,
        mode=mode,
        verbosity=verbosity,
        language=language,
        cite_style=cite_style,
        glossary_block=glossary_block,
    )
    response = get_llm().complete(prompt)
    answer_text = getattr(response, "text", None) or str(response)

    return {"answer": answer_text, "sources": extract_sources(source_nodes)}

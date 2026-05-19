from functools import lru_cache
import re
from typing import Any

from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever

from app.llm_config import (
    chat_api_key,
    chat_base_url,
    chat_model,
    embedding_api_key,
    embedding_model,
)
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
        model=chat_model(),
        api_key=chat_api_key(),
        api_base=chat_base_url(),
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
    Settings.embed_model = OpenAIEmbedding(
        model=embedding_model(),
        api_key=embedding_api_key(),
    )
    storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
    return load_index_from_storage(storage_context)


@lru_cache(maxsize=1)
def get_hybrid_retriever():
    get_llm()
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
        num_queries=3,
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


def source_key_from_metadata(meta: dict[str, Any]) -> str:
    return (
        str(
            meta.get("url")
            or meta.get("source_path")
            or meta.get("citation_label")
            or meta.get("title")
            or meta.get("file_name")
            or meta.get("clean_file")
            or ""
        ).strip()
    )


def _clean_source_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\.(pdf|docx|txt)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .")


def _source_year(meta: dict[str, Any]) -> str:
    for key in ("year", "date", "publication_year", "title", "citation_label", "file_name", "clean_file"):
        match = re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", str(meta.get(key) or ""))
        if match:
            return match.group(0)
    return ""


def _source_author(meta: dict[str, Any]) -> str:
    author = _clean_source_text(meta.get("author"))
    if author:
        first_author = re.split(r";|,|\band\b|\bet al\.?", author, flags=re.IGNORECASE)[0]
        return _clean_source_text(first_author)

    name = _clean_source_text(meta.get("file_name") or meta.get("clean_file"))
    match = re.match(
        r"([A-Z][A-Za-zÀ-ÖØ-öø-ÿ']+?)\s*(?:et\s+al|EtAl)\s*(?:\d{4})?$",
        name,
        flags=re.IGNORECASE,
    )
    if match:
        return f"{match.group(1)} et al."
    return ""


def citation_label_from_metadata(meta: dict[str, Any]) -> str:
    if meta.get("url"):
        domain = str(meta.get("domain") or meta.get("url") or "").strip()
        title = _clean_source_text(meta.get("title") or meta.get("citation_label"))
        return f"{domain} - {title}" if title and title != domain else domain

    year = _source_year(meta)
    author = _source_author(meta)
    title = _clean_source_text(
        meta.get("title")
        or meta.get("citation_label")
        or meta.get("file_name")
        or meta.get("clean_file")
        or "unknown"
    )
    if year:
        title = _clean_source_text(re.sub(rf"\b{year}\b", "", title))
    if author and year:
        return f"{author} ({year})"
    if year and title:
        return f"{title} ({year})"
    source_type = _clean_source_text(meta.get("source_type")).lower()
    if source_type in {"pdf", "docx", "txt"} and title:
        return f"{title} [document]"
    return title or "unknown"


def diversify_source_nodes(
    source_nodes: list[Any],
    *,
    recent_source_keys: list[str] | None = None,
    max_total: int = 8,
    max_per_source: int = 2,
) -> list[Any]:
    recent = set(recent_source_keys or [])
    decorated = []

    for idx, sn in enumerate(source_nodes or []):
        node = getattr(sn, "node", sn)
        meta = dict(getattr(node, "metadata", {}) or {})
        key = source_key_from_metadata(meta) or f"source-{idx}"
        score = float(getattr(sn, "score", 0.0) or 0.0)
        decorated.append(
            {
                "sn": sn,
                "key": key,
                "score": score,
                "is_recent": key in recent,
                "rank": idx,
            }
        )

    # Prefer not-recent sources first, then stronger scores, while preserving stable tie-breaking.
    decorated.sort(key=lambda item: (item["is_recent"], -item["score"], item["rank"]))

    selected = []
    per_source_counts: dict[str, int] = {}

    for item in decorated:
        key = item["key"]
        if per_source_counts.get(key, 0) >= max_per_source:
            continue
        selected.append(item["sn"])
        per_source_counts[key] = per_source_counts.get(key, 0) + 1
        if len(selected) >= max_total:
            break

    return selected


def format_retrieved_context(source_nodes: list[Any], limit: int = 8) -> str:
    chunks = []

    for i, sn in enumerate(source_nodes[:limit], start=1):
        node = getattr(sn, "node", sn)
        meta = dict(getattr(node, "metadata", {}) or {})
        label = citation_label_from_metadata(meta) or f"source {i}"
        text = node.get_text() if hasattr(node, "get_text") else str(node)
        chunks.append(f"[Source {i}: {label}]\n{text.strip()}")

    return "\n\n".join(chunks)


def extract_sources(source_nodes: list[Any]) -> list[dict]:
    sources = []
    for i, sn in enumerate(source_nodes or [], start=1):
        node = getattr(sn, "node", sn)
        meta = dict(getattr(node, "metadata", {}) or {})
        label = citation_label_from_metadata(meta)
        sources.append(
            {
                "id": i,
                "score": float(getattr(sn, "score", 0.0) or 0.0),
                "source_type": meta.get("source_type") or "unknown",
                "source_key": source_key_from_metadata(meta),
                "file_name": meta.get("file_name") or meta.get("clean_file") or meta.get("file_name"),
                "source_path": meta.get("source_path"),
                "url": meta.get("url"),
                "domain": meta.get("domain"),
                "title": meta.get("title"),
                "citation_label": meta.get("citation_label"),
                "clean_file": meta.get("clean_file") or meta.get("file_name"),
                "snippet": (node.get_text()[:240] + "…") if hasattr(node, "get_text") else None,
                "label": label,
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
    recent_source_keys: list[str] | None = None,
) -> dict:
    retrieved_nodes = retrieve_source_nodes(question)
    source_nodes = diversify_source_nodes(
        retrieved_nodes,
        recent_source_keys=recent_source_keys,
        max_total=8,
        max_per_source=2,
    )
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

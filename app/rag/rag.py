import os
from functools import lru_cache

from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever

from app.api.prompting import build_response_policy
from app.ui.config import PERSIST_DIR

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


def build_text_qa_prompt(mode: str, verbosity: str, language: str, cite_style: str) -> PromptTemplate:
    policy = build_response_policy(
        mode=mode,
        verbosity=verbosity,
        language=language,
        cite_style=cite_style,
    )
    return PromptTemplate(
        f"{policy}\n\n"
        "TASK:\n"
        "- Answer the user's question directly from the provided context.\n"
        "- If the context is insufficient for a requested detail, state that clearly.\n"
        "- Keep the requested mode behavior while remaining grounded in the context.\n"
        "\n"
        "Context:\n"
        "{{context_str}}\n"
        "\n"
        "Question:\n"
        "{{query_str}}\n"
        "\n"
        "Answer:"
    )


def query_rag(
    question: str,
    *,
    mode: str,
    verbosity: str,
    language: str,
    cite_style: str,
) -> dict:
    query_engine = RetrieverQueryEngine.from_args(
        retriever=get_hybrid_retriever(),
        llm=get_llm(),
        response_mode="compact",
        text_qa_template=build_text_qa_prompt(
            mode=mode,
            verbosity=verbosity,
            language=language,
            cite_style=cite_style,
        ),
    )
    resp = query_engine.query(question)

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

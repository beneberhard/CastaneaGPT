import os


def use_local_llm() -> bool:
    return os.getenv("LLM_PROVIDER", "openai").lower() == "local"


def chat_model() -> str:
    if use_local_llm():
        return os.getenv("LOCAL_MODEL", "local-model")
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def chat_base_url() -> str | None:
    if use_local_llm():
        return os.getenv("LOCAL_BASE_URL", "http://localhost:1234/v1")
    return None


def chat_api_key() -> str | None:
    if use_local_llm():
        return os.getenv("LOCAL_API_KEY", "lm_studio")
    return os.getenv("OPENAI_API_KEY")


def embedding_model() -> str:
    return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def embedding_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")

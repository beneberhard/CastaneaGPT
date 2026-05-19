from __future__ import annotations

import os
from pathlib import Path
import re
import requests

import json
from datetime import datetime
from urllib.parse import urlparse

from app.llm_config import embedding_api_key, embedding_model

def write_meta(out_txt: Path, meta: dict) -> None:
    meta_path = out_txt.with_suffix(out_txt.suffix + ".meta.json")  # .txt.meta.json
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


RAW = Path("data/corpus_raw")
CLEAN = Path("data/corpus_clean")
CLEAN.mkdir(parents=True, exist_ok=True)

# >>> ADDED: where to persist the LlamaIndex index
PERSIST_DIR = Path(os.getenv("INDEX_PERSIST_DIR", "storage/index_v1"))
PERSIST_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def humanize_filename(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"\.(pdf|docx|txt)$", "", stem, flags=re.IGNORECASE)
    stem = stem.replace("_", " ")
    stem = re.sub(r"\s+", " ", stem)
    return stem.strip(" -_.")


def choose_citation_label(*candidates: str) -> str:
    for candidate in candidates:
        cleaned = clean_text(str(candidate or ""))
        if cleaned:
            return cleaned
    return "Unknown source"


def ingest_pdfs():
    import pdfplumber

    pdf_dir = RAW / "pdf"
    for pdf in pdf_dir.glob("*.pdf"):
        out = CLEAN / f"{pdf.stem}.txt"

        meta_out = out.with_suffix(out.suffix + ".meta.json")
        if out.exists() and meta_out.exists():
            continue

        print(f"Ingesting PDF: {pdf.name}")
        pages = []
        pdf_title = ""
        pdf_author = ""
        with pdfplumber.open(pdf) as p:
            metadata = p.metadata or {}
            pdf_title = metadata.get("Title") or metadata.get("title") or ""
            pdf_author = metadata.get("Author") or metadata.get("author") or ""
            for page in p.pages:
                txt = page.extract_text()
                if txt:
                    pages.append(txt)

        citation_label = choose_citation_label(pdf_title, humanize_filename(pdf.name))
        out.write_text(clean_text("\n".join(pages)), encoding="utf-8")
        write_meta(out, {
            "source_type": "pdf",
            "file_name": pdf.name,
            "title": clean_text(pdf_title),
            "author": clean_text(pdf_author),
            "citation_label": citation_label,
            "source_path": str(pdf.resolve()),
            "ingested_at": datetime.now().isoformat(timespec="seconds"),
            })



def ingest_text_files():
    txt_dirs = [RAW / "files", RAW / "txt"]

    for txt_dir in txt_dirs:
        if not txt_dir.exists():
            continue

        for txt in sorted(txt_dir.glob("*.txt")):
            out = CLEAN / txt.name

            meta_out = out.with_suffix(out.suffix + ".meta.json")
            if out.exists() and meta_out.exists():
                continue

            print(f"Ingesting TXT: {txt.name}")
            cleaned_text = clean_text(
                txt.read_text(encoding="utf-8", errors="ignore")
            )

            if not cleaned_text:
                print("  ⚠️ text file is empty after cleaning, skipping")
                continue

            out.write_text(cleaned_text, encoding="utf-8")
            citation_label = choose_citation_label(humanize_filename(txt.name))
            write_meta(out, {
                "source_type": "txt",
                "file_name": txt.name,
                "title": humanize_filename(txt.name),
                "citation_label": citation_label,
                "source_path": str(txt.resolve()),
                "ingested_at": datetime.now().isoformat(timespec="seconds"),
                })


def ingest_word_files():
    from docx import Document as DocxDocument

    docx_files = sorted(RAW.rglob("*.docx"))

    for docx in docx_files:
        safe_name = re.sub(r"\W+", "_", str(docx.relative_to(RAW)).strip()).strip("_")
        out = CLEAN / f"docx_{safe_name}.txt"

        meta_out = out.with_suffix(out.suffix + ".meta.json")
        if out.exists() and meta_out.exists():
            continue

        print(f"Ingesting DOCX: {docx.name}")

        try:
            document = DocxDocument(docx)
        except Exception as e:
            print(f"  ❌ could not open DOCX: {e}")
            continue

        paragraphs = [p.text for p in document.paragraphs if p.text and p.text.strip()]
        cleaned_text = clean_text("\n".join(paragraphs))

        if not cleaned_text:
            print("  ⚠️ DOCX file is empty after cleaning, skipping")
            continue

        docx_title = getattr(document.core_properties, "title", "") or ""
        docx_author = getattr(document.core_properties, "author", "") or ""
        citation_label = choose_citation_label(docx_title, humanize_filename(docx.name))
        out.write_text(cleaned_text, encoding="utf-8")
        write_meta(out, {
            "source_type": "docx",
            "file_name": docx.name,
            "title": clean_text(docx_title),
            "author": clean_text(docx_author),
            "citation_label": citation_label,
            "source_path": str(docx.resolve()),
            "ingested_at": datetime.now().isoformat(timespec="seconds"),
            })


def ingest_urls():
    from bs4 import BeautifulSoup
    from readability import Document

    url_file = RAW / "urls.txt"
    print("Reading URLs from:", url_file.resolve())

    if not url_file.exists():
        print("No urls.txt found, skipping URL ingestion.")
        return

    for url in url_file.read_text(encoding="utf-8").splitlines():
        url = url.strip()
        if not url:
            continue

        slug = re.sub(r"\W+", "_", url)[:60]
        out = CLEAN / f"url_{slug}.txt"
        meta_out = out.with_suffix(out.suffix + ".meta.json")
        if out.exists() and meta_out.exists():
            continue

        print(f"Ingesting URL: {url}")

        try:
            response = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=20,
            )
            response.raise_for_status()
        except Exception as e:
            print(f"  ❌ request failed: {e}")
            continue

        article_text = ""
        full_page_text = ""
        extraction_method = "none"
        page_title = ""

        try:
            doc = Document(response.text)
            article_html = doc.summary(html_partial=True)
            article_soup = BeautifulSoup(article_html, "html.parser")
            article_text = clean_text(article_soup.get_text(" ").strip())
        except Exception as e:
            print(f"  ⚠️ article extraction failed: {e}")

        try:
            full_page_soup = BeautifulSoup(response.text, "html.parser")
            title_tag = full_page_soup.find("meta", property="og:title")
            if title_tag and title_tag.get("content"):
                page_title = title_tag["content"]
            elif full_page_soup.title and full_page_soup.title.string:
                page_title = full_page_soup.title.string
            for tag in full_page_soup(["script", "style", "noscript"]):
                tag.decompose()
            full_page_text = clean_text(full_page_soup.get_text(" ").strip())
        except Exception as e:
            print(f"  ⚠️ raw page extraction failed: {e}")

        text = article_text
        if len(article_text) >= 200:
            extraction_method = "readability"
        elif len(full_page_text) > len(article_text):
            text = full_page_text
            extraction_method = "full_page_fallback"
        elif article_text:
            extraction_method = "readability_short"

        if not text:
            print("  ⚠️ no usable text extracted, skipping")
            continue

        if len(text) < 200:
            print(f"  ⚠️ short extraction kept ({len(text)} chars, method={extraction_method})")
        else:
            print(f"  ✔ extracted {len(text)} chars using {extraction_method}")

        out.write_text(text, encoding="utf-8")
        citation_label = choose_citation_label(page_title, urlparse(url).netloc, url)
        write_meta(out, {
            "source_type": "url",
            "url": url,
            "domain": urlparse(url).netloc,
            "title": clean_text(page_title),
            "citation_label": citation_label,
            "extraction_method": extraction_method,
            "char_count": len(text),
            "ingested_at": datetime.now().isoformat(timespec="seconds"),
            })


# >>> ADDED: build + persist LlamaIndex index from CLEAN/*.txt
# >>> UPDATED: build + persist LlamaIndex index from CLEAN/*.txt + sidecar metadata (*.txt.meta.json)
def build_and_persist_index():
    import json
    import re
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core.node_parser import SentenceSplitter

    # -----------------------------
    # Chunking hyperparameters
    # -----------------------------
    CHUNK_SIZE = 1200        # characters for SentenceSplitter
    CHUNK_OVERLAP = 180      # characters overlap

    # Apply splitter globally for this ingest run
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    Settings.embed_model = OpenAIEmbedding(
        model=embedding_model(),
        api_key=embedding_api_key(),
    )

    print("Building index from:", CLEAN.resolve())
    print(f"Chunking: chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")

    txt_files = sorted(CLEAN.glob("*.txt"))
    if not txt_files:
        print("⚠️ No documents found in corpus_clean. Nothing to index.")
        return

    docs = []
    for txt in txt_files:
        text = txt.read_text(encoding="utf-8", errors="ignore")

        meta = {"clean_file": txt.name}
        meta_path = Path(str(txt) + ".meta.json")  # e.g. mydoc.txt.meta.json
        if meta_path.exists():
            try:
                meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
            except Exception as e:
                print(f"  ⚠️ Could not read metadata for {txt.name}: {e}")

        docs.append(Document(text=text, metadata=meta))

    # -----------------------------
    # Versioned persist dir bump
    # -----------------------------
    # If PERSIST_DIR is versioned, use it directly rather than auto-bumping.
    base = PERSIST_DIR.parent
    name = PERSIST_DIR.name  # "index_v1"
    m = re.match(r"^(.*)_v(\d+)$", name)
    if m:
        prefix, v = m.group(1), int(m.group(2))
        persist_dir = base / f"{prefix}_v{v}"
    else:
        # Fallback: use PERSIST_DIR as-is
        persist_dir = PERSIST_DIR

    persist_dir.mkdir(parents=True, exist_ok=True)

    # Build index (will use Settings.node_parser)
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=str(persist_dir))
    print("✔ Index persisted to:", persist_dir.resolve())



def main():
    ingest_pdfs()
    ingest_text_files()
    ingest_word_files()
    ingest_urls()
    build_and_persist_index()
    print("Ingestion + indexing complete.")


if __name__ == "__main__":
    main()



"""def build_and_persist_index():
    import json
    from llama_index.core import Document, VectorStoreIndex

    print("Building index from:", CLEAN.resolve())

    txt_files = sorted(CLEAN.glob("*.txt"))
    if not txt_files:
        print("⚠️ No documents found in corpus_clean. Nothing to index.")
        return

    docs = []
    for txt in txt_files:
        text = txt.read_text(encoding="utf-8", errors="ignore")

        meta = {"clean_file": txt.name}
        meta_path = Path(str(txt) + ".meta.json")  # e.g. mydoc.txt.meta.json
        if meta_path.exists():
            try:
                meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
            except Exception as e:
                print(f"  ⚠️ Could not read metadata for {txt.name}: {e}")

        docs.append(Document(text=text, metadata=meta))

    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    print("✔ Index persisted to:", PERSIST_DIR.resolve())
    """




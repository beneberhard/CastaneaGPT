from __future__ import annotations

import os
from pathlib import Path
import re
import requests

import json
from datetime import datetime
from urllib.parse import urlparse

from app.llm_config import embedding_api_key, embedding_model
from app.utils.source_catalogue import (
    catalogue_metadata, validate_source_files, load_catalogue, WEB_SOURCES_PATH,
)

ACTIVE_CLEAN_FILES = set()


def curated_metadata(meta):
    source_path = meta.get("source_path")
    if source_path:
        path = Path(source_path)
        relative = path.resolve().relative_to(RAW.resolve()).as_posix()
        entries = {entry['file']: entry for entry in load_catalogue()}
        if relative not in entries:
            raise ValueError(f'No authoritative metadata for {relative}')
        meta = {**meta, **catalogue_metadata(entries[relative])}
    return meta


def reuse_clean_text(out, source=None, url=None):
    """Refresh metadata even when extraction is cached; exclude stale files."""
    sidecar = out.with_suffix(out.suffix + '.meta.json')
    if not out.exists() or not sidecar.exists() or not out.read_text(encoding='utf-8').strip():
        return False
    meta = json.loads(sidecar.read_text(encoding='utf-8'))
    if url is not None and meta.get('url') != url:
        return False
    if source is not None:
        meta.update(source_path=str(source.resolve()), file_name=source.name)
    write_meta(out, meta)
    return True

def write_meta(out_txt: Path, meta: dict) -> None:
    meta_path = out_txt.with_suffix(out_txt.suffix + ".meta.json")  # .txt.meta.json
    meta = curated_metadata(meta)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    ACTIVE_CLEAN_FILES.add(out_txt)


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
        if reuse_clean_text(out, source=pdf):
            continue

        print(f"Ingesting PDF: {pdf.name}")
        pages = []
        pdf_title = ""
        pdf_author = ""
        ocr_pages = 0
        raster_document = None
        with pdfplumber.open(pdf) as p:
            metadata = p.metadata or {}
            pdf_title = metadata.get("Title") or metadata.get("title") or ""
            pdf_author = metadata.get("Author") or metadata.get("author") or ""
            for page_number, page in enumerate(p.pages):
                txt = page.extract_text() or ""
                if len(txt.strip()) < 40 and page.images:
                    import io
                    import shutil
                    import subprocess
                    import pypdfium2
                    if not shutil.which('tesseract'):
                        raise RuntimeError(f'OCR required for {pdf.name}, but Tesseract is unavailable')
                    if raster_document is None:
                        raster_document = pypdfium2.PdfDocument(str(pdf))
                    raster_page = raster_document[page_number]
                    bitmap = raster_page.render(scale=3)
                    image = bitmap.to_pil()
                    buffer = io.BytesIO()
                    image.save(buffer, format='PNG')
                    result = subprocess.run(
                        ['tesseract', 'stdin', 'stdout', '-l', 'eng+ita+deu'],
                        input=buffer.getvalue(), capture_output=True, check=True, timeout=180,
                    )
                    txt = result.stdout.decode('utf-8').strip()
                    image.close(); bitmap.close(); raster_page.close()
                    ocr_pages += 1
                    print(f'  OCR page {page_number + 1}/{len(p.pages)}', flush=True)
                if txt.strip():
                    pages.append(txt)
        if raster_document is not None:
            raster_document.close()
        if not pages:
            raise ValueError(f'No readable content in {pdf.name}; index was not rebuilt')

        citation_label = choose_citation_label(pdf_title, humanize_filename(pdf.name))
        out.write_text(clean_text("\n".join(pages)), encoding="utf-8")
        write_meta(out, {
            "source_type": "pdf",
            "extraction_method": "pdfplumber+ocr" if ocr_pages else "pdfplumber",
            "ocr_pages": ocr_pages,
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
            if reuse_clean_text(out, source=txt):
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
        if docx.name.startswith('~$'):
            continue
        safe_name = re.sub(r"\W+", "_", str(docx.relative_to(RAW)).strip()).strip("_")
        out = CLEAN / f"docx_{safe_name}.txt"

        meta_out = out.with_suffix(out.suffix + ".meta.json")
        if reuse_clean_text(out, source=docx):
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


def ingest_urls(cache_only=False):
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
        if reuse_clean_text(out, url=url):
            continue

        if out.exists():
            import hashlib
            out = CLEAN / f"url_{slug}_{hashlib.sha256(url.encode()).hexdigest()[:12]}.txt"
            if reuse_clean_text(out, url=url):
                continue
        if cache_only:
            print(f"  No cached extraction for URL; excluded: {url}")
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
    CHUNK_SIZE = 600        # tokens for SentenceSplitter
    CHUNK_OVERLAP = 90      # tokens overlap

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

    txt_files = sorted(ACTIVE_CLEAN_FILES)
    if not txt_files:
        raise ValueError("No active source documents available; existing index was not changed")

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

        # Bibliographic fields must reach chunks without bloating embedding input.
        docs.append(Document(
            text=text, metadata=meta,
            excluded_embed_metadata_keys=list(meta), excluded_llm_metadata_keys=list(meta),
        ))

    expected_ids = {source['source_id'] for source in load_catalogue()}
    actual_ids = {document.metadata.get('source_id') for document in docs if document.metadata.get('source_id')}
    if actual_ids != expected_ids:
        raise ValueError(f'Not all curated sources extracted: missing={sorted(expected_ids-actual_ids)}')

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

    import hashlib
    import shutil
    import tempfile
    from app.utils.source_catalogue import CATALOGUE_PATH
    # Build beside the live index; publish only after the entire rebuild succeeds.
    persist_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=persist_dir.name + '.building-', dir=persist_dir.parent))
    backup = persist_dir.with_name(persist_dir.name + '.backup-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    try:
        nodes = Settings.node_parser.get_nodes_from_documents(docs, show_progress=True)
        nodes = [node for node in nodes if node.get_text().strip()]
        indexed_ids = {node.metadata.get('source_id') for node in nodes if node.metadata.get('source_id')}
        if indexed_ids != expected_ids:
            raise ValueError(f'Curated sources have no readable chunks: {sorted(expected_ids-indexed_ids)}')
        index = VectorStoreIndex(nodes=nodes, show_progress=True)
        index.storage_context.persist(persist_dir=str(staging))
        web_sources = [document.metadata for document in docs if document.metadata.get('source_type') == 'url']
        manifest = {
            'catalogue_sha256': hashlib.sha256(CATALOGUE_PATH.read_bytes()).hexdigest(),
            'curated_source_ids': sorted(actual_ids), 'document_count': len(docs),
            'node_count': len(index.docstore.docs),
            'active_clean_files': [path.name for path in txt_files],
            'rebuilt_at': datetime.now().isoformat(timespec='seconds'),
            'web_sources': web_sources,
        }
        (staging / 'catalogue_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        if persist_dir.exists():
            persist_dir.rename(backup)
        try:
            staging.rename(persist_dir)
        except Exception:
            if backup.exists():
                backup.rename(persist_dir)
            raise
        WEB_SOURCES_PATH.write_text(json.dumps(web_sources, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(f'Index published: {persist_dir.resolve()} ({len(docs)} documents, {manifest["node_count"]} chunks)')
    print(f'Previous index retained at: {backup}')




def main(prepare_only=False, reuse_web_only=False):
    validate_source_files(RAW)
    ACTIVE_CLEAN_FILES.clear()
    ingest_pdfs()
    ingest_text_files()
    ingest_word_files()
    ingest_urls(cache_only=reuse_web_only)
    print(f"Prepared {len(ACTIVE_CLEAN_FILES)} active cleaned documents")
    if prepare_only:
        return
    build_and_persist_index()
    print("Ingestion + indexing complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract active sources, apply curated metadata, rebuild the index")
    parser.add_argument('--prepare-only', action='store_true', help='Refresh extraction/metadata without embedding or rebuilding')
    parser.add_argument('--reuse-web-only', action='store_true', help='Keep cached active URL sources without fetching websites')
    args = parser.parse_args()
    main(prepare_only=args.prepare_only, reuse_web_only=args.reuse_web_only)



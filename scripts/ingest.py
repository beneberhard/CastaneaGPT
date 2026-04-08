from __future__ import annotations

from pathlib import Path
import re
import requests

import json
from datetime import datetime
from urllib.parse import urlparse

def write_meta(out_txt: Path, meta: dict) -> None:
    meta_path = out_txt.with_suffix(out_txt.suffix + ".meta.json")  # .txt.meta.json
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


RAW = Path("data/corpus_raw")
CLEAN = Path("data/corpus_clean")
CLEAN.mkdir(parents=True, exist_ok=True)

# >>> ADDED: where to persist the LlamaIndex index
PERSIST_DIR = Path("storage/index_v1")
PERSIST_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


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
        with pdfplumber.open(pdf) as p:
            for page in p.pages:
                txt = page.extract_text()
                if txt:
                    pages.append(txt)

        out.write_text(clean_text("\n".join(pages)), encoding="utf-8")
        write_meta(out, {
            "source_type": "pdf",
            "file_name": pdf.name,
            "source_path": str(pdf.resolve()),
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

        doc = Document(response.text)
        html = doc.summary(html_partial=True)

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ").strip()

        if len(text) < 200:
            print("  ⚠️ extracted text too short, skipping")
            continue

        out.write_text(clean_text(text), encoding="utf-8")
        write_meta(out, {
            "source_type": "url",
            "url": url,
            "domain": urlparse(url).netloc,
            "ingested_at": datetime.now().isoformat(timespec="seconds"),
            })


# >>> ADDED: build + persist LlamaIndex index from CLEAN/*.txt
# >>> UPDATED: build + persist LlamaIndex index from CLEAN/*.txt + sidecar metadata (*.txt.meta.json)
def build_and_persist_index():
    import json
    import re
    from llama_index.core import Document, VectorStoreIndex, Settings
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
    # If PERSIST_DIR is storage/index_v1, automatically write to the next free version
    base = PERSIST_DIR.parent
    name = PERSIST_DIR.name  # "index_v1"
    m = re.match(r"^(.*)_v(\d+)$", name)
    if m:
        prefix, v = m.group(1), int(m.group(2))
        persist_dir = base / f"{prefix}_v{v}"
        while persist_dir.exists() and any(persist_dir.iterdir()):
            v += 1
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




import csv
import re
from functools import lru_cache
from pathlib import Path

from app.utils.language import detect_language

GLOSSARY_CSV_PATH = Path("data/glossary/glossary_100_terms.csv")
SUPPORTED_GLOSSARY_LANGUAGES = {"it", "en", "de"}


def normalize_language(language: str, fallback_text: str = "") -> str:
    """Map UI/detected language values to glossary-supported codes."""
    if language and language != "auto":
        normalized = language.strip().lower()
        if normalized in SUPPORTED_GLOSSARY_LANGUAGES:
            return normalized

    detected = detect_language(fallback_text or "")
    if detected.startswith("it"):
        return "it"
    if detected.startswith("de"):
        return "de"
    return "en"


@lru_cache(maxsize=1)
def load_glossary() -> list[dict[str, str]]:
    """Load glossary rows once and normalize whitespace."""
    rows: list[dict[str, str]] = []

    if not GLOSSARY_CSV_PATH.exists():
        return rows

    with GLOSSARY_CSV_PATH.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cleaned = {key: (value or "").strip() for key, value in row.items()}
            if cleaned.get("term_it") or cleaned.get("term_en") or cleaned.get("term_de"):
                rows.append(cleaned)

    return rows


def _term_pattern(term: str) -> re.Pattern[str] | None:
    term = (term or "").strip()
    if not term:
        return None
    return re.compile(rf"(?<!\w){re.escape(term.lower())}(?!\w)")


def find_glossary_matches(texts: list[str], target_language: str, limit: int = 12) -> list[dict[str, str]]:
    """Return glossary rows whose terms appear in the retrieved context."""
    if target_language not in SUPPORTED_GLOSSARY_LANGUAGES:
        target_language = "en"

    haystack = "\n".join(texts).lower()
    matches: list[dict[str, str]] = []
    seen_terms: set[str] = set()

    for row in load_glossary():
        matched_source_term = ""
        for field in ("term_it", "term_en", "term_de"):
            candidate = row.get(field, "")
            pattern = _term_pattern(candidate)
            if pattern and pattern.search(haystack):
                matched_source_term = candidate
                break

        if not matched_source_term:
            continue

        target_term = row.get(f"term_{target_language}", "")
        if not target_term:
            continue

        dedupe_key = target_term.lower()
        if dedupe_key in seen_terms:
            continue
        seen_terms.add(dedupe_key)

        matches.append(
            {
                "domain": row.get("domain", ""),
                "matched_source_term": matched_source_term,
                "term_it": row.get("term_it", ""),
                "term_en": row.get("term_en", ""),
                "term_de": row.get("term_de", ""),
                "target_language": target_language,
                "target_term": target_term,
                "notes": row.get("notes", ""),
            }
        )

        if len(matches) >= limit:
            break

    return matches


def build_glossary_context_block(matches: list[dict[str, str]], target_language: str) -> str:
    """Build a compact glossary instruction block for the final answer prompt."""
    if not matches:
        return ""

    language_names = {"it": "Italian", "en": "English", "de": "German"}
    target_name = language_names.get(target_language, target_language)
    lines = [
        "Glossary guidance:",
        f"- The final answer should use the {target_name} glossary translation for matched technical terms when relevant.",
        "- Prefer the glossary term over an improvised translation.",
        "- You may keep the original source-language term in parentheses when it improves clarity.",
        "- Do not invent glossary mappings for terms not listed below.",
        "Matched glossary terms:",
    ]

    for match in matches:
        base = (
            f"- {match['matched_source_term']} -> {match['target_term']} "
            f"(it: {match['term_it']}; en: {match['term_en']}; de: {match['term_de']})"
        )
        if match["notes"]:
            base += f" | notes: {match['notes']}"
        lines.append(base)

    return "\n".join(lines)

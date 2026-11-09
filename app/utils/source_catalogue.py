"""Authoritative bibliographic records and deterministic citation formatting."""
from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlparse

CATALOGUE_PATH = Path(__file__).resolve().parents[2] / 'data' / 'source_catalogue.json'
WEB_SOURCES_PATH = CATALOGUE_PATH.with_name('web_sources.json')


def sort_key(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', value) if not unicodedata.combining(c)).casefold()


def year_label(source):
    year = source.get('year')
    return f"{year}{source.get('year_suffix', '')}" if year else 'n.d.'


def inline_citation(source):
    if source.get('first_author'):
        return f"({source['first_author']}, {year_label(source)})"
    return f"({source.get('domain') or source.get('label') or 'Unknown source'}, n.d.)"


def reference_entry(source):
    if not source.get('first_author'):
        return source.get('label') or source.get('title') or 'Unknown source'
    author = source['first_author'] + (' et al.' if source.get('multiple_authors') else '')
    return f"{author} — {source['title']} — {year_label(source)}"


def safe_url(value):
    return value if value and urlparse(value).scheme in {'https', 'http'} and urlparse(value).netloc else None


@lru_cache(maxsize=1)
def load_catalogue():
    document = json.loads(CATALOGUE_PATH.read_text(encoding='utf-8'))
    sources = document['sources']
    ids, files = set(), set()
    for source in sources:
        sid, file = source['source_id'], source['file']
        if not isinstance(sid, int) or sid <= 0 or sid in ids or file in files:
            raise ValueError(f'Duplicate or invalid catalogue mapping: {sid}: {file}')
        if Path(file).is_absolute() or '..' in Path(file).parts:
            raise ValueError(f'Invalid source file path: {file}')
        if not source['first_author'].strip() or not source['title'].strip():
            raise ValueError(f'Incomplete catalogue metadata for source {sid}')
        if source.get('year') is not None and not isinstance(source['year'], int):
            raise ValueError(f'Invalid publication year for source {sid}')
        if source.get('year_suffix', '') and not re.fullmatch('[a-z]+', source['year_suffix']):
            raise ValueError(f'Invalid year suffix for source {sid}')
        ids.add(sid); files.add(file)
    return sources


def validate_source_files(raw):
    catalogue = load_catalogue()
    expected = {entry['file'] for entry in catalogue}
    actual = {path.relative_to(raw).as_posix() for path in raw.rglob('*')
              if path.is_file() and not path.name.startswith('~$')
              and path.suffix.lower() in {'.pdf', '.docx', '.txt'} and path.name != 'urls.txt'}
    if expected != actual:
        raise ValueError(f'Catalogue/file mismatch: missing files={sorted(expected-actual)}, '
                         f'files without metadata={sorted(actual-expected)}')
    for source in catalogue:
        match = re.match(r'^(\d+)(?:_|\.|\s)', Path(source['file']).name)
        if not match or int(match[1]) != source['source_id']:
            raise ValueError(f"Source number does not match file: {source['file']}")
    return {entry['file']: entry for entry in catalogue}


def catalogue_metadata(source):
    keys = ('source_id', 'first_author', 'authors_text', 'title', 'year', 'year_suffix',
            'multiple_authors', 'full_reference', 'doi', 'url')
    result = {key: source.get(key) for key in keys}
    result.update(author=source['authors_text'], citation_label=reference_entry(source),
                  inline_citation=inline_citation(source), catalogue_version=1)
    return result


def public_sources():
    return [{**source, 'inline_citation': inline_citation(source),
             'reference_entry': reference_entry(source), 'url': safe_url(source.get('url'))}
            for source in sorted(load_catalogue(), key=lambda s: (sort_key(s['first_author']), sort_key(s['title'])))]


SOURCE_TOKEN = re.compile(r'\[Source\s+(\d+)\]')


def format_answer_citations(answer, sources, enabled=True):
    """Resolve only model-selected, known IDs. Never attach guessed citations."""
    by_id = {source['id']: source for source in sources}
    used = {}
    def replace(match):
        source = by_id.get(int(match[1]))
        if source is None:
            return '[unverified source]' if enabled else ''
        if enabled:
            used[source.get('source_key') or source.get('source_id') or source['id']] = source
            return source.get('inline_citation') or inline_citation(source)
        return ''
    text = SOURCE_TOKEN.sub(replace, answer)
    cited = sorted(used.values(), key=lambda s: (sort_key(s.get('first_author') or s.get('label') or ''), sort_key(s.get('title') or '')))
    if enabled and cited:
        text += '\n\nSource reference list:\n' + '\n'.join(
            (source.get('reference_entry') or reference_entry(source))
            + (f"\n{safe_url(source.get('url'))}" if safe_url(source.get('url')) else '')
            for source in cited)
    return text, cited


class CitationStreamFormatter:
    """Hold partial source tokens across network chunks, then resolve them."""
    def __init__(self, sources, enabled=True):
        self.sources = sources
        self.enabled = enabled
        self.pending = ''

    def feed(self, text, final=False):
        self.pending += text
        cut = len(self.pending)
        if not final:
            start = self.pending.rfind('[')
            if start >= 0:
                tail = self.pending[start:]
                if '[Source'.startswith(tail) or re.fullmatch(r'\[Source\s*\d*', tail):
                    cut = start
        ready, self.pending = self.pending[:cut], self.pending[cut:]
        by_id = {s['id']: s for s in self.sources}
        def replace(match):
            if not self.enabled:
                return ''
            source = by_id.get(int(match[1]))
            return (source.get('inline_citation') or inline_citation(source)) if source else '[unverified source]'
        return SOURCE_TOKEN.sub(replace, ready)

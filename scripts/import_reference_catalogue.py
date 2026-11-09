"""Import the user-curated numbered Word bibliography; no network requests."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from xml.etree import ElementTree as ET
from zipfile import ZipFile


def sort_text(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', value) if not unicodedata.combining(c)).casefold()


def import_catalogue(docx: Path, raw: Path):
    with ZipFile(docx) as archive:
        root = ET.fromstring(archive.read('word/document.xml'))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    lines = [''.join(t.text or '' for t in p.findall('.//w:t', ns)).strip()
             for p in root.findall('.//w:p', ns)]
    files = defaultdict(list)
    for path in raw.rglob('*'):
        if path.is_file() and not path.name.startswith('~$') and path.suffix.lower() in {'.pdf', '.docx', '.txt'}:
            match = re.match(r'^(\d+)(?:_|\.|\s)', path.name)
            if match:
                files[int(match[1])].append(path.relative_to(raw).as_posix())
    entries = []
    for line in lines:
        match = re.match(r'^(\d+)_\s*(.+)$', line)
        if not match:
            if line:
                raise ValueError(f'Unnumbered bibliography paragraph: {line[:80]}')
            continue
        number, reference = int(match[1]), match[2]
        if len(files[number]) != 1:
            raise ValueError(f'Source {number} must match exactly one file: {files[number]}')
        if number in {33, 62}:
            title_start = 'Castagneti da frutto in Piemonte' if number == 33 else 'Valorizzazione dei soprasuoli'
            authors, _, body = reference.partition(title_start)
            authors = authors.rstrip('. ')
            body = title_start + body
        elif number == 15:
            authors, body = reference.split(' 2022. ', 1)
            authors = authors.rstrip('. ')
        else:
            authors, body = reference.split('.. ', 1)
        year_match = re.search(r',\s*(\d{4})\.', body)
        if number == 12:
            year_match = re.search(r', updated November (\d{4})\.', body)
        year = int(year_match[1]) if year_match else None
        title = body[:year_match.start()].strip() if year_match else body.rstrip('.')
        if number == 16:
            title = title.split('. EDISTAMPA')[0]
        if number == 28:
            title = 'How future-proof is Sweet chestnut (Castanea sativa) in a global change context?'
        first_author = authors.split(',', 1)[0].strip()
        if number == 33:
            first_author = 'IPLA S.p.A.'
        elif number in {16, 25}:
            first_author = 'Regione Campania' if number == 16 else 'Giunta Regionale della Campania'
        # Personal coauthors occur as surname, initials; do not count middle initials as authors.
        multiple = bool(re.search(r'\.\s*,\s*(?:and\s+)?[A-ZÀ-ÖØ-Þ][^,]+,\s*[A-ZÜÁ]', authors)) or ', and ' in authors
        if number in {33, 16, 25}:
            multiple = False
        url_match = re.search(r'https?://\S+', reference)
        url = url_match[0].rstrip('.') if url_match else None
        entries.append({
            'source_id': number, 'file': files[number][0], 'authors_text': authors,
            'first_author': first_author, 'multiple_authors': multiple,
            'title': title, 'year': year, 'year_suffix': '', 'url': url,
            'doi': url.removeprefix('https://doi.org/') if url and 'doi.org/' in url else None,
            'full_reference': reference,
        })
    ids = [entry['source_id'] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate source IDs in bibliography')
    if set(files) != set(ids):
        raise ValueError(f'Unmapped numbered files: {sorted(set(files) - set(ids))}')
    groups = defaultdict(list)
    for entry in entries:
        if entry['year']:
            groups[(sort_text(entry['first_author']), entry['year'])].append(entry)
    for group in groups.values():
        if len(group) > 1:
            for index, entry in enumerate(sorted(group, key=lambda e: (sort_text(e['title']), e['source_id']))):
                # This catalogue has at most two same-author/same-year publications.
                entry['year_suffix'] = chr(ord('a') + index)
    return {
        'schema_version': 1, 'provenance': {'file': docx.name, 'sha256': hashlib.sha256(docx.read_bytes()).hexdigest()},
        'reserved_source_ids': sorted(set(range(1, max(74, max(ids)) + 1)) - set(ids)),
        'sources': sorted(entries, key=lambda e: e['source_id']),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docx', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=Path('data/corpus_raw'))
    parser.add_argument('--output', type=Path, default=Path('data/source_catalogue.json'))
    args = parser.parse_args()
    catalogue = import_catalogue(args.docx, args.raw)
    args.output.write_text(json.dumps(catalogue, ensure_ascii=False, indent=2) + '\n')
    print(f"Imported and matched {len(catalogue['sources'])} sources -> {args.output}")
    for entry in catalogue['sources']:
        print(entry['source_id'], entry['first_author'], str(entry['year']) + entry['year_suffix'], entry['title'])

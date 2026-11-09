"""Authoritative bibliography mappings, citation integrity and streaming behavior."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from app.api.api import app
from app.rag.rag import extract_sources, format_retrieved_context
from app.utils.source_catalogue import (
    CitationStreamFormatter, catalogue_metadata, format_answer_citations,
    inline_citation, load_catalogue, public_sources, reference_entry, validate_source_files,
)


class CatalogueTests(unittest.TestCase):
    def test_imported_catalogue_preserves_numbers_and_disambiguates_years(self):
        sources = {s['source_id']: s for s in load_catalogue()}
        self.assertEqual(set(sources), set(range(1, 75)) - {7, 8, 9, 50, 57})
        self.assertEqual(sources[1]['first_author'], 'Mariotti')
        self.assertEqual(sources[1]['file'], 'pdf/1.pdf')
        self.assertEqual(inline_citation(sources[22]), '(Conedera, 2004a)')
        self.assertEqual(inline_citation(sources[3]), '(Conedera, 2004b)')
        self.assertEqual(inline_citation(sources[15]), '(Castellotti, 2022a)')
        self.assertEqual(inline_citation(sources[32]), '(Castellotti, 2022b)')
        self.assertEqual(inline_citation(sources[16]), '(Regione Campania, n.d.)')
        self.assertEqual(inline_citation(sources[26]), '(Spinelli, n.d.)')

    def test_catalogue_rejects_unmapped_or_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'missing files'):
                validate_source_files(Path(tmp))

    def test_library_endpoint_is_alphabetized_and_all_records_have_original_reference(self):
        result = TestClient(app).get('/sources')
        self.assertEqual(result.status_code, 200)
        sources = result.json()['sources']
        self.assertEqual(sources, public_sources())
        self.assertEqual(sources[0]['first_author'], 'Agnoletti')
        self.assertTrue(all(s['full_reference'] and s['reference_entry'] for s in sources))
        page = TestClient(app).get('/map').text
        self.assertIn('id="openSourceLibrary"', page)
        self.assertIn('id="sourceLibrary"', page)
        self.assertIn("fetch('/sources')", page)


class CitationTests(unittest.TestCase):
    def setUp(self):
        records = {s['source_id']: s for s in load_catalogue()}
        self.sources = [dict(catalogue_metadata(records[n]), id=i, source_key=f'catalogue:{n}')
                        for i, n in enumerate([1, 22, 3], 1)]

    def test_only_cited_publications_are_listed_once(self):
        answer, used = format_answer_citations('A [Source 1]. B [Source 1]. C [Source 3].', self.sources)
        self.assertIn('(Mariotti, 2009)', answer)
        self.assertIn('(Conedera, 2004b)', answer)
        self.assertNotIn('2004a', answer)
        self.assertEqual(len(used), 2)
        self.assertEqual(answer.count(reference_entry(self.sources[0])), 1)
        self.assertEqual(used[0]['first_author'], 'Conedera')

    def test_uncited_answer_does_not_receive_fabricated_references(self):
        self.assertEqual(format_answer_citations('No supporting evidence.', self.sources), ('No supporting evidence.', []))

    def test_unknown_source_is_not_turned_into_a_real_reference(self):
        answer, used = format_answer_citations('Claim [Source 999].', self.sources)
        self.assertIn('[unverified source]', answer)
        self.assertEqual(used, [])

    def test_none_removes_citations_and_references(self):
        answer, used = format_answer_citations('Claim [Source 1].', self.sources, enabled=False)
        self.assertEqual(answer, 'Claim .')
        self.assertEqual(used, [])

    def test_stream_resolves_citation_split_at_every_character(self):
        formatter = CitationStreamFormatter(self.sources)
        text = 'A [Source 2], B [Source 3].'
        result = ''.join(formatter.feed(char) for char in text) + formatter.feed('', final=True)
        self.assertEqual(result, 'A (Conedera, 2004a), B (Conedera, 2004b).')

    def test_two_chunks_of_one_publication_share_an_id(self):
        def node(source, text):
            return SimpleNamespace(node=SimpleNamespace(metadata=source, get_text=lambda: text), score=1)
        nodes = [node(self.sources[0], 'first chunk'), node(self.sources[0], 'second chunk'), node(self.sources[1], 'third chunk')]
        context = format_retrieved_context(nodes)
        self.assertEqual(context.count('[Source 1:'), 2)
        self.assertIn('[Source 2:', context)
        self.assertNotIn('[Source 3:', context)
        sources = extract_sources(nodes)
        self.assertEqual([s['source_id'] for s in sources], [1, 22])
        self.assertEqual(len(sources), 2)

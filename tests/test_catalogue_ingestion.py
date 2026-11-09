"""Ingestion refreshes curated metadata and excludes orphaned extractions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from scripts import ingest
from app.utils.source_catalogue import load_catalogue


class CatalogueIngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw, self.clean = self.root / 'raw', self.root / 'clean'
        (self.raw / 'pdf').mkdir(parents=True)
        self.clean.mkdir()
        self.source = (self.raw / 'pdf' / '1.pdf')
        self.source.write_bytes(b'not opened because cleaned extraction exists')
        self.record = dict(load_catalogue()[0])
        for target, value in [('RAW', self.raw), ('CLEAN', self.clean), ('ACTIVE_CLEAN_FILES', set())]:
            patcher = patch.object(ingest, target, value)
            patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch.object(ingest, 'load_catalogue', return_value=[self.record])
        patcher.start(); self.addCleanup(patcher.stop)

    def test_cached_text_gets_authoritative_metadata_without_reextraction(self):
        out = self.clean / '1.txt'
        out.write_text('Previously extracted document text.')
        sidecar = out.with_suffix('.txt.meta.json')
        sidecar.write_text(json.dumps({'title': 'Wrong embedded title', 'author': 'Wrong author', 'source_path': '/old/path'}))
        self.assertTrue(ingest.reuse_clean_text(out, source=self.source))
        result = json.loads(sidecar.read_text())
        self.assertEqual(result['source_id'], 1)
        self.assertEqual(result['title'], self.record['title'])
        self.assertEqual(result['author'], self.record['authors_text'])
        self.assertEqual(out.read_text(), 'Previously extracted document text.')
        self.assertEqual(ingest.ACTIVE_CLEAN_FILES, {out})

    def test_rebuild_excludes_stale_text_and_preserves_old_index(self):
        out = self.clean / '1.txt'
        out.write_text('Chestnut orchard management evidence. ' * 20)
        ingest.write_meta(out, {'source_path': str(self.source), 'source_type': 'pdf', 'file_name': '1.pdf'})
        (self.clean / 'stale.txt').write_text('An old deleted source should not be indexed.')
        live = self.root / 'index_v2'
        live.mkdir()
        (live / 'previous-marker').write_text('previous index')
        original_embed_model = Settings._embed_model
        self.addCleanup(setattr, Settings, '_embed_model', original_embed_model)
        with patch.object(ingest, 'PERSIST_DIR', live), \
             patch.object(ingest, 'WEB_SOURCES_PATH', self.root / 'web.json'), \
             patch('llama_index.embeddings.openai.OpenAIEmbedding', return_value=MockEmbedding(embed_dim=8)):
            ingest.build_and_persist_index()
        manifest = json.loads((live / 'catalogue_manifest.json').read_text())
        self.assertEqual(manifest['curated_source_ids'], [1])
        self.assertEqual(manifest['active_clean_files'], ['1.txt'])
        self.assertEqual(manifest['document_count'], 1)
        backups = list(self.root.glob('index_v2.backup-*'))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / 'previous-marker').read_text(), 'previous index')
        stored = json.loads((live / 'docstore.json').read_text())['docstore/data']
        chunks = [entry['__data__'] for entry in stored.values() if entry['__type__'] == '1']
        self.assertTrue(chunks)
        for node in chunks:
            self.assertEqual(node['metadata']['source_id'], 1)
            self.assertEqual(node['metadata']['title'], self.record['title'])
            self.assertIn('full_reference', node['excluded_embed_metadata_keys'])

    def test_empty_cached_extraction_is_not_reused(self):
        out = self.clean / '1.txt'
        out.write_text('   ')
        out.with_suffix('.txt.meta.json').write_text('{}')
        self.assertFalse(ingest.reuse_clean_text(out, source=self.source))
        self.assertEqual(ingest.ACTIVE_CLEAN_FILES, set())

    def test_source_with_no_chunks_cannot_replace_existing_index(self):
        out = self.clean / '1.txt'
        out.write_text('')
        ingest.write_meta(out, {'source_path': str(self.source), 'source_type': 'pdf'})
        live = self.root / 'index_v2'
        live.mkdir()
        (live / 'marker').write_text('keep')
        original_embed_model = Settings._embed_model
        self.addCleanup(setattr, Settings, '_embed_model', original_embed_model)
        with patch.object(ingest, 'PERSIST_DIR', live), \
             patch('llama_index.embeddings.openai.OpenAIEmbedding', return_value=MockEmbedding(embed_dim=8)):
            with self.assertRaisesRegex(ValueError, 'no readable chunks'):
                ingest.build_and_persist_index()
        self.assertEqual((live / 'marker').read_text(), 'keep')

    def test_empty_rebuild_keeps_existing_index(self):
        live = self.root / 'index_v2'
        live.mkdir()
        (live / 'marker').write_text('keep')
        original_embed_model = Settings._embed_model
        self.addCleanup(setattr, Settings, '_embed_model', original_embed_model)
        with patch.object(ingest, 'PERSIST_DIR', live), \
             patch('llama_index.embeddings.openai.OpenAIEmbedding', return_value=MockEmbedding(embed_dim=8)):
            with self.assertRaisesRegex(ValueError, 'No active source'):
                ingest.build_and_persist_index()
        self.assertEqual((live / 'marker').read_text(), 'keep')

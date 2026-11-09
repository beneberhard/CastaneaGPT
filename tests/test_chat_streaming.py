"""Streaming contract checks without external model or database calls."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from app.api.api import app, session_store
from app.rag.rag import stream_query_rag


class ChatStreamingTests(unittest.TestCase):
    @patch('app.api.api.stream_query_rag')
    def test_deltas_final_citations_and_memory(self, rag):
        rag.return_value = iter([
            {'type': 'delta', 'text': 'Chestnut '},
            {'type': 'delta', 'text': 'trees. [Source 1]\nÈ vero.'},
            {'type': 'sources', 'sources': [
                {'id': 1, 'label': 'Guide', 'source_key': 'guide', 'score': 1.0, 'first_author': 'Mariotti', 'title': 'Guide', 'year': 2009, 'inline_citation': '(Mariotti, 2009)'}]},
        ])
        response = TestClient(app).post('/chat?stream=true', json={'query': 'Chestnut?'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('application/x-ndjson', response.headers['content-type'])
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual([e['type'] for e in events], ['start', 'delta', 'delta', 'done'])
        result = events[-1]
        self.assertIn('(Mariotti, 2009)', result['answer'])
        self.assertNotIn('[Source 1]', result['answer'])
        self.assertIn('Source reference list:', result['answer'])
        self.assertEqual(result['sources'][0]['label'], 'Guide')
        sid = result['session_id']
        self.assertEqual(session_store.get(sid).turns[-1], ('assistant', result['answer']))
        # The returned session ID can be reused on the next request.
        with patch('app.api.api.query_rag', return_value={'answer': 'Follow-up', 'sources': []}) as query:
            TestClient(app).post('/chat', json={'query': 'More?', 'session_id': sid})
            self.assertIn('Chestnut?', query.call_args.args[0])
        session_store.clear(sid)

    @patch('app.api.api.stream_query_rag')
    def test_failure_emits_error_without_saving_partial_answer(self, rag):
        def broken():
            yield {'type': 'delta', 'text': 'Partial'}
            raise RuntimeError('private provider detail')
        rag.return_value = broken()
        with self.assertLogs('app.api.api', level='ERROR'):
            response = TestClient(app).post('/chat?stream=true', json={'query': 'Chestnut?'})
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual([e['type'] for e in events], ['start', 'delta', 'error'])
        self.assertNotIn('private provider detail', response.text)
        sid = events[0]['session_id']
        self.assertEqual(session_store.get(sid).turns, [])
        session_store.clear(sid)

    @patch('app.rag.rag.get_llm')
    @patch('app.rag.rag._prepare_rag', return_value=('prompt', []))
    def test_rag_stream_yields_deltas_without_repeating_cumulative_text(self, prepare, llm):
        llm.return_value.stream_complete.return_value = iter([
            SimpleNamespace(delta='Hello', text='Hello'),
            SimpleNamespace(delta='', text='Hello'),
            SimpleNamespace(delta=' world', text='Hello world'),
        ])
        self.assertEqual(list(stream_query_rag('question')), [
            {'type': 'sources', 'sources': []},
            {'type': 'delta', 'text': 'Hello'},
            {'type': 'delta', 'text': ' world'},
        ])
        llm.return_value.complete.assert_not_called()

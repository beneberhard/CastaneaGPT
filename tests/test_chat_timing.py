"""Latency diagnostics stay isolated per request and across stream resumes."""
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from llama_index.core.callbacks import CBEventType
from app.api.api import app
from app.rag.rag import _TimingHandler
from app.utils.timing import ChatTiming, record_detail, record_seconds


class ChatTimingTests(unittest.TestCase):
    def test_nested_measurements_are_isolated(self):
        outer, inner = ChatTiming(), ChatTiming()
        with outer.activate():
            record_seconds('stage', 1)
            with inner.activate():
                record_seconds('stage', 2)
            record_seconds('stage', 3)
        record_seconds('stage', 100)
        self.assertEqual(outer.seconds, {'stage': 4})
        self.assertEqual(inner.seconds, {'stage': 2})

    def test_callback_records_embedding_duration_without_payload(self):
        trace = ChatTiming()
        handler = _TimingHandler(CBEventType.EMBEDDING, 'embedding_request')
        with trace.activate(), patch('app.rag.rag.perf_counter', side_effect=[10, 12]):
            handler.on_event_start(CBEventType.EMBEDDING, event_id='test')
            handler.on_event_end(CBEventType.EMBEDDING, event_id='test')
        self.assertEqual(trace.seconds['embedding_request'], 2)
        self.assertEqual(handler.starts, {})

    @patch('app.api.api.stream_query_rag')
    def test_stream_logs_timings_after_multiple_context_resumes(self, rag):
        def output(*args, **kwargs):
            record_seconds('retrieval_total', 2)
            yield {'type': 'delta', 'text': 'Hello'}
            record_detail('stream_text_chunks', 1)
            yield {'type': 'sources', 'sources': []}
        rag.side_effect = output
        with self.assertLogs('uvicorn.error', level='INFO') as logs:
            response = TestClient(app).post('/chat?stream=true', json={
                'query': 'private question', 'use_session_memory': False})
        self.assertEqual(response.status_code, 200)
        report = json.loads(logs.output[-1].split('CHAT_TIMING ', 1)[1])
        self.assertEqual(report['seconds']['retrieval_total'], 2)
        self.assertEqual(report['stream_text_chunks'], 1)
        self.assertEqual(report['status'], 'complete')
        self.assertNotIn('private question', logs.output[-1])

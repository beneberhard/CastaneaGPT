"""Startup warms chat dependencies before serving requests, without model calls."""
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from app.api.api import app


class ApiStartupTests(unittest.TestCase):
    @patch('app.api.api.get_hybrid_retriever')
    def test_retriever_is_ready_before_requests_and_initialized_once(self, retriever):
        with self.assertLogs('uvicorn.error', level='INFO') as logs:
            with TestClient(app) as client:
                retriever.assert_called_once_with(expand_queries=False)
                self.assertEqual(client.get('/health').json(), {'status': 'ok'})
                self.assertEqual(client.get('/health').status_code, 200)
                retriever.assert_called_once_with(expand_queries=False)
        self.assertIn('Chat index and retriever ready in', '\n'.join(logs.output))

    @patch('app.api.api.get_hybrid_retriever', side_effect=RuntimeError('Index unavailable'))
    def test_initialization_failure_aborts_startup(self, retriever):
        with self.assertLogs('uvicorn.error', level='ERROR') as logs:
            with self.assertRaisesRegex(RuntimeError, 'Index unavailable'):
                with TestClient(app):
                    self.fail('Startup should not accept requests after initialization failure')
        retriever.assert_called_once_with(expand_queries=False)
        self.assertIn('API startup aborted', '\n'.join(logs.output))

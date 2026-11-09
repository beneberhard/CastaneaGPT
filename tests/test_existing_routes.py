"""Verify map integration and the existing chat contract without model calls."""
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from app.api.api import app


class ExistingRouteTests(unittest.TestCase):
    def test_map_has_existing_controls_and_new_overlay(self):
        response = TestClient(app).get('/map')
        self.assertEqual(response.status_code, 200)
        for marker in ('id="chatInput"', 'id="quiz-start"', 'id="imageAssessment"',
                       'id="openAssessment"', 'id="iaSpecies"', 'Assess tree damage', 'value="whole_tree"', 'Context is optional'):
            self.assertIn(marker, response.text)

    @patch('app.api.api.query_rag')
    def test_text_chat_still_returns_answer_and_sources(self, rag):
        rag.return_value = {'answer': 'Mock chestnut answer.', 'sources': []}
        response = TestClient(app).post('/chat', json={
            'query': 'Tell me about chestnut trees.', 'use_session_memory': False})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['answer'], 'Mock chestnut answer.')
        self.assertEqual(response.json()['sources'], [])
        rag.assert_called_once()

"""Offline adapter tests: python -m unittest discover -s tests -v."""

import os
import unittest
from unittest.mock import MagicMock, patch

import requests

from app.identification.plantnet import (
    DISEASE_ENDPOINT, TIMEOUT, identify_images, normalize_response,
)


class PlantNetTests(unittest.TestCase):
    def test_documented_response_is_ranked_and_normalized(self):
        result = normalize_response({
            "version": "mock-version",
            "results": [
                {"name": "ELSIAM", "score": 0.03,
                 "description": "Elsinoë ampelina - Anthracnose"},
                {"name": "APHISP", "score": 0.91, "description": "Aphis sp."},
            ],
        })
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.candidates[0].eppo_code, "APHISP")
        self.assertEqual(result.candidates[0].confidence, 0.91)
        self.assertIsNone(result.candidates[1].scientific_name)
        self.assertEqual(result.model_version, "mock-version")

    def test_uncertain_empty_and_malformed_results(self):
        for score in (None, 0.1):
            result = normalize_response({"results": [{"name": "APHISP", "score": score}]})
            self.assertEqual(result.status, "low_confidence")
        self.assertEqual(normalize_response({"results": []}).status, "no_result")
        for payload in ({}, [], {"results": [{"name": "APHISP", "score": 2}]},
                        {"results": [{"name": "APHISP", "score": float("nan")}]},
                        {"results": [{"name": "APHISP", "score": True}]}):
            self.assertEqual(normalize_response(payload).status, "api_error")

    @patch.dict(os.environ, {"PLANTNET_API_KEY": "mock-secret"})
    @patch("app.identification.plantnet.requests.post")
    def test_request_contract(self, post):
        response = MagicMock(status_code=200)
        response.json.return_value = {"results": []}
        post.return_value.__enter__.return_value = response
        result = identify_images([(b"validated-jpeg", "image/jpeg"),
                                  (b"validated-png", "image/png")], organ="branch")
        self.assertEqual(result.status, "no_result")
        args, kwargs = post.call_args
        self.assertEqual(args, (DISEASE_ENDPOINT,))
        self.assertEqual(kwargs["data"], [("organs", "auto"), ("organs", "auto")])
        self.assertEqual([part[0] for part in kwargs["files"]], ["images", "images"])
        self.assertEqual(kwargs["params"]["api-key"], "mock-secret")
        self.assertEqual(kwargs["timeout"], TIMEOUT)
        self.assertFalse(kwargs["allow_redirects"])
        self.assertNotIn("mock-secret", result.model_dump_json())

    @patch.dict(os.environ, {"PLANTNET_API_KEY": ""})
    @patch("app.identification.plantnet.requests.post")
    def test_missing_key_does_not_call_service(self, post):
        self.assertEqual(identify_images([(b"image", "image/jpeg")]).status, "not_configured")
        post.assert_not_called()

    @patch.dict(os.environ, {"PLANTNET_API_KEY": "mock-secret"})
    @patch("app.identification.plantnet.requests.post")
    def test_failures_do_not_expose_secrets(self, post):
        for exception, status in (
            (requests.Timeout("URL containing mock-secret"), "timeout"),
            (requests.ConnectionError("URL containing mock-secret"), "api_error"),
        ):
            post.side_effect = exception
            result = identify_images([(b"image", "image/jpeg")])
            self.assertEqual(result.status, status)
            self.assertNotIn("mock-secret", result.model_dump_json())
        post.side_effect = None
        for code in (400, 401, 403, 404, 413, 422, 429, 500, 302):
            post.return_value.__enter__.return_value = MagicMock(status_code=code)
            result = identify_images([(b"image", "image/jpeg")])
            self.assertEqual(result.status, "api_error")
        response = MagicMock(status_code=200)
        response.json.side_effect = ValueError("mock-secret")
        post.return_value.__enter__.return_value = response
        result = identify_images([(b"image", "image/jpeg")])
        self.assertEqual(result.status, "api_error")
        self.assertNotIn("mock-secret", result.model_dump_json())


if __name__ == "__main__":
    unittest.main()

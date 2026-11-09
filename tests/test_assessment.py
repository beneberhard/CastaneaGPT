"""Offline tests of narrative assessment and useful failure responses."""
import json
import unittest
from unittest.mock import MagicMock, patch

from app.identification.assessment import (
    AssessmentContext, assess_identification, build_retrieval_query, build_assessment_prompt, remove_inline_source_markers,
)
from app.identification.plantnet import normalize_response


class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.identification = normalize_response({"results": [
            {"name": "ENDOPA", "description": "Cryphonectria parasitica - Blight of chestnut", "score": 0.28387},
            {"name": "ERWITR", "description": "Erwinia tracheiphila - Bacterial wilt of cucurbits", "score": 0.05529},
        ]})
        self.context = AssessmentContext(organ="bark", location="Ticino",
                                         observation_date="2026-09-30", symptoms="Damaged bark")
        self.node = MagicMock()
        self.node.node.get_text.return_value = "Cryphonectria parasitica causes chestnut bark cankers."
        self.rag = MagicMock()
        self.rag.retrieve_source_nodes.return_value = [self.node]
        self.rag.diversify_source_nodes.return_value = [self.node]
        self.rag.extract_sources.return_value = [{"id": 1, "label": "Mock reference"}]
        self.rag.format_retrieved_context.return_value = "[Source 1] Cryphonectria parasitica causes chestnut bark cankers."
        self.rag.get_llm.return_value.complete.return_value.text = (
            "Chestnut blight is a possibility. The corpus describes bark cankers [Source 1]. "
            "This does not establish that cankers are present in the photographs.")

    def run_assessment(self):
        with patch("app.identification.assessment._load_rag", return_value=self.rag):
            return assess_identification(self.identification, self.context)

    def assert_useful_fallback(self, result):
        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["assessment_origin"], "provider_fallback")
        self.assertIn("Cryphonectria parasitica", result["summary"])
        self.assertIn("0.28387", result["summary"])
        self.assertIn("close-up", result["summary"])
        self.assertEqual(len(result["possible_causes"]), 2)
        self.assertNotIn("private-secret", json.dumps(result))

    def test_normalization_to_retrieval_to_narrative(self):
        result = self.run_assessment()
        self.assertEqual(result["status"], "preliminary")
        self.assertIsNone(result["error_code"])
        self.assertNotIn("[Source 1]", result["summary"])
        self.assertIn("Chestnut blight is a possibility", result["summary"])
        self.assertEqual(result["possible_causes"][0]["confidence"], 0.28387)
        self.assertEqual(result["sources"][0]["passage"], self.node.node.get_text())
        self.rag.retrieve_source_nodes.assert_called_once_with(result["retrieval_query"])
        self.rag.get_llm.return_value.complete.assert_called_once()
        prompt = self.rag.get_llm.return_value.complete.call_args.args[0]
        self.assertIn("[Source 1]", prompt)
        self.assertIn("ENDOPA", prompt)

    def test_query_and_prompt_include_context_and_grounding_rules(self):
        query = build_retrieval_query(self.identification, self.context)
        for text in ("unknown", "ENDOPA", "ERWITR", "Ticino", "2026-09-30", "Damaged bark"):
            self.assertIn(text, query)
        prompt = build_assessment_prompt(self.identification, self.context, "reference")
        for text in ("Do NOT return JSON", "have NOT seen the photographs", "No definitive diagnosis",
                     "not probabilities", "Do not invent host associations", "Preserve scope, negation and qualifications"):
            self.assertIn(text, prompt)

    def test_optional_symptoms_do_not_block_narrative(self):
        self.context = AssessmentContext()
        result = self.run_assessment()
        self.assertEqual(result["status"], "preliminary")
        self.assertIsNone(result["reported_symptoms"])
        self.assertEqual(result["uncertainty"], "high")

    def test_model_error_retains_candidates_and_references(self):
        self.rag.get_llm.return_value.complete.side_effect = RuntimeError("private-secret")
        result = self.run_assessment()
        self.assert_useful_fallback(result)
        self.assertEqual(result["error_code"], "MODEL_REQUEST_FAILED")
        self.assertTrue(result["sources"])

    def test_empty_model_output_retains_useful_response(self):
        self.rag.get_llm.return_value.complete.return_value.text = " "
        result = self.run_assessment()
        self.assert_useful_fallback(result)
        self.assertEqual(result["error_code"], "MODEL_OUTPUT_EMPTY")

    def test_retrieval_error_retains_provider_results(self):
        self.rag.retrieve_source_nodes.side_effect = RuntimeError("private-secret")
        result = self.run_assessment()
        self.assert_useful_fallback(result)
        self.assertEqual(result["error_code"], "RETRIEVAL_FAILED")
        self.rag.get_llm.assert_not_called()

    def test_empty_corpus_retains_provider_results(self):
        self.rag.diversify_source_nodes.return_value = []
        result = self.run_assessment()
        self.assert_useful_fallback(result)
        self.assertEqual(result["error_code"], "NO_CORPUS_PASSAGES")

    def test_no_candidates_still_produces_unknown_and_next_steps(self):
        self.identification = normalize_response({"results": []})
        self.rag.diversify_source_nodes.return_value = []
        result = self.run_assessment()
        self.assertIn("unknown", result["summary"])
        self.assertIn("close-up", result["summary"])
        self.assertEqual(result["possible_causes"], [])

    def test_failed_identification_skips_rag(self):
        self.identification.status = "timeout"
        result = self.run_assessment()
        self.assertEqual(result["error_code"], "IDENTIFICATION_UNAVAILABLE")
        self.rag.retrieve_source_nodes.assert_not_called()

    def test_inline_source_numbers_removed_without_altering_scores(self):
        text = "A possibility [Source 8]. Score 0.284 (28.4%). Reference (Source 1). More [1, 2]."
        result = remove_inline_source_markers(text)
        self.assertNotIn("Source", result)
        self.assertNotIn("[1, 2]", result)
        self.assertIn("0.284 (28.4%)", result)
        self.assertIn("A possibility.", result)

    def test_unknown_species_does_not_default_to_chestnut(self):
        query = build_retrieval_query(self.identification, AssessmentContext())
        self.assertIn("Reported tree species (unverified): unknown", query)
        self.assertNotIn("Castanea sativa", query)
        prompt = build_assessment_prompt(self.identification, AssessmentContext(), "passages")
        self.assertIn("Disease candidates do not establish", prompt)
        self.assertIn("must not automatically be applied", prompt)
        result = self.run_assessment()
        self.assertIsNone(result["reported_species"])

    def test_reported_non_chestnut_species_reaches_retrieval_and_model(self):
        self.context.species = "Quercus robur"
        result = self.run_assessment()
        self.assertEqual(result["reported_species"], "Quercus robur")
        self.assertIn("Quercus robur", result["retrieval_query"])
        self.assertNotIn("Castanea sativa", result["retrieval_query"])
        self.assertIn("Quercus robur", self.rag.get_llm.return_value.complete.call_args.args[0])
        self.assertIn("coverage of other species may be limited", result["sources_note"])

    def test_reported_chestnut_is_used_only_when_supplied(self):
        self.context.species = "Castanea sativa"
        self.assertIn("Reported tree species (unverified): Castanea sativa",
                      build_retrieval_query(self.identification, self.context))


if __name__ == "__main__":
    unittest.main()

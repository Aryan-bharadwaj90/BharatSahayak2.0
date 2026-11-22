import unittest
from unittest.mock import patch

from src import eligibility
from src.data_pipeline import create_structured_chunks


class ChunkTests(unittest.TestCase):
    def deadlines(self, **scheme):
        return [chunk["text"] for chunk in create_structured_chunks(scheme)
                if chunk["section_type"] == "deadlines_and_amounts"]

    def test_no_amounts(self):
        self.assertEqual(self.deadlines(benefits=["Free advice"], documents=["ID"],
                                        important_notes=["Bring your ID"]), [])

    def test_amounts_once_per_scheme(self):
        chunks = self.deadlines(benefits=["Annual premium is Rs. 20.", "Advice"],
                                premium="Annual premium is Rs. 20.", last_date="2026-12-01")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].count("Rs. 20"), 1)
        self.assertIn("2026-12-01", chunks[0])
        self.assertNotIn("Advice", chunks[0])


class RuleTests(unittest.TestCase):
    def assess(self, conditions, profile, logic="all"):
        scheme = {"scheme_id": "example", "eligibility_rules": {
            "logic": logic, "conditions": conditions}}
        with patch.object(eligibility, "_load_schemes", return_value=[scheme]):
            return eligibility.check_eligibility("example", profile)["status"]

    def test_missing_rule_keys(self):
        rule = {"field": "x", "operator": "eq", "value": True}
        for key in rule:
            with self.subTest(key=key):
                incomplete = rule.copy()
                del incomplete[key]
                self.assertEqual(self.assess([incomplete], {"x": True}), "unknown")

    def test_false_expected(self):
        rule = {"field": "x", "operator": "truthy", "value": False}
        self.assertEqual(self.assess([rule], {"x": False}), "satisfied")
        self.assertEqual(self.assess([rule], {"x": True}), "not_satisfied")

    def test_alternative_conditions(self):
        rules = [{"field": x, "operator": "eq", "value": True} for x in ("a", "b")]
        for profile, expected in [({"a": False}, "unknown"), ({"a": True}, "satisfied"),
                                  ({"a": False, "b": False}, "not_satisfied")]:
            self.assertEqual(self.assess(rules, profile, "any"), expected)
        self.assertEqual(self.assess(rules, {"a": True, "b": True}, "other"), "unknown")


class EvaluationTests(unittest.TestCase):
    def test_partial_evidence_and_paired_source_match(self):
        from evaluation.evaluation import retrieval_metrics

        reference = {"reference_evidence": ["Annual premium is Rs. 20.", "Apply at a branch."],
                     "reference_source_title": "Scheme Portal", "reference_source_url": "https://portal.example/"}
        chunk = {"text": "Annual premium is Rs. 20.", "source_title": "Scheme Portal",
                 "source_url": "https://portal.example"}
        wrong_title = dict(chunk, source_title="Other portal")
        wrong_url = dict(chunk, source_url="https://other.example")
        scores = retrieval_metrics(reference, [chunk, wrong_title])
        self.assertEqual(scores["evidence_coverage"], 0.5)
        self.assertEqual(scores["source_precision_at_5"], 0.5)
        self.assertEqual(retrieval_metrics(reference, [wrong_title, wrong_url])["source_hit_at_5"], 0)
        self.assertEqual(retrieval_metrics(reference, [])["evidence_coverage"], 0)


class AppTests(unittest.TestCase):
    def test_selection_passes_through_to_retrieval_and_eligibility(self):
        import app

        with (patch.object(app, "KB_READY", True),
              patch.object(app, "retrieve", return_value=[{"scheme_id": "PMFBY"}]) as retrieve,
              patch.object(app, "generate_answer", return_value={"answer": "checked"}) as generate):
            result = app.answer_user("Tell me about PMJJBY", "{}", "en", "PMFBY")
        self.assertIn("checked", result)
        retrieve.assert_called_once_with("Tell me about PMJJBY", top_k=5, scheme_id="PMFBY")
        self.assertEqual(generate.call_args.args[2]["scheme_id"], "PMFBY")


class RetrievalTests(unittest.TestCase):
    def test_category_label_drift_keeps_crop_insurance_searchable(self):
        import faiss
        import numpy as np
        from rank_bm25 import BM25Okapi
        from src import retrieval

        chunk = {"scheme_id": "PMFBY", "category": "Crop insurance for agriculture",
                 "text": "Insurance for notified crop losses."}
        vector = np.array([[1.0]], dtype="float32")
        index = faiss.IndexFlatIP(1)
        index.add(vector)
        with (patch.object(retrieval, "_metadata", [chunk]),
              patch.object(retrieval, "_index", index),
              patch.object(retrieval, "_bm25", BM25Okapi([chunk["text"].lower().split()])),
              patch.object(retrieval, "embed_query", return_value=vector),
              patch.object(retrieval, "_reranker_failed", True)):
            hits = retrieval.retrieve("crop losses for farmers")
            self.assertEqual([c["scheme_id"] for c in hits], ["PMFBY"])
            self.assertEqual(retrieval.retrieve("crop losses", scheme_id="PMJDY"), [])


class AnswerTests(unittest.TestCase):
    def test_source_corrections_and_malformed_provider_response(self):
        from src import rag

        chunk = {"text": "Evidence", "source_title": "Old title", "source_url": "https://portal.example"}
        config = {"LLM_API_KEY": "test", "LLM_BASE_URL": "https://llm.invalid", "LLM_MODEL": "test"}
        with (patch.dict(rag.os.environ, config), patch.dict(rag._ANSWER_CACHE, clear=True),
              patch.object(rag.requests, "post") as post):
            post.return_value.json.return_value = {"choices": [{"message": {"content": "Answer"}}]}
            rag.generate_answer("Question", [chunk])
            corrected = rag.generate_answer("Question", [dict(chunk, source_title="New title")])
            self.assertEqual(corrected["sources"][0]["source_title"], "New title")
            post.return_value.json.return_value = {"choices": [None]}
            self.assertTrue(rag.generate_answer("Another question", [chunk])["error"])

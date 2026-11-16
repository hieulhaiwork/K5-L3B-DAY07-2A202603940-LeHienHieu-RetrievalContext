"""Offline adapter contract checks; these do not measure retrieval quality."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dense import DenseRetriever


class DenseRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.embedder = SimpleNamespace(model_name="model", dimensions=2,
                                        embed_query=Mock(return_value=[1.0, 0.0]))
        self.row = {"chunk_id": "c1", "document_id": "doc", "text": "Evidence",
                    "source": "book.pdf", "page": 148, "metadata": {"application_scope": ["word"]},
                    "score": 0.75}
        self.store = SimpleNamespace(model_name="model", dimensions=2,
                                     search=Mock(return_value=[self.row]))
        self.retriever = DenseRetriever(self.embedder, self.store)

    def test_unfiltered_search_maps_lab6_result_to_lab7(self):
        result = self.retriever.search("question", filters=None, top_k=3)
        self.store.search.assert_called_once_with("B", [1.0, 0.0], top_k=3, filters=None)
        self.assertEqual(result, [{"id": "c1", "source_id": "doc", "text": "Evidence",
                                  "source": "book.pdf", "page": 148,
                                  "metadata": self.row["metadata"], "score": 0.75}])

    def test_filtered_search_uses_same_b_index_through_c_alias(self):
        filters = {"application_scope": "word", "section": "3.4"}
        self.retriever.search("question", filters=filters, top_k=5)
        self.store.search.assert_called_once_with("C", [1.0, 0.0], top_k=5, filters=filters)

    def test_invalid_inputs_fail_before_embedding_api(self):
        for kwargs in ({"query": " "}, {"query": "q", "top_k": True},
                       {"query": "q", "top_k": 0}, {"query": "q", "top_k": 101},
                       {"query": "q", "filters": []},
                       {"query": "q", "filters": {"gold_page": "148"}},
                       {"query": "q", "filters": {"section": ""}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.retriever.search(**kwargs)
        self.embedder.embed_query.assert_not_called()
        self.store.search.assert_not_called()

    def test_profile_mismatch_is_rejected(self):
        for attr, value in (("model_name", "other-model"), ("dimensions", 3)):
            with self.subTest(attr=attr):
                store = SimpleNamespace(model_name="model", dimensions=2)
                setattr(store, attr, value)
                with self.assertRaises(ValueError):
                    DenseRetriever(self.embedder, store)

    def test_backend_failure_is_not_replaced_with_fake_results(self):
        self.store.search.side_effect = RuntimeError("database unavailable")
        with self.assertRaisesRegex(RuntimeError, "database unavailable"):
            self.retriever.search("question")


if __name__ == "__main__":
    unittest.main()

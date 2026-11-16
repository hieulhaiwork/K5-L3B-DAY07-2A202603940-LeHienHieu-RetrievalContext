"""Offline contracts for candidate reranking, validation and error propagation."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rerank import RerankRetriever


def chunk(identifier):
    return {'id': identifier, 'source_id': 'book', 'source': 'book.pdf', 'page': 1,
            'text': 'Evidence ' + identifier, 'metadata': {'application_scope': ['word']}, 'score': 0.8}


def ranking(ids):
    return {'tool_calls': [{'type':'function','function':{'name':'rank_candidates',
            'arguments':json.dumps({'ranked_ids':ids})}}]}


class RerankTests(unittest.TestCase):
    def setUp(self):
        self.rows=[chunk('a'),chunk('b'),chunk('c')]
        self.dense=SimpleNamespace(embedder=SimpleNamespace(),search=Mock(return_value=self.rows))
        self.ranker=SimpleNamespace(model_name='gpt-4o',complete=Mock(return_value=ranking(['c','a','b'])))
        self.retriever=RerankRetriever(self.dense,self.ranker,candidate_k=20)

    def test_rerank_changes_order_without_changing_source_text(self):
        results=self.retriever.search('Question?',filters={'application_scope':'word'},top_k=2)
        self.assertEqual([c['id'] for c in results],['c','a'])
        self.assertEqual(results[0]['text'],'Evidence c')
        self.assertEqual(results[0]['dense_rank'],3)
        self.assertEqual(results[0]['dense_score'],0.8)
        self.assertEqual(results[0]['score'],1.0)
        self.dense.search.assert_called_once_with('Question?',filters={'application_scope':'word'},top_k=20)
        messages,tools=self.ranker.complete.call_args.args
        self.assertEqual(json.loads(messages[1]['content'])['query'],'Question?')
        self.assertEqual(tools[0]['function']['parameters']['properties']['ranked_ids']['items']['enum'],['a','b','c'])
        self.assertEqual(self.retriever.last_search['reranked_ids'],['c','a','b'])

    def test_non_permutation_is_rejected_not_silently_replaced(self):
        for ids in (['a','a','c'],['a','b'],['a','b','unknown']):
            self.ranker.complete.return_value=ranking(ids)
            with self.subTest(ids=ids),self.assertRaises(ValueError):
                self.retriever.search('Question?')

    def test_empty_candidates_do_not_call_model(self):
        self.dense.search.return_value=[]
        self.assertEqual(self.retriever.search('Question?'),[])
        self.ranker.complete.assert_not_called()

    def test_reranker_error_preserves_dense_candidates_and_propagates(self):
        self.ranker.complete.side_effect=RuntimeError('API unavailable')
        with self.assertRaises(RuntimeError):self.retriever.search('Question?')
        self.assertEqual(len(self.retriever.last_search['dense_candidates']),3)
        self.assertEqual(self.retriever.last_search['status'],'reranking')


if __name__=='__main__':unittest.main()

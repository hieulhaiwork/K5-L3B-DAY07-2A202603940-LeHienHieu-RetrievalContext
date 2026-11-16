"""Offline checks for RRF fusion and metadata filtering before cutoff."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.hybrid import HybridRetriever


def chunk(identifier, scope='word', text='comment'):
    return {'id': identifier, 'text': text, 'source_id': 'book', 'source': 'book.pdf',
            'page': 1, 'metadata': {'application_scope': [scope], 'section': '3.4'}, 'score': 0.9}


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.c = [chunk(x) for x in ('a','b','c')]
        self.dense = SimpleNamespace(embedder=SimpleNamespace(), search=Mock(return_value=[self.a,self.b,self.c]))
        self.retriever = HybridRetriever(self.dense,[self.a,self.b,self.c],candidate_k=3,rrf_k=60)

    def test_reciprocal_rank_fusion_combines_ranks_not_raw_scores(self):
        self.retriever.index.search = Mock(return_value=[{**self.b,'score':999}, self.c, self.a])
        result=self.retriever.search('comment',top_k=3)
        self.assertEqual([c['id'] for c in result],['b','a','c'])
        self.assertAlmostEqual(result[0]['score'],1/62+1/61)
        self.assertEqual(result[0]['component_ranks'],{'dense':2,'lexical':1})
        self.assertEqual(len({c['id'] for c in result}),3)
        self.assertEqual(len(self.retriever.last_search['component_rankings']['dense']),3)

    def test_lexical_filter_runs_before_candidate_cutoff(self):
        rows=[chunk(str(i),'word','comment comment') for i in range(4)]
        target=chunk('target','google_docs','comment text')
        dense=SimpleNamespace(embedder=SimpleNamespace(),search=Mock(return_value=[]))
        r=HybridRetriever(dense,rows+[target],candidate_k=1)
        result=r.search('comment',filters={'application_scope':'google_docs'},top_k=1)
        self.assertEqual([c['id'] for c in result],['target'])
        dense.search.assert_called_once_with('comment',filters={'application_scope':'google_docs'},top_k=1)

    def test_component_failure_does_not_silently_become_lexical_only(self):
        self.dense.search.side_effect=RuntimeError('DB unavailable')
        with self.assertRaises(RuntimeError): self.retriever.search('comment')
        self.assertIsNone(self.retriever.last_search)

    def test_invalid_query_and_top_k_do_not_call_dense(self):
        for query,top_k in [('',3),('comment',True),('comment',101)]:
            with self.assertRaises(ValueError): self.retriever.search(query,top_k=top_k)
        self.dense.search.assert_not_called()


if __name__=='__main__': unittest.main()

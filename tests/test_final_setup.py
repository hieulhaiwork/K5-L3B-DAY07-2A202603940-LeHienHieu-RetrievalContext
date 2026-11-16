"""Contracts for frozen final evaluation and safe index reproduction."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.benchmark import validate_selection
from src.rebuild_index import rebuild_index
from src.dense import ROOT


class FinalSelectionTests(unittest.TestCase):
    def test_final_method_and_benchmark_match_frozen_selection(self):
        selected=validate_selection(ROOT/'src/configs/selected.json','rerank',ROOT/'data/questions_final.json')
        self.assertEqual(selected['selection_split'],'dev')
        self.assertEqual(selected['selected_method'],'rerank')
        old=json.loads((ROOT/'data/questions.json').read_text(encoding='utf-8'))
        new=json.loads((ROOT/'data/questions_final.json').read_text(encoding='utf-8'))
        self.assertEqual([q for q in old if q['split']=='dev'],[q for q in new if q['split']=='dev'])
        self.assertTrue({q['question'] for q in old if q['split']=='test'}.isdisjoint({q['question'] for q in new if q['split']=='test'}))
        self.assertEqual(sum(q['split']=='test' and q['answerable'] for q in new),5)
        self.assertEqual(sum(q['split']=='test' and not q['answerable'] for q in new),2)

    def test_exploratory_test_is_rejected_for_final_run(self):
        with self.assertRaises(ValueError):
            validate_selection(ROOT/'src/configs/selected.json','dense',ROOT/'data/questions.json')

    def test_parameter_drift_is_rejected(self):
        with patch('src.benchmark.configuration',return_value={'top_k':99}):
            with self.assertRaises(ValueError):
                validate_selection(ROOT/'src/configs/selected.json','rerank',ROOT/'data/questions_final.json')


class IndexReproductionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'chunks.json'
        self.path.write_text(json.dumps([{'id':'c1','source_id':'book','source':'book.pdf','page':1,'text':'Evidence','metadata':{}}]),encoding='utf-8')
        self.store=SimpleNamespace(create_schema=Mock(),count=Mock(return_value=0),upsert=Mock())
        self.embedder=SimpleNamespace(embed_documents=Mock(return_value=[[1.0,0.0]]))
        self.retriever=SimpleNamespace(store=self.store,embedder=self.embedder,verify_index=Mock(return_value={'chunks':1}))

    def test_empty_index_uses_real_embedder_interface_and_preserves_ids(self):
        result=rebuild_index(self.retriever,self.path)
        self.assertEqual(result['action'],'created')
        self.embedder.embed_documents.assert_called_once_with(['Evidence'])
        strategy,chunks,vectors=self.store.upsert.call_args.args
        self.assertEqual((strategy,chunks[0].chunk_id,chunks[0].document_id),('B','c1','book'))
        self.assertEqual(vectors,[[1.0,0.0]])

    def test_existing_index_is_verified_without_reembedding_or_overwrite(self):
        self.store.count.return_value=1
        self.assertEqual(rebuild_index(self.retriever,self.path)['action'],'verified_existing')
        self.embedder.embed_documents.assert_not_called();self.store.upsert.assert_not_called()

    def test_existing_mismatch_is_not_replaced(self):
        self.store.count.return_value=1
        self.retriever.verify_index.side_effect=ValueError('snapshot mismatch')
        with self.assertRaises(ValueError):rebuild_index(self.retriever,self.path)
        self.embedder.embed_documents.assert_not_called();self.store.upsert.assert_not_called()


if __name__=='__main__':unittest.main()

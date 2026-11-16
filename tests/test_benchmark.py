"""Check benchmark denominators and separation of ranking from final context."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmark import retrieval_records, summarize


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.chunk = {'id': 'c1', 'source_id': 'book', 'page': 10, 'text': 'Valid evidence.', 'score': 0.9}
        self.gold = {'source_id': 'book', 'page': 10, 'evidence': 'Valid evidence.'}
        self.questions = [
            {'id': 'a', 'split': 'dev', 'answerable': True, 'gold': [self.gold]},
            {'id': 'u', 'split': 'dev', 'answerable': False, 'gold': []}]
        self.traces = [
            {'query_id': 'a', 'status': 'answered', 'events': [{'event': 'search', 'results': [self.chunk]}],
             'context': '[]', 'citations': [], 'search_count': 1},
            {'query_id': 'u', 'status': 'abstain', 'events': [{'event': 'search', 'results': []}],
             'context': '[]', 'citations': [], 'search_count': 1}]

    def test_unanswerable_is_excluded_and_context_is_not_ranking(self):
        records, summary = summarize(self.questions, self.traces)
        self.assertEqual(summary['dev']['retrieval']['hit3'], {'n': 1, 'N': 1, 'rate': 1.0})
        self.assertEqual(summary['dev']['retrieval']['context_exact_evidence']['n'], 0)
        self.assertIsNone(records[1]['hit3'])

    def test_previous_network_failure_is_counted_in_time_not_question_denominator(self):
        self.traces[0]['previous_attempts'] = [
            {'status': 'error', 'events': [], 'elapsed_ms': 5000,
             'model_calls': [{'error': {'type': 'NetworkError'}}],
             'reported_embedding_cost_usd': 0}]
        _, summary = summarize(self.questions, self.traces)
        self.assertEqual(summary['dev']['retrieval']['hit3']['N'], 1)
        self.assertEqual(summary['dev']['previous_error_attempts'], 1)
        self.assertEqual(summary['dev']['elapsed_seconds'], 5)
        self.assertIsNone(summary['dev']['estimated_chat_cost_usd'])
        self.assertEqual(summary['dev']['chat_calls_with_unknown_cost'], 1)

    def test_matching_text_on_wrong_page_is_not_a_hit(self):
        self.chunk['page'] = 11
        self.assertFalse(retrieval_records(self.questions, self.traces)[0]['hit1'])

    def test_reviewed_context_and_answer_labels_are_aggregated_separately(self):
        reviews = [{'query_id': 'a', 'answer_grade': 0, 'citation_support': [], 'context_grade': 1},
                   {'query_id': 'u', 'citation_support': [], 'abstention_correct': True}]
        _, summary = summarize(self.questions, self.traces, reviews)
        self.assertEqual(summary['dev']['context_evidence_rate']['n'], 0)
        self.assertEqual(summary['dev']['agent']['grounded_answer']['mean'], 0)
        self.assertEqual(summary['dev']['agent']['abstention'], {'n': 1, 'N': 1})


if __name__ == '__main__':
    unittest.main()

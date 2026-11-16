"""Offline checks for the real-agent loop; mocks do not measure answer quality."""
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent import SYSTEM, pack_context, run_agent
from src.chat import OpenAIController, estimate_cost

CONFIG = {"provider": "OpenAI", "model": "gpt-4o", "temperature": 0,
          "max_completion_tokens": 100, "max_searches": 1, "top_k": 3,
          "context_budget_chars": 4000,
          "prices_per_million_tokens": {"input": 2.5, "cached_input": 1.25, "output": 10}}
CHUNK = {"id": "c1", "source_id": "book", "source": "book.pdf", "page": 1,
         "text": "A comment can be resolved.", "metadata": {}, "score": 0.9}


def action(name, **args):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call-" + name, "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}


class AgentTests(unittest.TestCase):
    def run_case(self, final, *, results=None, error=None):
        controller = SimpleNamespace(calls=[], complete=Mock(side_effect=[action("search", query="Question?"), final]))
        retriever = SimpleNamespace(search=Mock(return_value=[CHUNK] if results is None else results, side_effect=error))
        return run_agent("Question?", controller, retriever, CONFIG), controller, retriever

    def test_search_then_grounded_answer_and_trace(self):
        result, controller, retriever = self.run_case(action("finish_answer", answer="It can be resolved [c1].", citations=["c1"]))
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["search_count"], 1)
        self.assertEqual(result["context_ids"], ["c1"])
        retriever.search.assert_called_once_with("Question?", filters=None, top_k=3)
        first_tools = controller.complete.call_args_list[0].args[1]
        self.assertEqual(first_tools[0]["function"]["parameters"]["properties"]["query"]["enum"], ["Question?"])
        sent = controller.complete.call_args_list[1].args[0]
        self.assertEqual(sent[-1]["role"], "tool")
        self.assertIn("A comment can be resolved", sent[-1]["content"])
        self.assertIn("untrusted", SYSTEM)
        self.assertEqual(result["events"][1]["results"], [CHUNK])

    def test_model_abstention_discards_unsupported_explanations(self):
        result, _, _ = self.run_case(action("abstain", reason="No supporting evidence"))
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["reason"], "model_abstention")

    def test_invalid_citations_never_become_answered(self):
        for answer, citations in [("Claim [outside].", ["outside"]), ("Claim.", ["c1"]),
                                  ("Claim [c1] [outside].", ["c1"]), ("Claim [c1].", ["c1", "c1"])]:
            with self.subTest(answer=answer, citations=citations):
                result, _, _ = self.run_case(action("finish_answer", answer=answer, citations=citations))
                self.assertEqual(result["status"], "abstain")
                self.assertEqual(result["reason"], "invalid_citations")

    def test_empty_context_cannot_support_answer(self):
        result, _, _ = self.run_case(action("finish_answer", answer="Claim [c1].", citations=["c1"]), results=[])
        self.assertEqual(result["status"], "abstain")

    def test_errors_remain_errors_and_hide_exception_secrets(self):
        result, _, _ = self.run_case(action("abstain", reason="none"), error=RuntimeError("private DSN"))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["stage"], "retrieval")
        self.assertNotIn("private DSN", json.dumps(result))

    def test_search_limit_prevents_additional_retrieval(self):
        result, _, retriever = self.run_case(action("search", query="Question?"))
        self.assertEqual(result["status"], "error")
        self.assertEqual(retriever.search.call_count, 1)

    def test_dense_baseline_rejects_query_rewriting(self):
        controller = SimpleNamespace(calls=[], complete=Mock(return_value=action("search", query="Rewritten")))
        retriever = SimpleNamespace(search=Mock())
        result = run_agent("Question?", controller, retriever, CONFIG)
        self.assertEqual(result["status"], "error")
        retriever.search.assert_not_called()

    def test_invalid_citation_can_be_repaired_once_without_new_search(self):
        actions = [action("search", query="Question?"),
                   action("finish_answer", answer="Claim.", citations=["c1"]),
                   action("finish_answer", answer="Claim [c1].", citations=["c1"])]
        controller = SimpleNamespace(calls=[], complete=Mock(side_effect=actions))
        retriever = SimpleNamespace(search=Mock(return_value=[CHUNK]))
        result = run_agent("Question?", controller, retriever,
                           {**CONFIG, "max_finalization_attempts": 2})
        self.assertEqual(result["status"], "answered")
        self.assertEqual(controller.complete.call_count, 3)
        self.assertEqual(retriever.search.call_count, 1)
        self.assertEqual(sum(e["event"] == "invalid_citations" for e in result["events"]), 1)

    def test_context_deduplicates_without_cutting_evidence(self):
        context, ids = pack_context([CHUNK, CHUNK], 4000)
        self.assertEqual(ids, ["c1"])
        self.assertEqual(json.loads(context)[0]["text"], CHUNK["text"])
        context, ids = pack_context([CHUNK], 2)
        self.assertEqual((context, ids), ("[]", []))


class ChatClientTests(unittest.TestCase):
    def test_actual_request_format_usage_and_no_credentials_in_trace(self):
        response = {"id": "response-1", "model": "gpt-4o-snapshot",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                              "prompt_tokens_details": {"cached_tokens": 40}},
                    "choices": [{"finish_reason": "tool_calls", "message": action("search", query="Question?")}]}
        fake = Mock()
        fake.__enter__ = Mock(return_value=io.StringIO(json.dumps(response)))
        fake.__exit__ = Mock(return_value=False)
        with patch("src.chat.urllib.request.urlopen", return_value=fake) as send:
            client = OpenAIController("secret-api-key", CONFIG)
            client.complete([{"role": "user", "content": "Question?"}], [])
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.openai.com/v1/chat/completions")
        self.assertFalse(json.loads(request.data)["parallel_tool_calls"])
        self.assertNotIn("secret-api-key", json.dumps(client.calls))
        self.assertAlmostEqual(client.calls[0]["estimated_cost_usd"], 0.0004)
        self.assertIsNone(estimate_cost(None, CONFIG["prices_per_million_tokens"]))


if __name__ == "__main__":
    unittest.main()

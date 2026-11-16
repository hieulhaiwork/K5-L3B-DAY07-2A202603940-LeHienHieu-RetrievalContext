"""Run Advanced dev RAG using the existing basic agent without modifying basic files.

From repository root:
  python advanced/src/benchmark_agent.py --method dense --output advanced/artifacts/benchmark/dev_dense_v1
  python advanced/src/benchmark_agent.py --method hybrid --output advanced/artifacts/benchmark/dev_hybrid_v1
  python advanced/src/benchmark_agent.py --output advanced/artifacts/benchmark/dev_dense_v1 --summarize --reviews advanced/artifacts/benchmark/dev_dense_v1/reviews.json
Gold is used only after agent execution for diagnostics/review, never in model input.
"""
import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import dotenv_values
from src.agent import run_agent
from src.chat import OpenAIController
from src.hybrid import HybridRetriever
from src.rerank import RerankRetriever
from embedding_api import OpenAIEmbeddings, MODEL, DIMENSIONS
from index_store import AdvancedStore


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class AdvancedController(OpenAIController):
    def complete(self, messages, tools):
        messages = copy.deepcopy(messages)
        messages[0]["content"] = messages[0]["content"].replace(
            "The corpus covers Microsoft Word and Google Docs.",
            "The corpus is NVIDIA's 2026 Annual Report. Page numbers are physical PDF pages, starting at 1."
        )
        messages[0]["content"] += (
            "\nPreserve reporting periods, table hierarchy, units and signs. "
            "For requested arithmetic show the inputs and calculation, and round only the final result. "
            "Distinguish percentage points from percent. Do not guess missing visual content or precise numbers. "
            "A comparison of two expenses does not imply one is included in the other."
        )
        return super().complete(messages, tools)


def adapt(chunk):
    return {"id": chunk["chunk_id"], "source_id": chunk["metadata"]["document_id"],
            "source": "advanced/data/2026-Annual-Report-Web.pdf",
            "page": chunk["page"], "text": chunk["text"], "metadata": chunk["metadata"],
            "content_type": chunk["content_type"]}


class AdvancedDense:
    def __init__(self, env, stats, chunks):
        self.store = AdvancedStore(env.get("ADVANCED_DATABASE_URL"), env.get("DATABASE_URL"))
        self.stats, self.chunks = stats, chunks
        self.last_search = None
        # Read-only snapshot verification before any paid query embeddings.
        with self.store.connect() as connection:
            connection.execute("SET TRANSACTION READ ONLY")
            profile = connection.execute(
                "SELECT model,dimensions,profile,chunk_count FROM advanced_collections WHERE collection=%s",
                (stats["collection"],)).fetchone()
            if profile != (MODEL, DIMENSIONS, stats["profile"], len(chunks)):
                raise ValueError("Advanced database collection profile mismatch")
            rows = connection.execute(
                "SELECT chunk_id,text_content,page,content_type,metadata,vector_dims(embedding) "
                "FROM advanced_chunks WHERE collection=%s", (stats["collection"],)).fetchall()
        actual = {r[0]: r[1:] for r in rows}
        expected = {c["chunk_id"]: (c["text"], c["page"], c["content_type"], c["metadata"], DIMENSIONS)
                    for c in chunks}
        if actual != expected:
            raise ValueError("Advanced database snapshot differs from local chunks")
        self.embedder = OpenAIEmbeddings(env.get("OPENAI_API_KEY"),
                                        ROOT / "advanced/artifacts/index/cache/embeddings")

    def search(self, query, *, filters=None, top_k=8):
        if filters:
            raise ValueError("Advanced baseline does not support filters")
        self.last_search = None
        vector = self.embedder.embed([query])[0]
        rows = self.store.search(self.stats["collection"], vector, top_k)
        self.last_search = {"method": "dense", "collection": self.stats["collection"],
                            "model": MODEL, "dimensions": DIMENSIONS}
        return [{**adapt(r), "score": r["score"]} for r in rows]


def diagnostics(question, trace):
    # Region identity is a diagnostic, not proof that the content or answer is correct.
    contexts = set(trace.get("context_ids", []))
    found = {}
    for event in trace.get("events", []):
        for chunk in event.get("results", []):
            if chunk["id"] in contexts:
                for record in chunk["metadata"].get("source_records", []):
                    found[record["record_id"]] = record
    trace["citation_sources"] = {
        chunk["id"]: {"page": chunk["page"], "source_records": chunk["metadata"].get("source_records", [])}
        for event in trace.get("events", []) for chunk in event.get("results", [])
        if chunk["id"] in trace.get("citations", [])
    }
    coverage = []
    for gold in question["gold"]:
        if gold.get("record_id"):
            present = gold["record_id"] in found
        else:
            # Page-only approximation for text evidence; manual review remains necessary.
            present = any(r["page"] == gold["page"] for r in found.values())
        coverage.append({"page": gold["page"], "record_id": gold.get("record_id"),
                         "matching_source_present": present,
                         "match_basis": "record_id" if gold.get("record_id") else "page_only"})
    return {"query_id": question["id"], "primary_group": question["primary_group"],
            "answerable": question["answerable"], "gold_region_diagnostics": coverage,
            "note": "Source presence does not establish answer correctness or complete evidence support."}


def summarize(questions, traces, reviews=None):
    by_id = {t["query_id"]: t for t in traces}
    labels = {r["query_id"]: r for r in (reviews or [])}
    if len(labels) != len(reviews or []) or not set(labels).issubset(by_id):
        raise ValueError("Invalid review IDs")
    for identifier, label in labels.items():
        if type(label["passed"]) is not bool or not label.get("reason"):
            raise ValueError("Each review needs a boolean passed and a reason")
        if by_id[identifier]["status"] == "error" and label["passed"]:
            raise ValueError("System errors must fail")
    complete = set(labels) == {q["id"] for q in questions}
    groups = {}
    for group in sorted({q["primary_group"] for q in questions}):
        ids = [q["id"] for q in questions if q["primary_group"] == group]
        groups[group] = {"total": len(ids), "reviewed": sum(i in labels for i in ids),
                         "passed": sum(labels[i]["passed"] for i in ids if i in labels) if complete else None}
    chat = sum(t.get("estimated_chat_cost_usd") or 0 for t in traces)
    embedding_tokens = sum(e.get("usage", {}).get("prompt_tokens", 0)
                           for t in traces for e in t.get("embedding_events", []))
    return {"split": questions[0]["split"], "total": len(questions), "completed": len(traces),
            "status_counts": dict(Counter(t["status"] for t in traces)), "review_complete": complete,
            "passed": sum(r["passed"] for r in labels.values()) if complete else None,
            "pass_rate_percent": 100 * sum(r["passed"] for r in labels.values()) / len(questions) if complete else None,
            "groups": groups, "elapsed_seconds_sum": round(sum(t["elapsed_ms"] for t in traces) / 1000, 3),
            "estimated_chat_cost_usd": chat,
            "chat_cost_complete": all(t.get("estimated_chat_cost_usd") is not None for t in traces),
            "new_embedding_prompt_tokens": embedding_tokens,
            "estimated_embedding_cost_usd": embedding_tokens * .02 / 1_000_000,
            "pricing_note": "Estimates from configured GPT-4o rates and embedding rate USD 0.02/million input tokens; not invoices.",
            "grade_note": "Binary grades require review of answer, citations and original PDF. Retrieval diagnostics are not grades."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "advanced/data/dev_questions.json")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--selection", type=Path, help="Frozen selection required for test execution")
    parser.add_argument("--method", choices=["dense", "hybrid", "rerank"], default="dense")
    parser.add_argument("--agent-mode", choices=["baseline", "multisearch"], default="baseline")
    parser.add_argument("--vision-new-calls", type=int, default=12)
    parser.add_argument("--vision-per-question", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--context-chars", type=int, default=24000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    questions = read(args.questions)
    if (not questions or any(q["split"] != args.split for q in questions)
            or len({q["id"] for q in questions}) != len(questions)):
        raise ValueError("Question IDs must be unique and all questions must match --split")
    output = args.output.resolve()
    if args.summarize:
        traces = read(output / "traces.json")
        if read(output / "run.json")["questions_hash"] != digest(questions):
            raise ValueError("Questions changed since execution")
        result = summarize(questions, traces, read(args.reviews) if args.reviews else None)
        write(output / "summary.json", result)
        print(json.dumps(result, indent=2))
        return
    if not 1 <= args.top_k <= 50 or args.context_chars < 1000:
        raise ValueError("Invalid retrieval/context limits")
    stats = read(ROOT / "advanced/artifacts/index/index_stats.json")
    chunks = [json.loads(line) for line in (ROOT / "advanced/artifacts/index/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    if stats["stage"] != "indexed" or stats["chunks"] != len(chunks):
        raise ValueError("Incomplete index")
    config = read(ROOT / "src/configs/agent.json")
    config.update(retrieval_method=args.method, top_k=args.top_k, max_searches=1,
                  context_budget_chars=args.context_chars, max_completion_tokens=1800)
    if args.agent_mode == "multisearch":
        if args.vision_new_calls < 0 or not 0 <= args.vision_per_question <= 4:
            raise ValueError("Invalid vision limits")
        config.update(agent_mode="multisearch", max_searches=6,
                      vision_new_calls=args.vision_new_calls, vision_per_question=args.vision_per_question)
    if args.method == "hybrid":
        config["hybrid"] = {"candidate_k": 30, "rrf_k": 60, "lexical": "tfidf"}
    if args.method == "rerank":
        config["rerank"] = {**read(ROOT / "src/configs/rerank.json"), "ranking": "complete_candidate_scores", "max_completion_tokens": 1024}
    run = {"questions_hash": digest(questions), "config": config, "collection": stats["collection"],
           "snapshot_hash": stats["profile"]["snapshot_hash"], "split": args.split,
           "questions_file": args.questions.resolve().relative_to(ROOT).as_posix(),
           "runtime_agent_sha256": hashlib.sha256((ROOT / "src/agent.py").read_bytes()).hexdigest(),
           "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.agent_mode == "multisearch":
        run["multisearch_agent_sha256"] = hashlib.sha256((ROOT / "advanced/src/multisearch_agent.py").read_bytes()).hexdigest()
    if args.method == "rerank":
        run["rerank_adapter_sha256"] = hashlib.sha256((ROOT / "advanced/src/rerank_adapter.py").read_bytes()).hexdigest()
    if args.split == "test" and not args.selection:
        raise ValueError("Test execution requires a frozen --selection")
    if args.selection:
        selection = read(args.selection)
        for key in ("questions_hash", "config", "collection", "snapshot_hash",
                    "runtime_agent_sha256", "multisearch_agent_sha256"):
            if selection.get(key) != run.get(key):
                raise ValueError("Execution differs from the frozen test selection: " + key)
        run["selection_file"] = args.selection.resolve().relative_to(ROOT).as_posix()
        run["selection_hash"] = digest(selection)
    if output.exists():
        if not args.resume or read(output / "run.json") != run:
            raise ValueError("Output already exists or run profile differs; choose a new directory")
        traces = read(output / "traces.json")
    else:
        output.mkdir(parents=True)
        traces = []
        write(output / "run.json", run)
        write(output / "traces.json", traces)
    env = dotenv_values(ROOT / ".env")
    dense = AdvancedDense(env, stats, chunks)
    retriever = HybridRetriever(dense, [adapt(c) for c in chunks], **config["hybrid"]) if args.method == "hybrid" else dense
    if args.method == "rerank":
        settings = config["rerank"]
        rank_config = {**config, **{k: settings[k] for k in ("model", "temperature", "max_completion_tokens")}}
        from rerank_adapter import ScoringReranker
        ranker = ScoringReranker(env.get("OPENAI_API_KEY"), rank_config)
        retriever = RerankRetriever(dense, ranker, candidate_k=settings["candidate_k"])
    controller = AdvancedController(env.get("OPENAI_API_KEY"), config)
    inspector = None
    if args.agent_mode == "multisearch":
        from multisearch_agent import VisualInspector, run_multisearch
        inspector = VisualInspector(ROOT, env, config, output, limit=args.vision_new_calls)
    completed = {t["query_id"] for t in traces}
    for question in questions:
        if question["id"] in completed:
            continue
        # Only the question text and ID enter the runtime agent.
        if inspector is not None:
            trace = run_multisearch(question["question"], controller, retriever, config, inspector, query_id=question["id"])
        else:
            trace = run_agent(question["question"], controller, retriever, config, query_id=question["id"])
        trace["index_profile"] = {"collection": stats["collection"], "profile": stats["profile"]}
        diagnostic = diagnostics(question, trace)
        trace["evaluation_diagnostics"] = diagnostic
        traces.append(trace)
        write(output / "traces.json", traces)
        write(output / "summary.json", summarize(questions, traces))
        print(json.dumps({"query_id": question["id"], "status": trace["status"],
                          "elapsed_ms": trace["elapsed_ms"], "estimated_chat_cost_usd": trace["estimated_chat_cost_usd"]}), flush=True)
    print("Saved traces and ungraded summary to " + str(output), flush=True)


if __name__ == "__main__":
    main()

"""Run retrieval/agent benchmarks and aggregate evidence and reviewed answer metrics."""
import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from starter.agent_metrics import aggregate
from .agent import run_agent
from .dense import ROOT
from .retrievers import configured_controller, build_retriever, configuration

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def hit(chunk, gold):
    return any(chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"]
               and g["evidence"] in chunk["text"] for g in gold)


def retrieval_records(questions, traces):
    by_id = {q["id"]: q for q in questions}
    records = []
    for t in traces:
        q = by_id[t["query_id"]]
        searches = [e for e in t["events"] if e["event"] == "search"]
        results = searches[0].get("results", []) if searches else []
        context = json.loads(t.get("context", "[]"))
        gold = q.get("gold", [])
        records.append({"query_id": q["id"], "split": q["split"], "answerable": q["answerable"],
                        "status": t["status"], "gold": gold,
                        "retrieval_available": bool(searches and "results" in searches[0]),
                        "ranking": [{"id": c["id"], "source_id": c["source_id"], "page": c["page"],
                                     "score": c["score"], "evidence_hit": hit(c, gold)} for c in results],
                        "hit1": bool(results and hit(results[0], gold)) if q["answerable"] else None,
                        "hit3": any(hit(c, gold) for c in results[:3]) if q["answerable"] else None,
                        "context_exact_evidence": any(hit(c, gold) for c in context) if q["answerable"] else None})
    return records


def ratio(n, total):
    return {"n": n, "N": total, "rate": n / total if total else None}


def summarize(questions, traces, reviews=None):
    records = retrieval_records(questions, traces)
    summaries = {}
    for split in ("dev", "test"):
        cases = [q for q in questions if q["split"] == split]
        if not cases:
            continue
        selected_ids = {q["id"] for q in cases}
        ts = [t for t in traces if t["query_id"] in selected_ids]
        rs = [r for r in records if r["split"] == split]
        ans = [r for r in rs if r["answerable"]]
        available = [r for r in ans if r["retrieval_available"]]
        attempts = [a for t in ts for a in t.get("previous_attempts", []) + [t]]
        costs = [c.get("estimated_cost_usd") for a in attempts for c in a.get("model_calls", []) + a.get("reranker_calls", [])]
        embeddings = [a.get("reported_embedding_cost_usd") for a in attempts]
        retrieval_times = [sum(e.get("elapsed_ms", 0) for e in t["events"] if e["event"] == "search") for t in attempts]
        summary = {"cases": len(cases), "completed_traces": len(ts),
                   "statuses": dict(Counter(t["status"] for t in ts)),
                   "retrieval_coverage": ratio(len(available), len(ans)),
                   "retrieval": {metric: ratio(sum(bool(r[metric]) for r in available), len(available))
                                 for metric in ("hit1", "hit3", "context_exact_evidence")},
                   "search_count_distribution": dict(Counter(str(t.get("search_count", 0)) for t in ts)),
                   "model_call_count_distribution": dict(Counter(str(len(t.get("model_calls", []))) for t in ts)),
                   "reranker_call_count_distribution": dict(Counter(str(len(t.get("reranker_calls", []))) for t in ts)),
                   "elapsed_seconds": sum(t.get("elapsed_ms", 0) for t in attempts) / 1000,
                   "previous_error_attempts": len(attempts) - len(ts),
                   "retrieval_seconds": sum(retrieval_times) / 1000,
                   "estimated_chat_cost_usd": sum(costs) if all(c is not None for c in costs) else None,
                   "estimated_chat_cost_known_usd": sum(c for c in costs if c is not None),
                   "chat_calls_with_unknown_cost": sum(c is None for c in costs),
                   "reported_embedding_cost_usd": sum(embeddings) if all(c is not None for c in embeddings) else None,
                   "chat_usage": {key: sum((c.get("usage") or {}).get(key, 0)
                                           for t in attempts for c in t.get("model_calls", []) + t.get("reranker_calls", []))
                                  for key in ("prompt_tokens", "completion_tokens", "total_tokens")}}
        if reviews is not None:
            labels = [r for r in reviews if r["query_id"] in selected_ids]
            summary["agent"] = aggregate(cases, ts, labels)
            label_by_id = {r["query_id"]: r for r in labels}
            successful_ans = [r for r in ans if r["status"] != "error"]
            for r in successful_ans:
                grade = label_by_id[r["query_id"]].get("context_grade")
                if type(grade) is not int or grade not in (0, 1, 2):
                    raise ValueError("A context grade 0/1/2 is required for each successful answerable case")
            summary["context_grade_counts"] = dict(Counter(str(label_by_id[r["query_id"]]["context_grade"]) for r in successful_ans))
            summary["context_evaluation_coverage"] = ratio(len(successful_ans), len(ans))
            summary["context_evidence_rate"] = ratio(sum(label_by_id[r["query_id"]]["context_grade"] == 2 for r in successful_ans), len(successful_ans))
        summaries[split] = summary
    return records, summaries


def validate_selection(selection_path, method, questions_path, root=ROOT):
    selected = load(selection_path)
    if method not in {selected['selected_method'], selected['baseline_method']}:
        raise ValueError('Final evaluation permits only the baseline and selected method')
    key = 'selected_config' if method == selected['selected_method'] else 'baseline_config'
    if configuration(method, root) != selected[key]:
        raise ValueError('Current configuration differs from the frozen selection')
    if Path(questions_path).resolve() != (Path(root) / selected['evaluation_questions']).resolve():
        raise ValueError('Use the frozen final benchmark file')
    all_questions = load(questions_path)
    if [q['id'] for q in all_questions if q['split'] == 'dev'] != selected['dev_question_ids']:
        raise ValueError('Dev questions differ from selection inputs')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("dense", "hybrid", "rerank"), default="dense")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/questions.json")
    parser.add_argument("--split", choices=("dev", "test", "all"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--selection", type=Path, help="Enforce frozen final configuration and benchmark")
    parser.add_argument("--retry-errors", action="store_true", help="Retry service errors or malformed reranker output; retain previous attempts")
    args = parser.parse_args()
    if args.selection:
        validate_selection(args.selection, args.method, args.questions)
    questions = [q for q in load(args.questions) if args.split == "all" or q["split"] == args.split]
    if not questions or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Empty questions or duplicate IDs")
    trace_path = args.output / "traces.json"
    if args.summarize_only:
        traces = load(trace_path)
        if {t["query_id"] for t in traces} != {q["id"] for q in questions}:
            raise ValueError("Trace IDs do not match selected questions")
    elif args.retry_errors:
        traces = load(trace_path)
        controller, config = configured_controller(args.method)
        by_id = {q["id"]: q for q in questions}
        with TemporaryDirectory(prefix="lab7-retry-") as cache:
            retriever = build_retriever(cache, args.method, config)
            profile = retriever.verify_index(ROOT / "data/chunks.json")
            for index, previous in enumerate(traces):
                if previous["status"] != "error":
                    continue
                retryable = any(c.get("error", {}).get("type") == "NetworkError"
                                or c.get("error", {}).get("http_status") in {429, 500, 502, 503, 504}
                                for c in previous.get("model_calls", []) + previous.get("reranker_calls", []))
                retryable |= any(e.get("event") in {"network_error", "provider_error"}
                                 for e in previous.get("embedding_events", []))
                retryable |= previous.get("error", {}).get("type") in {"OperationalError", "ConnectionTimeout"}
                # Retry malformed model ranking output, never a low-quality valid ranking.
                rerank_incomplete = any(e.get("retrieval_details", {}).get("status") == "reranking"
                                        for e in previous.get("events", []) if e.get("event") == "search")
                retryable |= (previous.get("error", {}).get("type") == "ValueError"
                              and bool(previous.get("reranker_calls")) and rerank_incomplete)
                if not retryable:
                    continue
                if previous["config"] != config:
                    raise ValueError("Cannot retry with a changed configuration")
                q = by_id[previous["query_id"]]
                history = previous.get("previous_attempts", [])
                old = {k: v for k, v in previous.items() if k != "previous_attempts"}
                trace = run_agent(q["question"], controller, retriever, config,
                                  query_id=q["id"], filters=q.get("filter"))
                trace.update({"split": q["split"], "index_profile": profile,
                              "previous_attempts": history + [old]})
                traces[index] = trace
                save(trace_path, traces)
                print(f"Retry {q['id']}: {trace['status']}", flush=True)
    else:
        if trace_path.exists():
            raise ValueError("Trace file exists; use --summarize-only or a new output folder")
        controller, config = configured_controller(args.method)
        traces = []
        with TemporaryDirectory(prefix="lab7-benchmark-") as cache:
            retriever = build_retriever(cache, args.method, config)
            profile = retriever.verify_index(ROOT / "data/chunks.json")
            args.output.mkdir(parents=True, exist_ok=True)
            save(trace_path, traces)
            # Dev first; the same unchanged baseline config is then run on test.
            ordered = sorted(questions, key=lambda q: q["split"] != "dev")
            for index, q in enumerate(ordered, 1):
                trace = run_agent(q["question"], controller, retriever, config,
                                  query_id=q["id"], filters=q.get("filter"))
                trace.update({"split": q["split"], "index_profile": profile})
                traces.append(trace)
                save(trace_path, traces)
                print(f"{index}/{len(ordered)} {q['id']} {q['split']}: {trace['status']}", flush=True)
    reviews = load(args.reviews) if args.reviews else None
    records, summary = summarize(questions, traces, reviews)
    save(args.output / "retrieval.json", records)
    save(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

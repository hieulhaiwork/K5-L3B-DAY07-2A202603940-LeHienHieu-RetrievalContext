"""Shared method configuration and retriever construction for CLI and benchmark."""
import json
from pathlib import Path
from dotenv import dotenv_values
from .chat import OpenAIController
from .dense import ROOT, from_project
from .hybrid import HybridRetriever
from .rerank import RerankRetriever


def configuration(method, root=ROOT):
    if method not in {"dense", "hybrid", "rerank"}:
        raise ValueError("Unsupported retrieval method")
    root = Path(root)
    config = json.loads((root / "src/configs/agent.json").read_text(encoding="utf-8"))
    config["retrieval_method"] = method
    if method in {"hybrid", "rerank"}:
        config[method] = json.loads((root / f"src/configs/{method}.json").read_text(encoding="utf-8"))
    return config


def configured_controller(method, root=ROOT):
    root = Path(root)
    config = configuration(method, root)
    return OpenAIController(dotenv_values(root / ".env").get("OPENAI_API_KEY"), config), config


def build_retriever(cache, method, config, root=ROOT):
    root = Path(root)
    dense = from_project(cache_dir=Path(cache), root=root)
    if method == "hybrid":
        chunks = json.loads((root / "data/chunks.json").read_text(encoding="utf-8"))
        return HybridRetriever(dense, chunks, **config["hybrid"])
    if method == "rerank":
        settings = config["rerank"]
        rank_config = {**config, **{k: settings[k] for k in ("model", "temperature", "max_completion_tokens")}}
        reranker = OpenAIController(dotenv_values(root / ".env").get("OPENAI_API_KEY"), rank_config)
        return RerankRetriever(dense, reranker, candidate_k=settings["candidate_k"])
    return dense

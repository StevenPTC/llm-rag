import argparse
import json
from pathlib import Path

from query import retrieve_stages


DEFAULT_STAGE_CUTOFFS = {
    "dense": [1, 3, 5, 10, 20],
    "hybrid": [1, 3, 5, 10, 20],
    "rerank": [1, 3, 5, 10, 20],
    "final": [1, 3, 5, 10],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline retrieval evaluation for the RAG system")
    parser.add_argument("--dataset", required=True, help="Path to evaluation JSONL dataset")
    parser.add_argument("--backend", choices=["faiss", "chroma"], default="faiss", help="Vector store backend")
    parser.add_argument("--dense-top-k", type=int, default=20, help="Number of dense retrieval candidates")
    parser.add_argument("--hybrid-top-k", type=int, default=20, help="Number of candidates kept after hybrid merge")
    parser.add_argument("--top-k", type=int, default=5, help="Final number of primary chunks after reranking")
    parser.add_argument("--product-list-top-k", type=int, default=20, help="Unique product contexts for product-list queries")
    parser.add_argument("--adjacent-window", type=int, default=1, help="How many adjacent chunks to include per selected chunk")
    parser.add_argument("--parent-context-tokens", type=int, default=700, help="Compact parent context token window")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Cross-encoder reranker model name")
    parser.add_argument("--output", default=None, help="Optional path to save detailed JSON report")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_query_args(eval_args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        backend=eval_args.backend,
        top_k=eval_args.top_k,
        product_list_top_k=eval_args.product_list_top_k,
        dense_top_k=eval_args.dense_top_k,
        hybrid_top_k=eval_args.hybrid_top_k,
        adjacent_window=eval_args.adjacent_window,
        parent_context_tokens=eval_args.parent_context_tokens,
        embed_model=eval_args.embed_model,
        reranker_model=eval_args.reranker_model,
        disable_hybrid=False,
        disable_rerank=False,
        llm_provider="ollama",
        chat_model=None,
        think="false",
        ollama_base_url="http://localhost:11434",
        ollama_timeout=1200,
        retrieval_only=True,
    )


def target_matches(result: dict, target: dict) -> bool:
    checks = []

    if target.get("chunk_id"):
        result_chunk_ids = {result.get("chunk_id"), result.get("context_id")}
        result_chunk_ids.update(result.get("matched_child_ids", []))
        checks.append(target["chunk_id"] in result_chunk_ids)
    if target.get("source"):
        checks.append(result.get("source") == target["source"])
    if target.get("page") is not None:
        checks.append(result.get("start_page", result.get("page")) <= target["page"] <= result.get("end_page", result.get("page")))
    if target.get("start_page") is not None:
        checks.append(result.get("end_page", result.get("page")) >= target["start_page"])
    if target.get("end_page") is not None:
        checks.append(result.get("start_page", result.get("page")) <= target["end_page"])
    if target.get("product"):
        checks.append(result.get("product") == target["product"])
    if target.get("product_codes"):
        checks.append(set(target["product_codes"]).issubset(set(result.get("product_codes", []))))
    if target.get("text_product_codes"):
        checks.append(set(target["text_product_codes"]).issubset(set(result.get("text_product_codes", []))))
    if target.get("source_product_codes"):
        checks.append(set(target["source_product_codes"]).issubset(set(result.get("source_product_codes", []))))
    if target.get("chip_models"):
        checks.append(set(target["chip_models"]).issubset(set(result.get("chip_models", []))))
    if target.get("vendors"):
        checks.append(set(target["vendors"]).issubset(set(result.get("vendors", []))))
    if target.get("version"):
        checks.append(result.get("version") == target["version"])
    if target.get("section"):
        section_value = (result.get("section") or result.get("title") or "").lower()
        checks.append(target["section"].lower() in section_value)
    if target.get("tags"):
        result_tags = set(result.get("tags", []))
        checks.append(set(target["tags"]).issubset(result_tags))
    if target.get("text_contains"):
        checks.append(target["text_contains"].lower() in result.get("text", "").lower())

    return bool(checks) and all(checks)


def first_relevant_rank(results: list[dict], targets: list[dict], cutoff: int | None = None) -> int | None:
    limit = cutoff if cutoff is not None else len(results)
    for rank, item in enumerate(results[:limit], start=1):
        if any(target_matches(item, target) for target in targets):
            return rank
    return None


def hit_at_k(results: list[dict], targets: list[dict], k: int) -> bool:
    return first_relevant_rank(results, targets, cutoff=k) is not None


def reciprocal_rank(results: list[dict], targets: list[dict], cutoff: int | None = None) -> float:
    rank = first_relevant_rank(results, targets, cutoff=cutoff)
    return 0.0 if rank is None else 1.0 / rank


def summarize_stage(stage_name: str, rows: list[dict]) -> dict:
    cutoffs = [cutoff for cutoff in DEFAULT_STAGE_CUTOFFS[stage_name] if any(f"hit@{cutoff}" in row for row in rows)]
    summary = {
        "queries": len(rows),
        "mrr": round(sum(row["mrr"] for row in rows) / max(len(rows), 1), 4),
        "mean_first_relevant_rank": None,
    }

    hit_ranks = [row["first_relevant_rank"] for row in rows if row["first_relevant_rank"] is not None]
    if hit_ranks:
        summary["mean_first_relevant_rank"] = round(sum(hit_ranks) / len(hit_ranks), 4)

    for cutoff in cutoffs:
        key = f"hit@{cutoff}"
        summary[key] = round(sum(1 for row in rows if row[key]) / max(len(rows), 1), 4)

    return summary


def evaluate_query(record: dict, query_args: argparse.Namespace) -> dict:
    stages = retrieve_stages(record["query"], query_args)
    targets = record.get("targets", [])
    if not targets:
        raise ValueError("Each evaluation row must include a non-empty `targets` list.")

    stage_results = {}
    for stage_name, results in stages.items():
        stage_row = {
            "first_relevant_rank": first_relevant_rank(results, targets),
            "mrr": reciprocal_rank(results, targets),
        }
        for cutoff in DEFAULT_STAGE_CUTOFFS[stage_name]:
            effective_cutoff = min(cutoff, len(results))
            if effective_cutoff <= 0:
                continue
            stage_row[f"hit@{cutoff}"] = hit_at_k(results, targets, cutoff)
        stage_results[stage_name] = stage_row

    dense_rank = stage_results["dense"]["first_relevant_rank"]
    rerank_rank = stage_results["rerank"]["first_relevant_rank"]
    final_rank = stage_results["final"]["first_relevant_rank"]

    return {
        "query": record["query"],
        "targets": targets,
        "stages": stage_results,
        "rerank_improved_vs_dense": (
            dense_rank is None and rerank_rank is not None
        ) or (
            dense_rank is not None and rerank_rank is not None and rerank_rank < dense_rank
        ),
        "final_improved_vs_dense": (
            dense_rank is None and final_rank is not None
        ) or (
            dense_rank is not None and final_rank is not None and final_rank < dense_rank
        ),
    }


def build_report(rows: list[dict]) -> dict:
    stage_names = ["dense", "hybrid", "rerank", "final"]
    by_stage = {stage_name: [] for stage_name in stage_names}

    for row in rows:
        for stage_name in stage_names:
            by_stage[stage_name].append(row["stages"][stage_name])

    summary = {stage_name: summarize_stage(stage_name, stage_rows) for stage_name, stage_rows in by_stage.items()}

    rerank_improved = sum(1 for row in rows if row["rerank_improved_vs_dense"])
    final_improved = sum(1 for row in rows if row["final_improved_vs_dense"])
    total = max(len(rows), 1)
    summary["improvement_vs_dense"] = {
        "rerank_improved_queries": rerank_improved,
        "rerank_improved_rate": round(rerank_improved / total, 4),
        "final_improved_queries": final_improved,
        "final_improved_rate": round(final_improved / total, 4),
        "delta_mrr_rerank_vs_dense": round(summary["rerank"]["mrr"] - summary["dense"]["mrr"], 4),
        "delta_mrr_final_vs_dense": round(summary["final"]["mrr"] - summary["dense"]["mrr"], 4),
        "delta_hit@5_rerank_vs_dense": round(summary["rerank"].get("hit@5", 0.0) - summary["dense"].get("hit@5", 0.0), 4),
        "delta_hit@5_final_vs_dense": round(summary["final"].get("hit@5", 0.0) - summary["dense"].get("hit@5", 0.0), 4),
    }
    return summary


def print_summary(summary: dict) -> None:
    print("\nOffline Retrieval Evaluation\n")
    for stage_name in ["dense", "hybrid", "rerank", "final"]:
        stage = summary[stage_name]
        print(
            f"{stage_name:>6} | "
            f"MRR={stage['mrr']:.4f} "
            f"Hit@1={stage.get('hit@1', 0.0):.4f} "
            f"Hit@3={stage.get('hit@3', 0.0):.4f} "
            f"Hit@5={stage.get('hit@5', 0.0):.4f} "
            f"MeanFirstRank={stage['mean_first_relevant_rank']}"
        )

    improvement = summary["improvement_vs_dense"]
    print("\nImprovement vs dense baseline")
    print(
        f"rerank improved queries={improvement['rerank_improved_queries']} "
        f"({improvement['rerank_improved_rate']:.4f}), "
        f"delta_mrr={improvement['delta_mrr_rerank_vs_dense']:.4f}, "
        f"delta_hit@5={improvement['delta_hit@5_rerank_vs_dense']:.4f}"
    )
    print(
        f"final improved queries={improvement['final_improved_queries']} "
        f"({improvement['final_improved_rate']:.4f}), "
        f"delta_mrr={improvement['delta_mrr_final_vs_dense']:.4f}, "
        f"delta_hit@5={improvement['delta_hit@5_final_vs_dense']:.4f}"
    )


def main() -> None:
    args = parse_args()
    dataset = load_jsonl(Path(args.dataset))
    query_args = build_query_args(args)

    rows = [evaluate_query(record, query_args) for record in dataset]
    summary = build_report(rows)
    print_summary(summary)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "queries": rows}, f, ensure_ascii=False, indent=2)
        print(f"\nSaved detailed report to: {output_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from rag_utils import (
    CHROMA_DIR,
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OPENAI_CHAT_MODEL,
    DEFAULT_RERANK_MODEL,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    build_embedding_text,
    build_prompt,
    build_rerank_text,
    build_structured_prompt,
    call_ollama_chat,
    embed_texts,
    ensure_openai_client,
    get_cross_encoder,
    list_ollama_models,
    load_jsonl,
)
from specs import (
    build_product_specs,
    detect_vendor_mentions,
    format_product_spec_line,
    format_structured_context,
    plan_specs_query,
    product_matches_plan,
    sort_products_by_retrieval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start an interactive prompt for asking multiple questions",
    )
    parser.add_argument("--backend", choices=["faiss", "chroma"], default="faiss", help="Vector store backend")
    parser.add_argument("--top-k", type=int, default=5, help="Final number of primary chunks after reranking")
    parser.add_argument(
        "--product-list-top-k",
        type=int,
        default=20,
        help="Maximum number of unique product contexts for vendor/product listing questions",
    )
    parser.add_argument("--dense-top-k", type=int, default=20, help="Number of dense retrieval candidates")
    parser.add_argument("--hybrid-top-k", type=int, default=20, help="Number of candidates kept after hybrid merge")
    parser.add_argument("--adjacent-window", type=int, default=1, help="How many adjacent chunks to include per selected chunk")
    parser.add_argument(
        "--parent-context-tokens",
        type=int,
        default=700,
        help="Maximum compact parent-context token window passed to the LLM for each final context",
    )
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding model name")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANK_MODEL, help="Cross-encoder reranker model name")
    parser.add_argument("--disable-hybrid", action="store_true", help="Disable lexical+dense hybrid retrieval")
    parser.add_argument("--disable-rerank", action="store_true", help="Disable reranking and keep hybrid order")
    parser.add_argument(
        "--llm-provider",
        choices=["ollama", "openai"],
        default=DEFAULT_LLM_PROVIDER,
        help="LLM provider for final answer generation",
    )
    parser.add_argument(
        "--chat-model",
        default=None,
        help="LLM chat model name. For Ollama, omit this to select from local models.",
    )
    parser.add_argument(
        "--think",
        choices=["false", "true", "low", "medium", "high"],
        default="false",
        help="Thinking mode for supported Ollama models",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help="Ollama server base URL",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=1200,
        help="Timeout in seconds for the Ollama API call",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only print retrieved contexts, do not call the chat model",
    )
    parser.add_argument(
        "--debug-retrieval",
        action="store_true",
        help="Write a retrieval debug JSON file for this query",
    )
    parser.add_argument(
        "--debug-dir",
        default=str(METADATA_PATH.parent / "debug"),
        help="Directory for retrieval debug files",
    )
    parser.add_argument(
        "--no-print-results",
        action="store_true",
        help="Do not print retrieved contexts to stdout",
    )
    parser.add_argument(
        "--disable-structured-filter",
        action="store_true",
        help="Disable specs metadata query planning and structured filtering",
    )
    parser.add_argument(
        "--structured-product-limit",
        type=int,
        default=20,
        help="Maximum number of filtered products passed to the LLM for structured spec queries",
    )
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    english_tokens = re.findall(r"[a-z0-9][a-z0-9._/-]*", lowered)
    chinese_tokens = re.findall(r"[\u4e00-\u9fff]{1,4}", text)
    return english_tokens + chinese_tokens


def load_metadata() -> tuple[list[dict], dict[str, dict]]:
    records = load_jsonl(METADATA_PATH)
    metadata_by_id = {record["chunk_id"]: record for record in records}
    return records, metadata_by_id


def known_vendors_from_records(records: list[dict]) -> list[str]:
    return sorted(unique_list_values(records, "vendors"), key=len, reverse=True)


def search_faiss(question: str, top_k: int, embed_model: str, records: list[dict]) -> list[dict]:
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("Missing package: faiss-cpu. Install with `pip install faiss-cpu`.") from exc

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError("Run `python3 index.py --backend faiss` before querying.")

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    query_vector = embed_texts([question], model_name=embed_model)
    scores, indices = index.search(query_vector, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        item = records[idx].copy()
        item["dense_score"] = float(score)
        item["dense_rank"] = rank
        results.append(item)
    return results


def search_chroma(question: str, top_k: int, embed_model: str, metadata_by_id: dict[str, dict]) -> list[dict]:
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError("Missing package: chromadb. Install with `pip install chromadb`.") from exc

    if not CHROMA_DIR.exists():
        raise FileNotFoundError("Run `python3 index.py --backend chroma` before querying.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name="pdf_chunks")
    query_vector = embed_texts([question], model_name=embed_model)
    response = collection.query(query_embeddings=query_vector.tolist(), n_results=top_k)

    results = []
    for rank, (chunk_id, distance) in enumerate(zip(response["ids"][0], response["distances"][0]), start=1):
        metadata = metadata_by_id[chunk_id].copy()
        metadata["dense_score"] = 1.0 / (1.0 + float(distance))
        metadata["dense_rank"] = rank
        results.append(metadata)
    return results


def lexical_search(question: str, records: list[dict], top_k: int) -> list[dict]:
    query_tokens = tokenize(question)
    if not query_tokens:
        return []

    document_tokens = [tokenize(build_embedding_text(record)) for record in records]
    avg_doc_len = sum(len(tokens) for tokens in document_tokens) / max(len(document_tokens), 1)
    doc_freq = Counter()
    for tokens in document_tokens:
        doc_freq.update(set(tokens))

    query_tf = Counter(query_tokens)
    total_docs = len(records)
    scored = []

    for record, tokens in zip(records, document_tokens):
        token_counts = Counter(tokens)
        score = 0.0
        doc_len = max(len(tokens), 1)
        for token, query_count in query_tf.items():
            tf = token_counts.get(token, 0)
            if not tf:
                continue
            df = doc_freq.get(token, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            k1 = 1.5
            b = 0.75
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1))
            score += query_count * idf * numerator / denominator
        if score > 0:
            item = record.copy()
            item["lexical_score"] = float(score)
            scored.append(item)

    scored.sort(key=lambda item: item["lexical_score"], reverse=True)
    for rank, item in enumerate(scored, start=1):
        item["lexical_rank"] = rank
    return scored[:top_k]


def reciprocal_rank_fusion(dense_results: list[dict], lexical_results: list[dict], top_k: int) -> list[dict]:
    merged: dict[str, dict] = {}

    for result in dense_results:
        item = result.copy()
        item.setdefault("lexical_score", 0.0)
        item.setdefault("lexical_rank", None)
        item["hybrid_score"] = 1.0 / (60 + item["dense_rank"])
        merged[item["chunk_id"]] = item

    for result in lexical_results:
        if result["chunk_id"] in merged:
            merged[result["chunk_id"]]["lexical_score"] = result["lexical_score"]
            merged[result["chunk_id"]]["lexical_rank"] = result["lexical_rank"]
            merged[result["chunk_id"]]["hybrid_score"] += 1.0 / (60 + result["lexical_rank"])
        else:
            item = result.copy()
            item.setdefault("dense_score", 0.0)
            item.setdefault("dense_rank", None)
            item["hybrid_score"] = 1.0 / (60 + item["lexical_rank"])
            merged[item["chunk_id"]] = item

    ranked = sorted(merged.values(), key=lambda item: item["hybrid_score"], reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["hybrid_rank"] = rank
    return ranked[:top_k]


def detect_requested_vendors(question: str, known_vendors: list[str] | None = None) -> list[str]:
    return detect_vendor_mentions(question, known_vendors=known_vendors)


def detect_requested_chip_models(question: str) -> list[str]:
    chips = [
        match.upper()
        for match in re.findall(
            r"\b(?:RK\d{4}[A-Z0-9]*|STM32[A-Z0-9]+|RTL\d+[A-Z0-9]*|RZ/[A-Z0-9]+)\b",
            question,
            flags=re.IGNORECASE,
        )
    ]
    chips.extend(match for match in re.findall(r"\bi\.MX[0-9A-Za-z]+(?:\s+\w+)?\b", question, flags=re.IGNORECASE))
    return list(dict.fromkeys(chips))


def asks_for_product_list(question: str) -> bool:
    return bool(re.search(r"(?i)(產品|型號|有哪些|列出|尋找|list|which|what|product|model)", question))


def metadata_query_bonus(question: str, item: dict, known_vendors: list[str] | None = None) -> float:
    requested_vendors = detect_requested_vendors(question, known_vendors=known_vendors)
    requested_chips = set(detect_requested_chip_models(question))
    if not requested_vendors and not requested_chips:
        return 0.0

    item_vendors = set(item.get("vendors", []))
    item_chips = set(item.get("chip_models", []))
    list_intent = asks_for_product_list(question)
    bonus = 0.0

    if requested_chips:
        if requested_chips.intersection(item_chips):
            bonus += 8.0 if list_intent else 4.0
        elif item_chips:
            bonus -= 8.0 if list_intent else 4.0

    for vendor in requested_vendors:
        if vendor in item_vendors:
            bonus += 6.0 if list_intent else 3.0
        else:
            bonus -= 6.0 if list_intent else 3.0

    if list_intent and (item.get("product_codes") or item.get("chip_models") or item.get("product")):
        bonus += 1.0
    if item.get("product_codes") and item.get("chip_models"):
        bonus += 0.5

    return bonus


def add_metadata_vendor_candidates(question: str, records: list[dict], candidates: list[dict]) -> list[dict]:
    known_vendors = known_vendors_from_records(records)
    requested_vendors = set(detect_requested_vendors(question, known_vendors=known_vendors))
    requested_chips = set(detect_requested_chip_models(question))
    if not (requested_vendors or requested_chips) or not asks_for_product_list(question):
        return candidates

    merged = {item["chunk_id"]: item.copy() for item in candidates}
    base_score = max((item.get("hybrid_score", item.get("dense_score", 0.0)) for item in candidates), default=0.0)

    for record in records:
        if requested_chips and not requested_chips.intersection(record.get("chip_models", [])):
            continue
        if requested_vendors and not requested_vendors.intersection(record.get("vendors", [])):
            continue
        if not (record.get("product") or record.get("product_codes") or record.get("chip_models")):
            continue
        if record["chunk_id"] in merged:
            merged[record["chunk_id"]]["metadata_vendor_match"] = True
            continue

        item = record.copy()
        item["dense_score"] = 0.0
        item["lexical_score"] = 0.0
        item["hybrid_score"] = base_score + metadata_query_bonus(question, item, known_vendors=known_vendors)
        item["metadata_vendor_match"] = True
        merged[item["chunk_id"]] = item

    ranked = sorted(
        merged.values(),
        key=lambda item: (
            item.get("metadata_vendor_match", False),
            item.get("hybrid_score", item.get("dense_score", 0.0)),
        ),
        reverse=True,
    )
    for rank, item in enumerate(ranked, start=1):
        item["hybrid_rank"] = rank
    return ranked


def product_identity_key(item: dict) -> str:
    if item.get("product_codes"):
        return "|".join(item["product_codes"])
    if item.get("source"):
        return item["source"]
    if item.get("chip_models"):
        return "|".join(item["chip_models"])
    return item["chunk_id"]


def unique_list_values(items: list[dict], key: str) -> list[str]:
    seen = set()
    values = []
    for item in items:
        for value in item.get(key, []):
            normalized = str(value).strip()
            if not normalized:
                continue
            lookup_key = normalized.upper()
            if lookup_key in seen:
                continue
            seen.add(lookup_key)
            values.append(normalized)
    return values


def select_final_primary_results(
    question: str,
    reranked_results: list[dict],
    args: argparse.Namespace,
    known_vendors: list[str] | None = None,
) -> list[dict]:
    requested_vendors = set(detect_requested_vendors(question, known_vendors=known_vendors))
    requested_chips = set(detect_requested_chip_models(question))
    if not (requested_vendors or requested_chips) or not asks_for_product_list(question):
        return reranked_results[: args.top_k]

    limit = max(args.top_k, getattr(args, "product_list_top_k", 20))
    selected = []
    seen_products = set()

    for item in reranked_results:
        if requested_chips and not requested_chips.intersection(item.get("chip_models", [])):
            continue
        if requested_vendors and not requested_vendors.intersection(item.get("vendors", [])):
            continue
        key = product_identity_key(item)
        if key in seen_products:
            continue
        selected.append(item)
        seen_products.add(key)
        if len(selected) >= limit:
            break

    return selected or reranked_results[: args.top_k]


def token_spans(text: str) -> list[dict]:
    return [
        {"start": match.start(), "end": match.end()}
        for match in re.finditer(r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*|[\u4e00-\u9fff]|[^\s]", text)
    ]


def compact_text_window(text: str, start: int, end: int, max_tokens: int) -> str:
    spans = token_spans(text)
    if not spans or len(spans) <= max_tokens:
        return text.strip()

    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    first_token = 0
    while first_token < len(spans) and spans[first_token]["end"] <= start:
        first_token += 1

    last_token = first_token
    while last_token < len(spans) and spans[last_token]["start"] < end:
        last_token += 1

    core_count = max(last_token - first_token, 1)
    budget = max(max_tokens, core_count)
    remaining = max(budget - core_count, 0)
    left_extra = remaining // 2
    right_extra = remaining - left_extra
    window_start = max(0, first_token - left_extra)
    window_end = min(len(spans), last_token + right_extra)

    if window_end - window_start < budget:
        shortage = budget - (window_end - window_start)
        window_start = max(0, window_start - shortage)
        window_end = min(len(spans), window_end + shortage)

    char_start = spans[window_start]["start"]
    char_end = spans[window_end - 1]["end"]
    prefix = "[...]\n" if window_start > 0 else ""
    suffix = "\n[...]" if window_end < len(spans) else ""
    return f"{prefix}{text[char_start:char_end].strip()}{suffix}"


def compact_parent_context(seed: dict, child_items: list[dict], max_tokens: int) -> str:
    parent_text = seed.get("parent_text") or seed.get("text", "")
    parent_start = seed.get("parent_char_start", seed.get("char_start", 0))
    ranges = []

    for item in child_items:
        if item.get("char_start") is None or item.get("char_end") is None:
            continue
        ranges.append(
            (
                max(0, item["char_start"] - parent_start),
                max(0, item["char_end"] - parent_start),
            )
        )

    if not parent_text:
        return "\n\n".join(item.get("text", "") for item in child_items if item.get("text"))
    if not ranges:
        return parent_text.strip()

    start = min(item[0] for item in ranges)
    end = max(item[1] for item in ranges)
    return compact_text_window(parent_text, start, end, max_tokens=max_tokens)


def heuristic_rerank_score(question: str, item: dict, known_vendors: list[str] | None = None) -> float:
    query_tokens = set(tokenize(question))
    text_tokens = tokenize(build_rerank_text(item))
    overlap = len(query_tokens.intersection(text_tokens))
    metadata_bonus = 0.0

    searchable_metadata = " ".join(
        [
            item.get("section", ""),
            item.get("title", ""),
            item.get("product", ""),
            " ".join(item.get("product_codes", [])),
            " ".join(item.get("chip_models", [])),
            " ".join(item.get("vendors", [])),
            item.get("version", ""),
            " ".join(item.get("tags", [])),
            item.get("table_context", ""),
            item.get("list_structure", ""),
        ]
    ).lower()
    metadata_bonus = sum(1 for token in query_tokens if token and token in searchable_metadata)

    structural_bonus = 0.0
    if item.get("table_context"):
        structural_bonus += 0.2
    if item.get("list_structure"):
        structural_bonus += 0.1
    if item.get("heading_level"):
        structural_bonus += 0.05

    return (
        float(item.get("hybrid_score", item.get("dense_score", 0.0)))
        + metadata_query_bonus(question, item, known_vendors=known_vendors)
        + overlap * 0.15
        + metadata_bonus * 0.1
        + structural_bonus
    )


def rerank_results(
    question: str,
    candidates: list[dict],
    reranker_model: str,
    known_vendors: list[str] | None = None,
) -> list[dict]:
    if not candidates:
        return []

    try:
        reranker = get_cross_encoder(reranker_model)
        pairs = [(question, build_rerank_text(item)) for item in candidates]
        scores = reranker.predict(pairs)
        reranked = []
        for item, score in zip(candidates, scores):
            updated = item.copy()
            updated["rerank_score"] = float(score) + metadata_query_bonus(
                question,
                updated,
                known_vendors=known_vendors,
            )
            reranked.append(updated)
        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    except Exception:
        reranked = []
        for item in candidates:
            updated = item.copy()
            updated["rerank_score"] = heuristic_rerank_score(question, updated, known_vendors=known_vendors)
            reranked.append(updated)
        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

    for rank, item in enumerate(reranked, start=1):
        item["rerank_rank"] = rank
    return reranked


def retrieve_stages(question: str, args: argparse.Namespace) -> dict[str, list[dict]]:
    records, metadata_by_id = load_metadata()
    known_vendors = known_vendors_from_records(records)

    if args.backend == "faiss":
        dense_results = search_faiss(question, args.dense_top_k, args.embed_model, records)
    else:
        dense_results = search_chroma(question, args.dense_top_k, args.embed_model, metadata_by_id)

    if args.disable_hybrid:
        hybrid_results = dense_results[: args.hybrid_top_k]
    else:
        lexical_results = lexical_search(question, records, args.dense_top_k)
        hybrid_results = reciprocal_rank_fusion(dense_results, lexical_results, args.hybrid_top_k)

    hybrid_results = add_metadata_vendor_candidates(question, records, hybrid_results)
    rerank_candidates = hybrid_results

    if args.disable_rerank:
        reranked_results = hybrid_results
        for rank, item in enumerate(reranked_results, start=1):
            item["rerank_score"] = item.get("hybrid_score", item.get("dense_score", 0.0))
            item["rerank_rank"] = rank
    else:
        reranked_results = rerank_results(
            question,
            rerank_candidates,
            args.reranker_model,
            known_vendors=known_vendors,
        )

    final_primary_results = select_final_primary_results(
        question,
        reranked_results,
        args,
        known_vendors=known_vendors,
    )
    final_seed_results = expand_adjacent_chunks(final_primary_results, metadata_by_id, args.adjacent_window)

    final_results = materialize_parent_contexts(final_seed_results, getattr(args, "parent_context_tokens", 700))
    return {
        "dense": dense_results,
        "hybrid": hybrid_results,
        "rerank": reranked_results,
        "final": final_results,
    }


def materialize_parent_contexts(results: list[dict], parent_context_tokens: int = 700) -> list[dict]:
    contexts = []
    by_parent_id: dict[str, dict] = {}

    for item in results:
        parent_id = item.get("parent_id") or item["chunk_id"]
        if parent_id in by_parent_id:
            existing = by_parent_id[parent_id]
            existing["_child_items"].append(item)
            existing["rerank_score"] = max(existing.get("rerank_score", 0.0), item.get("rerank_score", 0.0))
            continue

        context = item.copy()
        context["context_id"] = parent_id
        context["retrieval_role"] = "compact_parent_context"
        context["_child_items"] = [item]

        by_parent_id[parent_id] = context
        contexts.append(context)

    for context in contexts:
        child_items = sorted(context.pop("_child_items"), key=lambda item: item.get("char_start", 0))
        context["matched_child_ids"] = [item["chunk_id"] for item in child_items]
        context["matched_child_texts"] = [item.get("text", "") for item in child_items if item.get("text")]
        context["child_text"] = "\n\n".join(context["matched_child_texts"])
        context["product_codes"] = unique_list_values(child_items, "product_codes")
        context["text_product_codes"] = unique_list_values(child_items, "text_product_codes")
        context["source_product_codes"] = unique_list_values(child_items, "source_product_codes")
        context["chip_models"] = unique_list_values(child_items, "chip_models")
        context["vendors"] = unique_list_values(child_items, "vendors")
        if context.get("chip_models") and ("ST" in context.get("vendors", [])) and context["chip_models"][0].startswith("STM32"):
            context["product"] = f"ST {context['chip_models'][0]}"
        elif context.get("chip_models"):
            context["product"] = context["chip_models"][0]
        elif context.get("product_codes"):
            context["product"] = context["product_codes"][0]
        context["text"] = compact_parent_context(context, child_items, max(parent_context_tokens, 1))
        context["page"] = min(item.get("page", context.get("page", 0)) for item in child_items)
        context["start_page"] = min(item.get("start_page", item.get("page", context["page"])) for item in child_items)
        context["end_page"] = max(item.get("end_page", item.get("page", context["page"])) for item in child_items)
        context["char_start"] = min(item.get("char_start", context.get("char_start", 0)) for item in child_items)
        context["char_end"] = max(item.get("char_end", context.get("char_end", 0)) for item in child_items)

    return contexts


def expand_adjacent_chunks(results: list[dict], metadata_by_id: dict[str, dict], window: int) -> list[dict]:
    if window <= 0:
        return results

    expanded = []
    seen = set()

    for item in results:
        if item["chunk_id"] not in seen:
            expanded.append(item)
            seen.add(item["chunk_id"])

        current = item
        prev_ids = []
        for _ in range(window):
            prev_chunk_id = current.get("prev_chunk_id")
            if not prev_chunk_id or prev_chunk_id not in metadata_by_id or prev_chunk_id in seen:
                break
            prev_ids.append(prev_chunk_id)
            current = metadata_by_id[prev_chunk_id]

        for prev_chunk_id in reversed(prev_ids):
            adjacent = metadata_by_id[prev_chunk_id].copy()
            adjacent["is_adjacent"] = True
            adjacent["adjacent_to"] = item["chunk_id"]
            adjacent.setdefault("rerank_score", item.get("rerank_score", 0.0) - 0.001)
            expanded.append(adjacent)
            seen.add(prev_chunk_id)

        current = item
        for _ in range(window):
            next_chunk_id = current.get("next_chunk_id")
            if not next_chunk_id or next_chunk_id not in metadata_by_id or next_chunk_id in seen:
                break
            adjacent = metadata_by_id[next_chunk_id].copy()
            adjacent["is_adjacent"] = True
            adjacent["adjacent_to"] = item["chunk_id"]
            adjacent.setdefault("rerank_score", item.get("rerank_score", 0.0) - 0.001)
            expanded.append(adjacent)
            seen.add(next_chunk_id)
            current = metadata_by_id[next_chunk_id]

    return expanded


def retrieve(question: str, args: argparse.Namespace) -> list[dict]:
    return retrieve_stages(question, args)["final"]


def print_results(results: list[dict]) -> None:
    print("\nRetrieved contexts:\n")
    for idx, item in enumerate(results, start=1):
        start_page = item.get("start_page", item["page"])
        end_page = item.get("end_page", item["page"])
        page_label = start_page if start_page == end_page else f"{start_page}-{end_page}"
        section = item.get("section") or item.get("title") or ""
        section_label = f" section={section}" if section else ""
        product_label = f" product={item['product']}" if item.get("product") else ""
        codes_label = f" product_codes={','.join(item['product_codes'])}" if item.get("product_codes") else ""
        text_codes_label = f" text_product_codes={','.join(item['text_product_codes'])}" if item.get("text_product_codes") else ""
        source_codes_label = f" source_product_codes={','.join(item['source_product_codes'])}" if item.get("source_product_codes") else ""
        chips_label = f" chip_models={','.join(item['chip_models'])}" if item.get("chip_models") else ""
        vendors_label = f" vendors={','.join(item['vendors'])}" if item.get("vendors") else ""
        version_label = f" version={item['version']}" if item.get("version") else ""
        adjacent_label = " adjacent=true" if item.get("is_adjacent") else ""
        print(
            f"[{idx}] rerank={item.get('rerank_score', 0.0):.4f} "
            f"hybrid={item.get('hybrid_score', 0.0):.4f} "
            f"dense={item.get('dense_score', 0.0):.4f} "
            f"lexical={item.get('lexical_score', 0.0):.4f} "
            f"source={item['source']} page={page_label}{section_label}{product_label}"
            f"{codes_label}{text_codes_label}{source_codes_label}{chips_label}{vendors_label}{version_label}{adjacent_label}"
        )
        print(item["text"])
        print()


DEBUG_RESULT_KEYS = [
    "chunk_id",
    "context_id",
    "parent_id",
    "prev_chunk_id",
    "next_chunk_id",
    "matched_child_ids",
    "retrieval_role",
    "is_adjacent",
    "adjacent_to",
    "dense_rank",
    "dense_score",
    "lexical_rank",
    "lexical_score",
    "hybrid_rank",
    "hybrid_score",
    "rerank_rank",
    "rerank_score",
    "metadata_vendor_match",
    "source",
    "page",
    "start_page",
    "end_page",
    "char_start",
    "char_end",
    "section",
    "title",
    "heading_level",
    "product",
    "product_codes",
    "text_product_codes",
    "source_product_codes",
    "chip_models",
    "vendors",
    "version",
    "doc_type",
    "tags",
    "table_context",
    "list_structure",
    "specs",
    "matched_child_texts",
    "child_text",
    "text",
]


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def debug_result_record(item: dict) -> dict:
    return {
        key: json_safe(item[key])
        for key in DEBUG_RESULT_KEYS
        if key in item and item[key] not in (None, [], {})
    }


def debug_args_record(args: argparse.Namespace) -> dict:
    return {
        key: json_safe(value)
        for key, value in vars(args).items()
        if key not in {"question"}
    }


def write_retrieval_debug_file(
    question: str,
    args: argparse.Namespace,
    stages: dict[str, list[dict]],
    structured_plan: dict,
    filtered_products: list[dict] | None,
) -> Path:
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = debug_dir / f"retrieval_{timestamp}.json"

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "args": debug_args_record(args),
        "stages": {
            stage_name: [debug_result_record(item) for item in results]
            for stage_name, results in stages.items()
        },
        "structured_filter": {
            "plan": json_safe(structured_plan),
            "filtered_products": json_safe(filtered_products or []),
        },
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    latest_path = debug_dir / "retrieval_latest.json"
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return path


def build_structured_filtered_products(
    question: str,
    plan: dict,
    stages: dict[str, list[dict]],
    limit: int,
) -> tuple[list[dict], str]:
    records, _ = load_metadata()
    products = build_product_specs(records)
    matching_products = [product for product in products if product_matches_plan(product, plan)]
    retrieval_order = stages.get("rerank", []) + stages.get("hybrid", []) + stages.get("dense", [])
    matching_products = sort_products_by_retrieval(matching_products, retrieval_order)
    matching_products = matching_products[: max(limit, 1)]
    structured_context = format_structured_context(plan, matching_products, max_products=limit)
    return matching_products, structured_context


def print_structured_results(plan: dict, products: list[dict]) -> None:
    print("\nStructured specs filter:\n")
    for condition in plan.get("conditions", []):
        print(f"- {condition['field']} {condition['op']} {condition['value']}")

    if not products:
        print("\nNo product passed all structured conditions.\n")
        return

    fields = []
    for condition in plan.get("conditions", []):
        field = condition["field"]
        if field != "vendor" and field not in fields:
            fields.append(field)
    if "cpu_soc" not in fields:
        fields.insert(0, "cpu_soc")

    print("\nFiltered products:\n")
    for idx, product in enumerate(products, start=1):
        print(f"[{idx}] {format_product_spec_line(product, fields)}")
    print()


def answer_with_openai(question: str, results: list[dict], chat_model: str) -> str:
    client = ensure_openai_client()
    response = client.responses.create(model=chat_model, input=build_prompt(question, results))
    return response.output_text


def answer_prompt_with_openai(prompt: str, chat_model: str) -> str:
    client = ensure_openai_client()
    response = client.responses.create(model=chat_model, input=prompt)
    return response.output_text


def answer_with_ollama(
    question: str,
    results: list[dict],
    chat_model: str,
    think: str,
    ollama_base_url: str,
    ollama_timeout: int,
) -> str:
    return call_ollama_chat(
        prompt=build_prompt(question, results),
        model=chat_model,
        think=False if think == "false" else (True if think == "true" else think),
        base_url=ollama_base_url,
        timeout=ollama_timeout,
    )


def answer_prompt_with_ollama(
    prompt: str,
    chat_model: str,
    think: str,
    ollama_base_url: str,
    ollama_timeout: int,
) -> str:
    return call_ollama_chat(
        prompt=prompt,
        model=chat_model,
        think=False if think == "false" else (True if think == "true" else think),
        base_url=ollama_base_url,
        timeout=ollama_timeout,
    )


def select_ollama_model(base_url: str) -> str:
    models = list_ollama_models(base_url=base_url)
    if not models:
        raise RuntimeError("No Ollama models found. Install one with `ollama pull <model>` first.")

    if len(models) == 1:
        print(f"Using the only local Ollama model: {models[0]}\n")
        return models[0]

    print("Local Ollama models:")
    for idx, model in enumerate(models, start=1):
        print(f"{idx}. {model}")

    while True:
        choice = input("Select a model number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            selected_model = models[int(choice) - 1]
            print(f"Selected model: {selected_model}\n")
            return selected_model

        print(f"Please enter a number from 1 to {len(models)}.")


def answer_question(question: str, args: argparse.Namespace, chat_model: str | None = None) -> None:
    stages = retrieve_stages(question, args)
    results = stages["final"]
    if not args.no_print_results:
        print_results(results)

    records, _ = load_metadata()
    known_vendors = known_vendors_from_records(records)
    structured_plan = (
        {"is_structured": False, "conditions": []}
        if args.disable_structured_filter
        else plan_specs_query(question, known_vendors=known_vendors)
    )
    structured_context = ""
    filtered_products = []
    if structured_plan.get("is_structured"):
        filtered_products, structured_context = build_structured_filtered_products(
            question,
            structured_plan,
            stages,
            args.structured_product_limit,
        )
        if not args.no_print_results:
            print_structured_results(structured_plan, filtered_products)

    if args.debug_retrieval:
        debug_path = write_retrieval_debug_file(question, args, stages, structured_plan, filtered_products)
        print(f"Retrieval debug written to: {debug_path}")

    if args.retrieval_only:
        return

    try:
        if args.llm_provider == "openai":
            resolved_chat_model = chat_model or args.chat_model or DEFAULT_OPENAI_CHAT_MODEL
            print(f"Calling OpenAI for final answer with model `{resolved_chat_model}`...\n")
            if structured_plan.get("is_structured"):
                answer = answer_prompt_with_openai(
                    build_structured_prompt(question, structured_context),
                    resolved_chat_model,
                )
            else:
                answer = answer_with_openai(question, results, resolved_chat_model)
        else:
            resolved_chat_model = chat_model or args.chat_model or select_ollama_model(args.ollama_base_url)
            print(
                f"Calling Ollama for final answer with model `{resolved_chat_model}` "
                f"at `{args.ollama_base_url}` with think=`{args.think}`...\n"
            )
            if structured_plan.get("is_structured"):
                answer = answer_prompt_with_ollama(
                    build_structured_prompt(question, structured_context),
                    resolved_chat_model,
                    args.think,
                    args.ollama_base_url,
                    args.ollama_timeout,
                )
            else:
                answer = answer_with_ollama(
                    question,
                    results,
                    resolved_chat_model,
                    args.think,
                    args.ollama_base_url,
                    args.ollama_timeout,
                )
    except (ConnectionError, EnvironmentError, ImportError, RuntimeError, TimeoutError) as exc:
        print(f"LLM answer skipped: {exc}")
        return

    print("Answer:\n")
    print(answer)


def resolve_chat_model(args: argparse.Namespace) -> str | None:
    if args.retrieval_only:
        return None
    if args.llm_provider == "openai":
        return args.chat_model or DEFAULT_OPENAI_CHAT_MODEL
    return args.chat_model or select_ollama_model(args.ollama_base_url)


def interactive_loop(args: argparse.Namespace) -> None:
    chat_model = resolve_chat_model(args)
    print("Interactive RAG CLI. Type `exit`, `quit`, or `q` to stop.\n")

    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            return

        print()
        answer_question(question, args, chat_model=chat_model)
        print()


def main() -> None:
    args = parse_args()

    if args.interactive or not args.question:
        interactive_loop(args)
        return

    answer_question(args.question, args)


if __name__ == "__main__":
    main()

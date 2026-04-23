import argparse
import math
import re
from collections import Counter

from rag_utils import (
    CHROMA_DIR,
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OPENAI_CHAT_MODEL,
    DEFAULT_RERANK_MODEL,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    build_prompt,
    call_ollama_chat,
    embed_texts,
    ensure_openai_client,
    get_cross_encoder,
    list_ollama_models,
    load_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--backend", choices=["faiss", "chroma"], default="faiss", help="Vector store backend")
    parser.add_argument("--top-k", type=int, default=5, help="Final number of primary chunks after reranking")
    parser.add_argument("--dense-top-k", type=int, default=20, help="Number of dense retrieval candidates")
    parser.add_argument("--hybrid-top-k", type=int, default=20, help="Number of candidates kept after hybrid merge")
    parser.add_argument("--adjacent-window", type=int, default=1, help="How many adjacent chunks to include per selected chunk")
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

    document_tokens = [tokenize(record.get("text", "")) for record in records]
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


def heuristic_rerank_score(question: str, item: dict) -> float:
    query_tokens = set(tokenize(question))
    text_tokens = tokenize(item.get("text", ""))
    overlap = len(query_tokens.intersection(text_tokens))
    metadata_bonus = 0.0

    searchable_metadata = " ".join(
        [
            item.get("section", ""),
            item.get("title", ""),
            item.get("product", ""),
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

    return float(item.get("hybrid_score", item.get("dense_score", 0.0))) + overlap * 0.15 + metadata_bonus * 0.1 + structural_bonus


def rerank_results(question: str, candidates: list[dict], reranker_model: str) -> list[dict]:
    if not candidates:
        return []

    try:
        reranker = get_cross_encoder(reranker_model)
        pairs = [(question, item["text"]) for item in candidates]
        scores = reranker.predict(pairs)
        reranked = []
        for item, score in zip(candidates, scores):
            updated = item.copy()
            updated["rerank_score"] = float(score)
            reranked.append(updated)
        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    except Exception:
        reranked = []
        for item in candidates:
            updated = item.copy()
            updated["rerank_score"] = heuristic_rerank_score(question, updated)
            reranked.append(updated)
        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

    for rank, item in enumerate(reranked, start=1):
        item["rerank_rank"] = rank
    return reranked


def retrieve_stages(question: str, args: argparse.Namespace) -> dict[str, list[dict]]:
    records, metadata_by_id = load_metadata()

    if args.backend == "faiss":
        dense_results = search_faiss(question, args.dense_top_k, args.embed_model, records)
    else:
        dense_results = search_chroma(question, args.dense_top_k, args.embed_model, metadata_by_id)

    if args.disable_hybrid:
        hybrid_results = dense_results[: args.hybrid_top_k]
    else:
        lexical_results = lexical_search(question, records, args.dense_top_k)
        hybrid_results = reciprocal_rank_fusion(dense_results, lexical_results, args.hybrid_top_k)

    if args.disable_rerank:
        reranked_results = hybrid_results[: args.top_k]
        for rank, item in enumerate(reranked_results, start=1):
            item["rerank_score"] = item.get("hybrid_score", item.get("dense_score", 0.0))
            item["rerank_rank"] = rank
    else:
        reranked_results = rerank_results(question, hybrid_results, args.reranker_model)

    final_results = expand_adjacent_chunks(reranked_results[: args.top_k], metadata_by_id, args.adjacent_window)
    return {
        "dense": dense_results,
        "hybrid": hybrid_results,
        "rerank": reranked_results,
        "final": final_results,
    }


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
        version_label = f" version={item['version']}" if item.get("version") else ""
        adjacent_label = " adjacent=true" if item.get("is_adjacent") else ""
        print(
            f"[{idx}] rerank={item.get('rerank_score', 0.0):.4f} "
            f"hybrid={item.get('hybrid_score', 0.0):.4f} "
            f"dense={item.get('dense_score', 0.0):.4f} "
            f"lexical={item.get('lexical_score', 0.0):.4f} "
            f"source={item['source']} page={page_label}{section_label}{product_label}{version_label}{adjacent_label}"
        )
        print(item["text"])
        print()


def answer_with_openai(question: str, results: list[dict], chat_model: str) -> str:
    client = ensure_openai_client()
    response = client.responses.create(model=chat_model, input=build_prompt(question, results))
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


def main() -> None:
    args = parse_args()
    results = retrieve(args.question, args)
    print_results(results)

    if args.retrieval_only:
        return

    try:
        if args.llm_provider == "openai":
            chat_model = args.chat_model or DEFAULT_OPENAI_CHAT_MODEL
            print(f"Calling OpenAI for final answer with model `{chat_model}`...\n")
            answer = answer_with_openai(args.question, results, chat_model)
        else:
            chat_model = args.chat_model or select_ollama_model(args.ollama_base_url)
            print(
                f"Calling Ollama for final answer with model `{chat_model}` "
                f"at `{args.ollama_base_url}` with think=`{args.think}`...\n"
            )
            answer = answer_with_ollama(
                args.question,
                results,
                chat_model,
                args.think,
                args.ollama_base_url,
                args.ollama_timeout,
            )
    except (ConnectionError, EnvironmentError, ImportError, RuntimeError, TimeoutError) as exc:
        print(f"LLM answer skipped: {exc}")
        return

    print("Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()

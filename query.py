import argparse

from rag_utils import (
    CHROMA_DIR,
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OPENAI_CHAT_MODEL,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    build_prompt,
    call_ollama_chat,
    embed_texts,
    ensure_openai_client,
    list_ollama_models,
    load_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--backend", choices=["faiss", "chroma"], default="faiss", help="Vector store backend")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved chunks")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding model name")
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


def search_faiss(question: str, top_k: int, embed_model: str) -> list[dict]:
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("Missing package: faiss-cpu. Install with `pip install faiss-cpu`.") from exc

    if not FAISS_INDEX_PATH.exists() or not FAISS_METADATA_PATH.exists():
        raise FileNotFoundError("Run `python3 index.py --backend faiss` before querying.")

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    metadata = load_jsonl(FAISS_METADATA_PATH)
    query_vector = embed_texts([question], model_name=embed_model)
    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = metadata[idx].copy()
        item["score"] = float(score)
        results.append(item)
    return results


def search_chroma(question: str, top_k: int, embed_model: str) -> list[dict]:
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
    ids = response["ids"][0]
    documents = response["documents"][0]
    metadatas = response["metadatas"][0]
    distances = response["distances"][0]

    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        results.append(
            {
                "chunk_id": chunk_id,
                "source": metadata["source"],
                "page": metadata["page"],
                "start_page": metadata.get("start_page", metadata["page"]),
                "end_page": metadata.get("end_page", metadata["page"]),
                "chunk_index": metadata["chunk_index"],
                "section": metadata.get("section", ""),
                "title": metadata.get("title", ""),
                "doc_type": metadata.get("doc_type", ""),
                "language": metadata.get("language", ""),
                "text": document,
                "score": float(distance),
            }
        )
    return results


def print_results(results: list[dict]) -> None:
    print("\nRetrieved contexts:\n")
    for idx, item in enumerate(results, start=1):
        start_page = item.get("start_page", item["page"])
        end_page = item.get("end_page", item["page"])
        page_label = start_page if start_page == end_page else f"{start_page}-{end_page}"
        section = item.get("section") or item.get("title") or ""
        section_label = f" section={section}" if section else ""
        print(f"[{idx}] score={item['score']:.4f} source={item['source']} page={page_label}{section_label}")
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

    if args.backend == "faiss":
        results = search_faiss(args.question, args.top_k, args.embed_model)
    else:
        results = search_chroma(args.question, args.top_k, args.embed_model)

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

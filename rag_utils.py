from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_DIR = Path.home() / "rag-project"
DATA_DIR = PROJECT_DIR / "data"
CHUNKS_PATH = DATA_DIR / "pdf_chunks.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
METADATA_PATH = DATA_DIR / "embedding_metadata.jsonl"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
FAISS_METADATA_PATH = DATA_DIR / "faiss_metadata.jsonl"
CHROMA_DIR = DATA_DIR / "chroma_db"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_OPENAI_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_CHAT_MODEL = "qwen3.5:9b"
DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_embedding_inputs(path: Path = CHUNKS_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")
    return load_jsonl(path)


def get_sentence_transformer(model_name: str = DEFAULT_EMBED_MODEL):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Missing package: sentence-transformers. Install with `pip install sentence-transformers`."
        ) from exc

    return SentenceTransformer(model_name)


def get_cross_encoder(model_name: str = DEFAULT_RERANK_MODEL):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "Missing package: sentence-transformers. Install with `pip install sentence-transformers`."
        ) from exc

    return CrossEncoder(model_name)


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 32,
) -> np.ndarray:
    import numpy as np

    model = get_sentence_transformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype="float32")


def save_embeddings(embeddings: np.ndarray, path: Path = EMBEDDINGS_PATH) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)


def load_embeddings(path: Path = EMBEDDINGS_PATH) -> np.ndarray:
    import numpy as np

    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    return np.load(path)


def ensure_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Missing package: openai. Install with `pip install openai`.") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    return OpenAI(api_key=api_key)


def list_ollama_models(
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout: int = 10,
) -> list[str]:
    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/api/tags",
        headers={"Content-Type": "application/json"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Cannot reach Ollama at {base_url}. Make sure `ollama serve` is running."
        ) from exc

    models = body.get("models", [])
    return [model["name"] for model in models if model.get("name")]


def call_ollama_chat(
    prompt: str,
    model: str = DEFAULT_CHAT_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout: int = 120,
    think: bool | str = False,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "think": think,
    }
    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Cannot reach Ollama at {base_url}. Make sure `ollama serve` is running."
        ) from exc
    except TimeoutError as exc:
        raise TimeoutError(
            f"Ollama did not respond within {timeout} seconds. Try a smaller model or increase the timeout."
        ) from exc

    message = body.get("message", {})
    content = message.get("content", "").strip()
    if not content:
        raise RuntimeError("Ollama returned an empty response.")
    return content


def format_contexts(results: list[dict]) -> str:
    blocks = []
    for idx, item in enumerate(results, start=1):
        start_page = item.get("start_page", item["page"])
        end_page = item.get("end_page", item["page"])
        page_label = start_page if start_page == end_page else f"{start_page}-{end_page}"
        section = item.get("section") or item.get("title") or ""
        metadata_lines = [
            f"[Context {idx}]",
            f"source: {item['source']}",
            f"page: {page_label}",
        ]
        if section:
            metadata_lines.append(f"section: {section}")
        if item.get("product"):
            metadata_lines.append(f"product: {item['product']}")
        if item.get("version"):
            metadata_lines.append(f"version: {item['version']}")
        if item.get("heading_level") is not None:
            metadata_lines.append(f"heading_level: {item['heading_level']}")
        if item.get("list_structure"):
            metadata_lines.append(f"list_structure: {item['list_structure']}")
        if item.get("table_context"):
            metadata_lines.append(f"table_context: {item['table_context']}")
        if item.get("char_start") is not None and item.get("char_end") is not None:
            metadata_lines.append(f"char_range: {item['char_start']}-{item['char_end']}")
        if item.get("is_adjacent"):
            metadata_lines.append("retrieval_role: adjacent_supporting_chunk")
        metadata_lines.append(f"text: {item['text']}")
        blocks.append(
            "\n".join(metadata_lines)
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, results: list[dict]) -> str:
    context_text = format_contexts(results)
    return (
        "You are a retrieval-augmented assistant. Answer only from the provided context. "
        "You must synthesize information across multiple contexts when needed instead of relying on a single chunk. "
        "If the evidence is incomplete or conflicting, say so explicitly.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_text}\n\n"
        "Instructions:\n"
        "1. Read all contexts before answering.\n"
        "2. Merge adjacent or related contexts when the answer spans multiple chunks.\n"
        "3. Do not invent facts outside the context.\n"
        "4. Prefer the most specific evidence such as tables, lists, versions, and product names.\n"
        "5. If the answer cannot be fully supported, state what is missing.\n\n"
        "Output in Traditional Chinese with this structure:\n"
        "Answer: <direct answer>\n"
        "Evidence:\n"
        "- <source filename + page + concise supporting fact>\n"
        "- <source filename + page + concise supporting fact>\n"
        "Notes: <optional caveat or missing information>"
    )

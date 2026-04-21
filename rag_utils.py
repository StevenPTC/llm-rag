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
        blocks.append(
            "\n".join(
                [
                    f"[Context {idx}]",
                    f"source: {item['source']}",
                    f"page: {item['page']}",
                    f"text: {item['text']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, results: list[dict]) -> str:
    context_text = format_contexts(results)
    return (
        "You are a helpful RAG assistant. Answer only from the provided context. "
        "If the context is insufficient, say you do not know.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_text}\n\n"
        "Please answer in Traditional Chinese and cite the source filename and page number when possible."
    )

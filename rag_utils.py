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


# RAG 階段：共用設定
# 路徑常數集中管理，讓 ingestion、embedding、index 與 query 使用同一批 artifacts。
# 這可降低「向量檔、metadata、chunk 檔不一致」造成 retrieval 錯位的風險。


def _metadata_value(value: object) -> str:
    # RAG 階段：Embedding / LLM 推論
    # 將 metadata 安全轉成文字表示。Embedding 與 prompt 都需要可讀的欄位文字，
    # 但 None、list、dict 若直接轉字串會產生雜訊或格式不一致，因此在此統一處理。
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if item is None or item == []:
                continue
            if isinstance(item, list):
                item_text = ", ".join(str(entry).strip() for entry in item if str(entry).strip())
            else:
                item_text = str(item).strip()
            if item_text:
                parts.append(f"{key}={item_text}")
        return "; ".join(parts)
    return str(value).strip()


def build_metadata_aware_text(record: dict, content_key: str = "text") -> str:
    # RAG 階段：Embedding / Re-ranking
    # 不只嵌入 chunk 正文，也把 title、section、product、specs 等 metadata 放入文字。
    # 設計意圖是讓查詢「產品型號、vendor、規格」時，向量能捕捉結構化欄位的語意。
    fields = [
        ("Title", record.get("title")),
        ("Section", record.get("section")),
        ("Product", record.get("product")),
        ("Product codes", record.get("product_codes")),
        ("Text product codes", record.get("text_product_codes")),
        ("Source product codes", record.get("source_product_codes")),
        ("Chip models", record.get("chip_models")),
        ("Vendors", record.get("vendors")),
        ("Version", record.get("version")),
        ("Document type", record.get("doc_type")),
        ("Tags", record.get("tags")),
        ("Specs", record.get("specs")),
        ("Table context", record.get("table_context")),
        ("List structure", record.get("list_structure")),
    ]
    lines = [f"{label}: {value}" for label, raw_value in fields if (value := _metadata_value(raw_value))]
    content = _metadata_value(record.get(content_key))
    if content:
        lines.append(f"Content: {content}")
    return "\n".join(lines)


def build_embedding_text(record: dict) -> str:
    # RAG 階段：Embedding
    # Embedding 使用 metadata-aware text，讓向量同時代表 chunk 內容與其文件脈絡。
    return build_metadata_aware_text(record, content_key="text")


def build_rerank_text(record: dict) -> str:
    # RAG 階段：Re-ranking
    # Re-ranker 看到的文字與 embedding 一致，可避免第一階段召回和第二階段排序
    # 使用不同語意來源而產生排序偏差。
    return build_metadata_aware_text(record, content_key="text")


def load_jsonl(path: Path) -> list[dict]:
    # RAG 階段：資料前處理 / Retrieval
    # JSONL 讓每個 page/chunk/metadata record 可以逐行儲存與檢查，適合 RAG pipeline
    # 中間產物除錯，也避免單一大型 JSON 損壞時整批資料難以讀取。
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
    # RAG 階段：Embedding
    # 延遲 import 可讓只使用查詢或工具函式的情境不必立刻載入大型 ML 依賴。
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Missing package: sentence-transformers. Install with `pip install sentence-transformers`."
        ) from exc

    return SentenceTransformer(model_name)


def get_cross_encoder(model_name: str = DEFAULT_RERANK_MODEL):
    # RAG 階段：Re-ranking
    # CrossEncoder 用 question + chunk 成對評分，比單純向量相似度更能判斷「是否回答問題」。
    # 因其成本較高，所以只對候選 chunks 使用。
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
    # RAG 階段：Embedding
    # normalize_embeddings=True 讓向量長度固定，後續 FAISS 內積搜尋即可視為 cosine similarity。
    # float32 是向量索引常用格式，可降低記憶體與磁碟用量。
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
    # RAG 階段：LLM 推論
    # OpenAI client 僅在使用 OpenAI provider 時建立，避免本機 Ollama 流程要求不必要的 API key。
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
    # RAG 階段：LLM 推論
    # 查詢本機 Ollama 模型清單，讓 CLI 可以在沒有指定 chat model 時互動式選擇模型。
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
    # RAG 階段：LLM 推論
    # 這裡只負責送出已組好的 prompt。Retrieval 的約束與引用格式都在 prompt 建構階段完成，
    # 讓 provider 切換時不改變回答規則。
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
    # RAG 階段：LLM 推論
    # 將 retrieval 結果格式化成帶來源、頁碼、產品 metadata 的 evidence blocks。
    # LLM 需要這些欄位才能回答時附上可追溯證據，而不是只看裸文字。
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
        if item.get("product_codes"):
            metadata_lines.append(f"product_codes: {_metadata_value(item['product_codes'])}")
        if item.get("text_product_codes"):
            metadata_lines.append(f"text_product_codes: {_metadata_value(item['text_product_codes'])}")
        if item.get("source_product_codes"):
            metadata_lines.append(f"source_product_codes: {_metadata_value(item['source_product_codes'])}")
        if item.get("chip_models"):
            metadata_lines.append(f"chip_models: {_metadata_value(item['chip_models'])}")
        if item.get("vendors"):
            metadata_lines.append(f"vendors: {_metadata_value(item['vendors'])}")
        if item.get("version"):
            metadata_lines.append(f"version: {item['version']}")
        if item.get("specs"):
            metadata_lines.append(f"specs: {_metadata_value(item['specs'])}")
        if item.get("heading_level") is not None:
            metadata_lines.append(f"heading_level: {item['heading_level']}")
        if item.get("list_structure"):
            metadata_lines.append(f"list_structure: {item['list_structure']}")
        if item.get("table_context"):
            metadata_lines.append(f"table_context: {item['table_context']}")
        if item.get("char_start") is not None and item.get("char_end") is not None:
            metadata_lines.append(f"char_range: {item['char_start']}-{item['char_end']}")
        if item.get("context_id"):
            metadata_lines.append(f"context_id: {item['context_id']}")
        if item.get("matched_child_ids"):
            metadata_lines.append(f"matched_child_ids: {', '.join(item['matched_child_ids'])}")
        if item.get("retrieval_role"):
            metadata_lines.append(f"retrieval_role: {item['retrieval_role']}")
        elif item.get("is_adjacent"):
            metadata_lines.append("retrieval_role: adjacent_supporting_chunk")
        if item.get("child_text") and item.get("child_text") != item.get("text"):
            metadata_lines.append(f"matched_child_text: {item['child_text']}")
        metadata_lines.append(f"text: {item['text']}")
        blocks.append(
            "\n".join(metadata_lines)
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, results: list[dict]) -> str:
    # RAG 階段：LLM 推論
    # Prompt 明確要求只根據 context 回答、合併多個 chunks、指出證據不足。
    # 這是 RAG 的防幻覺邊界：檢索提供資料，LLM 負責整合與表達，不補外部事實。
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
        "5. If the question asks for product names or model names, extract only product/model names explicitly shown in the context. Do not answer unrelated specifications.\n"
        "6. If the answer cannot be fully supported, state what is missing.\n\n"
        "Output in Traditional Chinese with this structure:\n"
        "Answer: <direct answer>\n"
        "Evidence:\n"
        "- <source filename + page + concise supporting fact>\n"
        "- <source filename + page + concise supporting fact>\n"
        "Notes: <optional caveat or missing information>"
    )


def build_structured_prompt(question: str, structured_context: str) -> str:
    # RAG 階段：LLM 推論
    # 結構化規格查詢已由程式完成篩選，LLM 在此只做格式化與說明。
    # 這避免模型自行判斷數值條件，降低規格比較題的幻覺與算錯風險。
    return (
        "You are a retrieval-augmented assistant. The system has already converted the user's question "
        "into structured conditions and already filtered the product list with program logic. "
        "Do not decide whether a product qualifies; only format and explain the filtered products below.\n\n"
        f"Question: {question}\n\n"
        f"Structured context:\n{structured_context}\n\n"
        "Rules for comparison and recommendation questions:\n"
        "1. If the question asks to list products, return a concise product list from all provided structured context rows.\n"
        "2. List only products that are explicitly in 'Filtered matching products'.\n"
        "3. Do not recommend products that failed or are missing from the filtered list.\n"
        "4. Every product or recommendation must include the matching specification evidence.\n"
        "5. If no product passed the structured filter, say no clearly supported match was found.\n"
        "6. Do not invent specifications outside the structured context.\n\n"
        "Output in Traditional Chinese with this structure:\n"
        "Answer: <direct answer>\n"
        "Products:\n"
        "- <product>: <why it matches, with exact specs>\n"
        "Evidence:\n"
        "- <source filename + page + concise supporting fact>\n"
        "Notes: <optional caveat or missing information>"
    )

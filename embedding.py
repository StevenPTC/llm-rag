import argparse
from pathlib import Path

from rag_utils import (
    DEFAULT_EMBED_MODEL,
    EMBEDDINGS_PATH,
    METADATA_PATH,
    build_embedding_text,
    embed_texts,
    load_embedding_inputs,
    save_embeddings,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    # RAG 階段：Embedding
    # 讓模型名稱與 batch size 由 CLI 控制，方便在不改程式的情況下比較不同 embedding
    # 模型或依硬體記憶體調整批次大小。
    parser = argparse.ArgumentParser(description="Generate embeddings from pdf_chunks.jsonl")
    parser.add_argument("--input", default=None, help="Path to chunk JSONL file")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL, help="Embedding model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    return parser.parse_args()


def main() -> None:
    # RAG 階段：Embedding
    # 讀取前處理完成的 chunk JSONL，先把 text 與 metadata 組成 embedding 專用文字，
    # 再產生正規化向量。metadata 同步寫出，是為了確保向量索引中的第 N 筆可回查
    # 到同一筆 chunk 與來源證據。
    args = parse_args()
    input_path = Path(args.input) if args.input else None
    records = load_embedding_inputs(path=input_path) if input_path else load_embedding_inputs()
    texts = [build_embedding_text(record) for record in records]
    embeddings = embed_texts(texts, model_name=args.model, batch_size=args.batch_size)

    save_embeddings(embeddings, EMBEDDINGS_PATH)
    write_jsonl(records, METADATA_PATH)

    print(f"Embedded {len(records)} chunks")
    print(f"Saved embeddings to: {EMBEDDINGS_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")
    print(f"Embedding dimension: {embeddings.shape[1]}")


if __name__ == "__main__":
    main()

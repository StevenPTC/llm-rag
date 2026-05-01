import argparse
from pathlib import Path

from rag_utils import CHROMA_DIR, EMBEDDINGS_PATH, FAISS_INDEX_PATH, FAISS_METADATA_PATH, METADATA_PATH, load_embeddings, load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    # RAG 階段：Retrieval 索引建立
    # backend 參數讓同一批 embeddings 可切換 FAISS 或 Chroma。這有助於在本機速度、
    # 持久化能力與部署環境之間做取捨，而不改動 embedding 產製流程。
    parser = argparse.ArgumentParser(description="Build a vector index from embeddings")
    parser.add_argument("--backend", choices=["faiss", "chroma"], default="faiss", help="Vector store backend")
    return parser.parse_args()


def build_faiss_index() -> None:
    # RAG 階段：Retrieval 索引建立
    # FAISS IndexFlatIP 使用內積搜尋；前一階段已將 embeddings normalize，因此內積
    # 等價於 cosine similarity，適合做語意相似度排序。
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("Missing package: faiss-cpu. Install with `pip install faiss-cpu`.") from exc

    embeddings = load_embeddings(EMBEDDINGS_PATH)
    metadata = load_jsonl(METADATA_PATH)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    write_jsonl(metadata, FAISS_METADATA_PATH)

    print(f"Built FAISS index with {index.ntotal} vectors")
    print(f"Saved index to: {FAISS_INDEX_PATH}")
    print(f"Saved metadata to: {FAISS_METADATA_PATH}")


def build_chroma_index() -> None:
    # RAG 階段：Retrieval 索引建立
    # Chroma 版本把文件文字、metadata 與 embedding 一起持久化，適合需要較完整
    # 向量資料庫能力的情境；重建時先刪除舊 collection，避免重複 id 或殘留資料。
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError("Missing package: chromadb. Install with `pip install chromadb`.") from exc

    embeddings = load_embeddings(EMBEDDINGS_PATH)
    metadata = load_jsonl(METADATA_PATH)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="pdf_chunks")
    existing = collection.count()
    if existing:
        client.delete_collection(name="pdf_chunks")
        collection = client.get_or_create_collection(name="pdf_chunks")

    ids = [record["chunk_id"] for record in metadata]
    documents = [record["text"] for record in metadata]
    metadatas = [
        # Chroma metadata 只支援基本型別，因此 list 型欄位轉成逗號分隔字串。
        # 原始完整 metadata 仍保存在 JSONL；Chroma 這份主要用於查詢結果快速展示。
        {
            "source": record["source"],
            "page": record["page"],
            "start_page": record.get("start_page", record["page"]),
            "end_page": record.get("end_page", record["page"]),
            "char_start": record.get("char_start", 0),
            "char_end": record.get("char_end", 0),
            "chunk_index": record["chunk_index"],
            "chunk_role": record.get("chunk_role", ""),
            "token_count": record.get("token_count", 0),
            "section": record.get("section", ""),
            "title": record.get("title", ""),
            "doc_type": record.get("doc_type", ""),
            "product": record.get("product", ""),
            "product_codes": ",".join(record.get("product_codes", [])),
            "text_product_codes": ",".join(record.get("text_product_codes", [])),
            "source_product_codes": ",".join(record.get("source_product_codes", [])),
            "chip_models": ",".join(record.get("chip_models", [])),
            "vendors": ",".join(record.get("vendors", [])),
            "version": record.get("version", ""),
            "language": record.get("language", ""),
            "tags": ",".join(record.get("tags", [])),
            "file_path": record.get("file_path", ""),
            "heading_level": record.get("heading_level", 0),
            "list_structure": record.get("list_structure", ""),
            "table_context": record.get("table_context", ""),
            "prev_chunk_id": record.get("prev_chunk_id", ""),
            "next_chunk_id": record.get("next_chunk_id", ""),
            "parent_id": record.get("parent_id", ""),
            "parent_index": record.get("parent_index", 0),
            "parent_child_index": record.get("parent_child_index", 0),
            "parent_child_count": record.get("parent_child_count", 0),
            "parent_token_count": record.get("parent_token_count", 0),
            "parent_start_page": record.get("parent_start_page", record.get("start_page", record["page"])),
            "parent_end_page": record.get("parent_end_page", record.get("end_page", record["page"])),
            "parent_char_start": record.get("parent_char_start", record.get("char_start", 0)),
            "parent_char_end": record.get("parent_char_end", record.get("char_end", 0)),
        }
        for record in metadata
    ]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )

    print(f"Built Chroma collection with {collection.count()} vectors")
    print(f"Saved database to: {CHROMA_DIR}")


def main() -> None:
    # RAG 階段：Retrieval 索引建立
    # 索引建立依賴 embedding.py 先產生 embeddings 與 metadata；缺任一檔案就中止，
    # 避免建立出向量和 chunk metadata 對不上的索引。
    args = parse_args()
    Path(METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)

    if not Path(EMBEDDINGS_PATH).exists() or not Path(METADATA_PATH).exists():
        raise FileNotFoundError("Run `python3 embedding.py` before building the index.")

    if args.backend == "faiss":
        build_faiss_index()
    else:
        build_chroma_index()


if __name__ == "__main__":
    main()

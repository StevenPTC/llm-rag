import argparse
from pathlib import Path

from rag_utils import CHROMA_DIR, EMBEDDINGS_PATH, FAISS_INDEX_PATH, FAISS_METADATA_PATH, METADATA_PATH, load_embeddings, load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a vector index from embeddings")
    parser.add_argument("--backend", choices=["faiss", "chroma"], default="faiss", help="Vector store backend")
    return parser.parse_args()


def build_faiss_index() -> None:
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
        {
            "source": record["source"],
            "page": record["page"],
            "chunk_index": record["chunk_index"],
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

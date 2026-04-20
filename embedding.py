import argparse
from pathlib import Path

from rag_utils import (
    DEFAULT_EMBED_MODEL,
    EMBEDDINGS_PATH,
    METADATA_PATH,
    embed_texts,
    load_embedding_inputs,
    save_embeddings,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embeddings from pdf_chunks.jsonl")
    parser.add_argument("--input", default=None, help="Path to chunk JSONL file")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL, help="Embedding model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input) if args.input else None
    records = load_embedding_inputs(path=input_path) if input_path else load_embedding_inputs()
    texts = [record["text"] for record in records]
    embeddings = embed_texts(texts, model_name=args.model, batch_size=args.batch_size)

    save_embeddings(embeddings, EMBEDDINGS_PATH)
    write_jsonl(records, METADATA_PATH)

    print(f"Embedded {len(records)} chunks")
    print(f"Saved embeddings to: {EMBEDDINGS_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")
    print(f"Embedding dimension: {embeddings.shape[1]}")


if __name__ == "__main__":
    main()

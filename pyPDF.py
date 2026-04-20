import json
import re
from pathlib import Path

import fitz  # PyMuPDF


DOCS_DIR = Path.home() / "rag-project" / "docs"
OUTPUT_DIR = Path.home() / "rag-project" / "data"
RAW_OUTPUT = OUTPUT_DIR / "pdf_pages.jsonl"
CHUNK_OUTPUT = OUTPUT_DIR / "pdf_chunks.jsonl"


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        text = normalize_text(page.get_text("text"))
        if not text:
            continue

        pages.append(
            {
                "source": pdf_path.name,
                "page": i + 1,
                "text": text,
            }
        )

    return pages


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def build_chunks(pages: list[dict], chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    chunk_records = []

    for page in pages:
        chunks = chunk_text(page["text"], chunk_size=chunk_size, overlap=overlap)
        for idx, chunk in enumerate(chunks, start=1):
            chunk_records.append(
                {
                    "chunk_id": f'{page["source"]}-p{page["page"]}-c{idx}',
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": idx,
                    "text": chunk,
                }
            )

    return chunk_records


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {DOCS_DIR}")

    all_pages = []
    for pdf_file in pdf_files:
        all_pages.extend(extract_pdf_pages(pdf_file))

    all_chunks = build_chunks(all_pages)

    write_jsonl(all_pages, RAW_OUTPUT)
    write_jsonl(all_chunks, CHUNK_OUTPUT)

    print(f"Processed {len(pdf_files)} PDF files")
    print(f"Saved page-level text to: {RAW_OUTPUT}")
    print(f"Saved chunked text to: {CHUNK_OUTPUT}")
    print(f"Total pages: {len(all_pages)}")
    print(f"Total chunks: {len(all_chunks)}")


if __name__ == "__main__":
    main()

import json
import re
from pathlib import Path

import fitz  # PyMuPDF


DOCS_DIR = Path.home() / "rag-project" / "docs"
OUTPUT_DIR = Path.home() / "rag-project" / "data"
RAW_OUTPUT = OUTPUT_DIR / "pdf_pages.jsonl"
CHUNK_OUTPUT = OUTPUT_DIR / "pdf_chunks.jsonl"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
SPLIT_SEPARATORS = ["\n\n", "\n", "。", ".", " ", ""]
HEADER_FOOTER_MIN_PAGES = 2


SECTION_PATTERNS = [
    re.compile(r"^\s*(?:\d+(?:\.\d+)+[.)]?|\d+[.)])\s+.{3,80}$"),
    re.compile(r"^\s*(?:chapter|section)\s+\d+[:.)]?\s+.{3,80}$", re.IGNORECASE),
    re.compile(r"^\s*(?:hardware|software|features|specifications|dimensions|block diagram|i/o|display|memory|cpu)\s*$", re.IGNORECASE),
]


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    with fitz.open(pdf_path) as doc:
        raw_pages = [
            {
                "source": pdf_path.name,
                "file_path": str(pdf_path),
                "page": i + 1,
                "text": normalize_text(page.get_text("text")),
            }
            for i, page in enumerate(doc)
        ]

    repeated_lines = find_repeated_header_footer_lines(raw_pages)
    pages = []

    for page in raw_pages:
        text = remove_header_footer(page["text"], repeated_lines)
        if not text:
            continue

        page["text"] = text
        pages.append(page)

    return pages


def find_repeated_header_footer_lines(pages: list[dict]) -> set[str]:
    line_counts: dict[str, int] = {}

    for page in pages:
        lines = [line.strip() for line in page["text"].splitlines() if line.strip()]
        for line in lines[:2] + lines[-2:]:
            normalized = re.sub(r"\s+", " ", line)
            if normalized.isdigit() or len(normalized) < 3:
                continue
            line_counts[normalized] = line_counts.get(normalized, 0) + 1

    min_count = min(HEADER_FOOTER_MIN_PAGES, max(len(pages), 1))
    return {line for line, count in line_counts.items() if count >= min_count}


def remove_header_footer(text: str, repeated_lines: set[str]) -> str:
    cleaned_lines = []

    for line in text.splitlines():
        normalized = re.sub(r"\s+", " ", line.strip())
        if not normalized:
            cleaned_lines.append("")
            continue
        if normalized in repeated_lines:
            continue
        if re.fullmatch(r"(?:page\s*)?\d+\s*(?:/|of)\s*\d+", normalized, flags=re.IGNORECASE):
            continue
        if normalized.lower() in {"confidential", "company confidential"}:
            continue
        cleaned_lines.append(line)

    return normalize_text("\n".join(cleaned_lines))


def recursive_split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    separators: list[str] | None = None,
) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    separators = separators or SPLIT_SEPARATORS
    separator = separators[0]
    remaining_separators = separators[1:]

    if separator:
        splits = text.split(separator)
    else:
        splits = list(text)

    if separator and len(splits) == 1:
        return recursive_split_text(text, chunk_size, remaining_separators)

    chunks = []
    current = ""
    joiner = separator

    for split in splits:
        candidate = split if not current else f"{current}{joiner}{split}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current.strip():
            chunks.extend(split_long_text(current.strip(), chunk_size, remaining_separators))
        current = split

    if current.strip():
        chunks.extend(split_long_text(current.strip(), chunk_size, remaining_separators))

    return chunks


def split_long_text(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        return [text[start : start + chunk_size].strip() for start in range(0, len(text), chunk_size)]
    return recursive_split_text(text, chunk_size, separators)


def add_overlap(chunks: list[str], overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for previous, chunk in zip(chunks, chunks[1:]):
        prefix = previous[-overlap:].strip()
        overlapped.append(f"{prefix}\n{chunk}" if prefix else chunk)
    return overlapped


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    chunks = recursive_split_text(text, chunk_size=chunk_size)
    return add_overlap([chunk.strip() for chunk in chunks if chunk.strip()], overlap=overlap)


def is_section_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 100:
        return False
    return any(pattern.match(line) for pattern in SECTION_PATTERNS)


def extract_title_and_section(text: str) -> tuple[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    section = ""

    for line in lines:
        if is_section_heading(line):
            section = line
            break

    title = section or (lines[0] if lines else "")
    return title[:120], section[:120]


def infer_doc_type(pdf_path: str | None) -> str:
    if not pdf_path:
        return "document"
    name = Path(pdf_path).stem.lower()
    if "faq" in name:
        return "faq"
    if "manual" in name:
        return "manual"
    if re.search(r"p\d{2}d\d+", name):
        return "datasheet"
    return "document"


def detect_language(text: str) -> str:
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    ascii_letters = len(re.findall(r"[A-Za-z]", text))
    if chinese_chars and chinese_chars >= ascii_letters * 0.2:
        return "zh"
    return "en"


def build_document_text(pages: list[dict]) -> tuple[str, list[dict]]:
    parts = []
    spans = []
    cursor = 0

    for page in pages:
        separator = "\n\n" if parts else ""
        cursor += len(separator)
        parts.append(separator + page["text"])
        start = cursor
        end = start + len(page["text"])
        spans.append({"page": page["page"], "start": start, "end": end})
        cursor = end

    return "".join(parts), spans


def find_chunk_pages(chunk: str, document_text: str, page_spans: list[dict], search_start: int) -> tuple[int, int, int]:
    chunk_start = document_text.find(chunk, search_start)
    if chunk_start < 0:
        chunk_start = document_text.find(chunk[: min(len(chunk), 80)])
    if chunk_start < 0:
        chunk_start = search_start

    chunk_end = chunk_start + len(chunk)
    matched_pages = [
        span["page"]
        for span in page_spans
        if span["start"] < chunk_end and span["end"] > chunk_start
    ]

    if not matched_pages:
        nearest = min(page_spans, key=lambda span: abs(span["start"] - chunk_start))
        matched_pages = [nearest["page"]]

    return matched_pages[0], matched_pages[-1], chunk_start


def build_chunks(pages: list[dict], chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[dict]:
    chunk_records = []
    pages_by_source: dict[str, list[dict]] = {}

    for page in pages:
        pages_by_source.setdefault(page["source"], []).append(page)

    for source, source_pages in pages_by_source.items():
        source_pages = sorted(source_pages, key=lambda item: item["page"])
        document_text, page_spans = build_document_text(source_pages)
        chunks = chunk_text(document_text, chunk_size=chunk_size, overlap=overlap)
        file_path = source_pages[0].get("file_path", "")
        search_start = 0

        for idx, chunk in enumerate(chunks, start=1):
            start_page, end_page, chunk_start = find_chunk_pages(chunk, document_text, page_spans, search_start)
            search_start = max(chunk_start + 1, search_start)
            title, section = extract_title_and_section(chunk)
            chunk_records.append(
                {
                    "chunk_id": f"{source}-p{start_page}-{end_page}-c{idx}",
                    "source": source,
                    "file_path": file_path,
                    "page": start_page,
                    "start_page": start_page,
                    "end_page": end_page,
                    "chunk_index": idx,
                    "section": section,
                    "title": title,
                    "doc_type": infer_doc_type(file_path),
                    "language": detect_language(chunk),
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

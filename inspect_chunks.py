import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from rag_utils import CHUNKS_PATH


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*|[\u4e00-\u9fff]|[^\s]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a chunk inspection report")
    parser.add_argument("--chunks", default=str(CHUNKS_PATH), help="Path to pdf_chunks.jsonl")
    parser.add_argument(
        "--output",
        default=str(CHUNKS_PATH.parent / "debug" / "chunk_inspection.md"),
        help="Output report path",
    )
    parser.add_argument("--format", choices=["md", "jsonl"], default="md", help="Output format")
    parser.add_argument("--source", default=None, help="Only include chunks from one source filename")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of chunks to include")
    parser.add_argument("--preview-chars", type=int, default=1400, help="Text preview length for Markdown output")
    parser.add_argument("--include-full-text", action="store_true", help="Do not truncate Markdown chunk text")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def estimate_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text or ""))


def page_label(record: dict) -> str:
    start_page = record.get("start_page", record.get("page"))
    end_page = record.get("end_page", record.get("page"))
    if start_page == end_page:
        return str(start_page)
    return f"{start_page}-{end_page}"


def list_value(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def text_preview(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: max(limit - 16, 0)].rstrip() + "\n...[truncated]"


def boundary_notes(record: dict) -> list[str]:
    notes = []
    text = record.get("text") or ""
    if not text.strip():
        notes.append("empty_text")
    if record.get("char_start") is None or record.get("char_end") is None:
        notes.append("missing_char_range")
    if text and re.match(r"^[a-z0-9]", text, flags=re.IGNORECASE):
        notes.append("starts_with_word_char")
    if text and re.search(r"[a-z0-9]$", text, flags=re.IGNORECASE):
        notes.append("ends_with_word_char")
    if record.get("table_context"):
        notes.append(f"table_context={record['table_context']}")
    if record.get("list_structure"):
        notes.append(f"list_structure={record['list_structure']}")
    return notes


def build_inspection_records(records: list[dict]) -> list[dict]:
    inspected = []
    previous_by_source: dict[str, dict] = {}

    for index, record in enumerate(records, start=1):
        source = record.get("source", "")
        previous = previous_by_source.get(source)
        char_start = record.get("char_start")
        char_end = record.get("char_end")
        overlap_chars = None
        gap_chars = None

        if previous and char_start is not None and previous.get("char_end") is not None:
            delta = char_start - previous["char_end"]
            if delta < 0:
                overlap_chars = abs(delta)
            elif delta > 0:
                gap_chars = delta

        text = record.get("text") or ""
        inspected_record = {
            "index": index,
            "chunk_id": record.get("chunk_id"),
            "source": source,
            "page": record.get("page"),
            "page_label": page_label(record),
            "char_start": char_start,
            "char_end": char_end,
            "char_length": len(text),
            "estimated_tokens": estimate_tokens(text),
            "parent_id": record.get("parent_id"),
            "prev_chunk_id": record.get("prev_chunk_id"),
            "next_chunk_id": record.get("next_chunk_id"),
            "section": record.get("section") or record.get("title"),
            "heading_level": record.get("heading_level"),
            "product": record.get("product"),
            "product_codes": record.get("product_codes", []),
            "chip_models": record.get("chip_models", []),
            "vendors": record.get("vendors", []),
            "tags": record.get("tags", []),
            "specs": record.get("specs", {}),
            "overlap_with_previous_chars": overlap_chars,
            "gap_from_previous_chars": gap_chars,
            "notes": boundary_notes(record),
            "text": text,
        }
        inspected.append(inspected_record)
        previous_by_source[source] = record

    return inspected


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_markdown(records: list[dict], output_path: Path, preview_chars: int, include_full_text: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_counts = Counter(record["source"] for record in records)

    lines = [
        "# Chunk Inspection",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- total_chunks: {len(records)}",
        f"- sources: {len(source_counts)}",
        "",
        "## Source Summary",
        "",
    ]
    for source, count in source_counts.most_common():
        lines.append(f"- {source}: {count}")

    current_source = None
    for record in records:
        if record["source"] != current_source:
            current_source = record["source"]
            lines.extend(["", f"## {current_source}", ""])

        notes = ", ".join(record["notes"]) if record["notes"] else "none"
        text = record["text"] if include_full_text else text_preview(record["text"], preview_chars)
        lines.extend(
            [
                f"### {record['index']}. {record['chunk_id']}",
                "",
                f"- page: {record['page_label']}",
                f"- char_range: {record['char_start']}-{record['char_end']}",
                f"- chars/tokens: {record['char_length']} / {record['estimated_tokens']}",
                f"- parent_id: {record['parent_id']}",
                f"- prev/next: {record['prev_chunk_id']} / {record['next_chunk_id']}",
                f"- section: {record['section'] or ''}",
                f"- product_codes: {list_value(record['product_codes'])}",
                f"- chip_models: {list_value(record['chip_models'])}",
                f"- vendors: {list_value(record['vendors'])}",
                f"- overlap/gap from previous chars: {record['overlap_with_previous_chars']} / {record['gap_from_previous_chars']}",
                f"- notes: {notes}",
                "",
                "```text",
                text,
                "```",
                "",
            ]
        )

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def main() -> None:
    args = parse_args()
    chunk_path = Path(args.chunks)
    output_path = Path(args.output)

    records = load_jsonl(chunk_path)
    if args.source:
        records = [record for record in records if record.get("source") == args.source]
    if args.limit is not None:
        records = records[: max(args.limit, 0)]

    inspection_records = build_inspection_records(records)
    if args.format == "jsonl":
        write_jsonl(inspection_records, output_path)
    else:
        write_markdown(inspection_records, output_path, args.preview_chars, args.include_full_text)

    print(f"Chunk inspection written to: {output_path}")


if __name__ == "__main__":
    main()

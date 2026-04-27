import json
import re
from bisect import bisect_left
from pathlib import Path

import fitz  # PyMuPDF

from specs import extract_specs_from_text


DOCS_DIR = Path.home() / "rag-project" / "docs"
OUTPUT_DIR = Path.home() / "rag-project" / "data"
RAW_OUTPUT = OUTPUT_DIR / "pdf_pages.jsonl"
CHUNK_OUTPUT = OUTPUT_DIR / "pdf_chunks.jsonl"
DEFAULT_CHUNK_SIZE = 320
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_PARENT_CHUNK_SIZE = 850
DEFAULT_PARENT_CHUNK_OVERLAP = 100
SPLIT_SEPARATORS = ["\n\n", "\n", "Q:", "A:", "Question:", "Answer:", ". ", "; ", ", ", " ", ""]
HEADER_FOOTER_MIN_PAGES = 2


SECTION_PATTERNS = [
    re.compile(r"^\s*(?:\d+(?:\.\d+)+[.)]?|\d+[.)])\s+.{3,80}$"),
    re.compile(r"^\s*(?:chapter|section)\s+\d+[:.)]?\s+.{3,80}$", re.IGNORECASE),
    re.compile(r"^\s*(?:hardware|software|features|specifications|dimensions|block diagram|i/o|display|memory|cpu)\s*$", re.IGNORECASE),
]

DOCUMENT_SECTION_PATTERNS = [
    re.compile(r"^\s*(?:\d+(?:\.\d+)+[.)]?|\d+[.)])\s+.{3,80}$"),
    re.compile(r"^\s*(?:chapter|section)\s+\d+[:.)]?\s+.{3,80}$", re.IGNORECASE),
    re.compile(r"^\s*(?:hardware|software|features|specifications|dimensions|block diagram)\s*$", re.IGNORECASE),
]

TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*|[\u4e00-\u9fff]|[^\s]",
    re.UNICODE,
)


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def format_table_as_rows(table: list[list[str | None]], table_index: int) -> str:
    cleaned_rows = []
    for row in table:
        cells = [normalize_text(cell or "") for cell in row]
        if any(cells):
            cleaned_rows.append(cells)

    if not cleaned_rows:
        return ""

    max_columns = max(len(row) for row in cleaned_rows)
    normalized_rows = [row + [""] * (max_columns - len(row)) for row in cleaned_rows]
    first_row = normalized_rows[0]
    first_row_has_header_shape = (
        len([cell for cell in first_row if cell]) >= 2
        and sum(bool(re.search(r"[A-Za-z\u4e00-\u9fff]", cell)) for cell in first_row) >= 2
    )

    if first_row_has_header_shape:
        headers = [
            cell if cell else f"Column {idx}"
            for idx, cell in enumerate(first_row, start=1)
        ]
        data_rows = normalized_rows[1:]
    else:
        headers = [f"Column {idx}" for idx in range(1, max_columns + 1)]
        data_rows = normalized_rows

    lines = [f"Table: Table {table_index}"]
    for row_index, row in enumerate(data_rows, start=1):
        lines.append(f"Row {row_index}:")
        for header, cell in zip(headers, row):
            if cell:
                lines.append(f"{header}: {cell}")

    return "\n".join(lines)


def extract_table_texts_with_pdfplumber(pdf_path: Path) -> dict[int, list[str]]:
    try:
        import pdfplumber
    except ImportError:
        return {}

    table_texts: dict[int, list[str]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            page_tables = []
            for table_index, table in enumerate(page.extract_tables() or [], start=1):
                table_text = format_table_as_rows(table, table_index)
                if table_text:
                    page_tables.append(table_text)
            if page_tables:
                table_texts[page_index] = page_tables
    return table_texts


def extract_structured_page_text(page: fitz.Page, table_texts: list[str] | None = None) -> str:
    blocks = page.get_text("blocks", sort=True)
    parts = []

    for block in blocks:
        text = normalize_text(block[4] if len(block) > 4 else "")
        if not text:
            continue
        parts.append(text)

    if table_texts:
        for idx, table_text in enumerate(table_texts, start=1):
            parts.append(f"[TABLE {idx}]\n{normalize_text(table_text)}")

    return normalize_text("\n\n".join(parts))


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    table_texts = extract_table_texts_with_pdfplumber(pdf_path)
    with fitz.open(pdf_path) as doc:
        raw_pages = [
            {
                "source": pdf_path.name,
                "file_path": str(pdf_path),
                "page": i + 1,
                "text": extract_structured_page_text(page, table_texts.get(i + 1, [])),
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


def make_chunk_span(text: str, start: int, end: int, base_offset: int = 0) -> dict | None:
    relative_start = start - base_offset
    relative_end = end - base_offset

    while relative_start < relative_end and text[relative_start].isspace():
        relative_start += 1
        start += 1
    while relative_end > relative_start and text[relative_end - 1].isspace():
        relative_end -= 1
        end -= 1

    if relative_start >= relative_end:
        return None
    return {"text": text[relative_start:relative_end], "start": start, "end": end}


def recursive_split_spans(
    text: str,
    start_offset: int = 0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    separators: list[str] | None = None,
) -> list[dict]:
    if len(text) <= chunk_size:
        span = make_chunk_span(text, start_offset, start_offset + len(text), base_offset=start_offset)
        return [span] if span else []

    separators = separators or SPLIT_SEPARATORS
    separator = separators[0]
    remaining_separators = separators[1:]

    if not separator:
        spans = []
        for relative_start in range(0, len(text), chunk_size):
            relative_end = min(relative_start + chunk_size, len(text))
            span = make_chunk_span(
                text,
                start_offset + relative_start,
                start_offset + relative_end,
                base_offset=start_offset,
            )
            if span:
                spans.append(span)
        return spans

    split_points = []
    cursor = 0
    while True:
        idx = text.find(separator, cursor)
        if idx < 0:
            break
        split_points.append(idx + len(separator))
        cursor = idx + len(separator)

    if not split_points:
        return recursive_split_spans(text, start_offset, chunk_size, remaining_separators)

    spans = []
    current_start = 0
    current_end = 0

    for split_end in split_points + [len(text)]:
        if split_end - current_start <= chunk_size:
            current_end = split_end
            continue

        if current_end > current_start:
            spans.extend(
                split_long_span(
                    text[current_start:current_end],
                    start_offset + current_start,
                    chunk_size,
                    remaining_separators,
                )
            )
            current_start = current_end

        if split_end - current_start > chunk_size:
            spans.extend(
                split_long_span(
                    text[current_start:split_end],
                    start_offset + current_start,
                    chunk_size,
                    remaining_separators,
                )
            )
            current_start = split_end

        current_end = split_end

    if current_end > current_start:
        spans.extend(
            split_long_span(
                text[current_start:current_end],
                start_offset + current_start,
                chunk_size,
                remaining_separators,
            )
        )

    return spans


def split_long_span(text: str, start_offset: int, chunk_size: int, separators: list[str]) -> list[dict]:
    if len(text) <= chunk_size:
        span = make_chunk_span(text, start_offset, start_offset + len(text), base_offset=start_offset)
        return [span] if span else []
    return recursive_split_spans(text, start_offset, chunk_size, separators)


def token_spans(text: str) -> list[dict]:
    return [
        {"token": match.group(0), "start": match.start(), "end": match.end()}
        for match in TOKEN_PATTERN.finditer(text)
    ]


def estimate_token_count(text: str) -> int:
    return len(token_spans(text))


def choose_token_split_end(
    text: str,
    tokens: list[dict],
    token_starts: list[int],
    start_index: int,
    max_end_index: int,
    min_tokens: int,
    separators: list[str],
) -> int:
    if max_end_index >= len(tokens):
        return len(tokens)

    min_end_index = min(max(start_index + min_tokens, start_index + 1), max_end_index)
    min_char = tokens[min_end_index - 1]["end"]
    max_char = tokens[max_end_index - 1]["end"]

    for separator in [item for item in separators if item]:
        cursor = min_char
        selected_split = -1
        while True:
            idx = text.find(separator, cursor, max_char + 1)
            if idx < 0:
                break
            split_char = idx + len(separator)
            if split_char <= max_char:
                selected_split = split_char
            cursor = idx + max(len(separator), 1)

        if selected_split > 0:
            end_index = bisect_left(token_starts, selected_split)
            if end_index > start_index:
                return end_index

    return max_end_index


def token_split_spans(
    text: str,
    start_offset: int = 0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: list[str] | None = None,
) -> list[dict]:
    tokens = token_spans(text)
    if not tokens:
        return []
    if len(tokens) <= chunk_size:
        span = make_chunk_span(text, start_offset, start_offset + len(text), base_offset=start_offset)
        return [span] if span else []

    separators = separators or SPLIT_SEPARATORS
    token_starts = [token["start"] for token in tokens]
    min_tokens = max(20, int(chunk_size * 0.55))
    spans = []
    start_index = 0

    while start_index < len(tokens):
        max_end_index = min(start_index + chunk_size, len(tokens))
        end_index = choose_token_split_end(
            text,
            tokens,
            token_starts,
            start_index,
            max_end_index,
            min_tokens,
            separators,
        )
        if end_index <= start_index:
            end_index = max_end_index

        relative_start = tokens[start_index]["start"]
        relative_end = tokens[end_index - 1]["end"]
        span = make_chunk_span(
            text,
            start_offset + relative_start,
            start_offset + relative_end,
            base_offset=start_offset,
        )
        if span:
            spans.append(span)

        if end_index >= len(tokens):
            break

        next_start = max(end_index - overlap, start_index + 1)
        start_index = next_start

    return spans


def align_to_text_boundary(text: str, start: int, limit: int = 40) -> int:
    if start <= 0 or start >= len(text):
        return max(0, min(start, len(text)))
    if text[start].isspace() or text[start - 1].isspace():
        return start

    search_start = max(0, start - limit)
    for idx in range(start - 1, search_start - 1, -1):
        if text[idx].isspace():
            return idx + 1

    search_end = min(len(text), start + limit)
    for idx in range(start, search_end):
        if text[idx].isspace():
            return idx + 1
    return start


def add_overlap_spans(text: str, spans: list[dict], overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[dict]:
    if overlap <= 0:
        return spans

    overlapped = []
    previous_end = 0
    for idx, span in enumerate(spans):
        start = span["start"] if idx == 0 else align_to_text_boundary(text, max(0, previous_end - overlap))
        overlapped_span = make_chunk_span(text, start, span["end"])
        if overlapped_span:
            overlapped.append(overlapped_span)
        previous_end = span["end"]
    return overlapped


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    start_offset: int = 0,
) -> list[dict]:
    return token_split_spans(text, start_offset=start_offset, chunk_size=chunk_size, overlap=overlap)


def is_section_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 100:
        return False
    if re.match(r"^\d+(?:\.\d+)?\s*(?:inch|inches|mm|cm|v|hz|khz|mhz|ghz|gb|mb)\b", line, re.IGNORECASE):
        return False
    return any(pattern.match(line) for pattern in SECTION_PATTERNS)


def is_document_section_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 100:
        return False
    if re.match(r"^\d+(?:\.\d+)?\s*(?:inch|inches|mm|cm|v|hz|khz|mhz|ghz|gb|mb)\b", line, re.IGNORECASE):
        return False
    return any(pattern.match(line) for pattern in DOCUMENT_SECTION_PATTERNS)


def extract_title_and_section(text: str) -> tuple[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    section = ""

    for line in lines:
        if is_section_heading(line):
            section = line
            break

    title = section or (lines[0] if lines else "")
    return title[:120], section[:120]


def extract_heading_level(text: str) -> int:
    first_nonempty = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_nonempty:
        return 0

    numbered = re.match(r"^(\d+(?:\.\d+)*)", first_nonempty)
    if numbered:
        return numbered.group(1).count(".") + 1
    if first_nonempty.isupper() and len(first_nonempty.split()) <= 8:
        return 1
    if is_section_heading(first_nonempty):
        return 2
    return 0


def iter_line_spans(text: str) -> list[dict]:
    spans = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        start = cursor
        end = cursor + len(line)
        spans.append({"start": start, "end": end, "text": line.rstrip("\n")})
        cursor = end
    if cursor < len(text):
        spans.append({"start": cursor, "end": len(text), "text": text[cursor:]})
    return spans


def split_document_sections(text: str) -> list[dict]:
    line_spans = iter_line_spans(text)
    if not line_spans:
        return []

    sections = []
    current_start = 0
    current_section = ""
    current_heading_level = 0

    for line in line_spans:
        stripped = line["text"].strip()
        if not is_document_section_heading(stripped):
            continue

        if line["start"] > current_start:
            section_text = text[current_start:line["start"]]
            if section_text.strip():
                sections.append(
                    {
                        "start": current_start,
                        "end": line["start"],
                        "text": section_text,
                        "section": current_section,
                        "heading_level": current_heading_level,
                    }
                )

        current_start = line["start"]
        current_section = stripped[:120]
        current_heading_level = extract_heading_level(stripped)

    if current_start < len(text):
        section_text = text[current_start:]
        if section_text.strip():
            sections.append(
                {
                    "start": current_start,
                    "end": len(text),
                    "text": section_text,
                    "section": current_section,
                    "heading_level": current_heading_level,
                }
            )

    if not sections:
        return [{"start": 0, "end": len(text), "text": text, "section": "", "heading_level": 0}]

    merged_sections = []
    idx = 0
    while idx < len(sections):
        section = sections[idx]
        if section.get("section") and estimate_token_count(section["text"]) <= 8 and idx + 1 < len(sections):
            next_section = sections[idx + 1].copy()
            parent_section = section["section"]
            child_section = next_section.get("section", "")
            next_section["start"] = section["start"]
            next_section["text"] = text[section["start"]:next_section["end"]]
            next_section["section"] = (
                f"{parent_section} / {child_section}" if child_section and child_section != parent_section else parent_section
            )[:120]
            next_section["heading_level"] = section.get("heading_level", next_section.get("heading_level", 0))
            merged_sections.append(next_section)
            idx += 2
            continue

        merged_sections.append(section)
        idx += 1

    return merged_sections


def extract_list_structure(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_count = sum(bool(re.match(r"^(?:[-*•]|[0-9]+[.)]|[A-Za-z][.)])\s+", line)) for line in lines)
    if bullet_count >= 3:
        return "list"
    if bullet_count >= 1:
        return "mixed"
    return ""


def extract_table_context(text: str) -> str:
    if "[TABLE" in text:
        return "extracted_table"

    lines = [line for line in text.splitlines() if line.strip()]
    pipe_lines = sum("|" in line for line in lines)
    spaced_columns = sum(bool(re.search(r"\S\s{2,}\S", line)) for line in lines)
    if pipe_lines >= 2 or spaced_columns >= 3:
        return "table_like_layout"
    return ""


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


def unique_values(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        normalized = re.sub(r"\s+", " ", value).strip()
        if not normalized:
            continue
        key = normalized.upper()
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def extract_product_codes(text: str) -> list[str]:
    return unique_values(
        [match.upper() for match in re.findall(r"\bP\d{2}D\d{5}(?:-\d{2})?(?:-[A-Z0-9]+)?\b", text, flags=re.IGNORECASE)]
    )


def extract_product_identity(source: str, text: str) -> dict:
    source_product_codes = extract_product_codes(source)
    text_product_codes = extract_product_codes(text)
    product_codes = text_product_codes or source_product_codes
    chip_models = unique_values(
        [
            match.upper()
            for match in re.findall(
                r"\b(?:RK\d{4}[A-Z0-9]*|STM32[A-Z0-9]+|RTL\d+[A-Z0-9]*|RZ/[A-Z0-9]+)\b",
                text,
                flags=re.IGNORECASE,
            )
        ]
    )

    for match in re.findall(r"\bi\.MX[0-9A-Za-z]+(?:\s+\w+)?\b", text):
        chip_models.append(match)
    chip_models = unique_values(chip_models)

    vendors = []
    if re.search(r"\bST\b", text, flags=re.IGNORECASE) or any(model.startswith("STM32") for model in chip_models):
        vendors.append("ST")
    if re.search(r"\bRockchip\b", text, flags=re.IGNORECASE) or any(model.startswith("RK") for model in chip_models):
        vendors.append("Rockchip")
    if re.search(r"\bNXP\b", text, flags=re.IGNORECASE) or any(model.lower().startswith("i.mx") for model in chip_models):
        vendors.append("NXP")
    if any(model.startswith("RTL") for model in chip_models):
        vendors.append("Realtek")
    if any(model.startswith("RZ/") for model in chip_models):
        vendors.append("Renesas")
    vendors = unique_values(vendors)

    primary_product = ""
    if chip_models:
        primary_product = f"ST {chip_models[0]}" if vendors[:1] == ["ST"] and chip_models[0].startswith("STM32") else chip_models[0]
    elif product_codes:
        primary_product = product_codes[0]

    return {
        "product": primary_product,
        "product_codes": product_codes,
        "text_product_codes": text_product_codes,
        "source_product_codes": source_product_codes,
        "chip_models": chip_models,
        "vendors": vendors,
    }


def extract_product(source: str, text: str) -> str:
    return extract_product_identity(source, text)["product"]


def extract_version(text: str) -> str:
    patterns = [
        r"\bFW\s*v?\d+(?:\.\d+){1,3}\b",
        r"\bFirmware\s*v?\d+(?:\.\d+){1,3}\b",
        r"\bYocto\s+\d+(?:\.\d+){1,2}\b",
        r"\bAndroid\s+\d+(?:\.\d+){0,2}\b",
        r"\bv\d+(?:\.\d+){1,3}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return ""


def extract_tags(text: str) -> list[str]:
    tag_keywords = {
        "firmware": r"\b(?:firmware|fw)\b",
        "upgrade": r"\b(?:upgrade|update)\b",
        "hardware": r"\bhardware\b",
        "software": r"\bsoftware\b",
        "ethernet": r"\bethernet\b",
        "usb": r"\busb\b",
        "display": r"\b(?:display|hdmi|lvds|mipi|dsi)\b",
        "camera": r"\b(?:camera|csi)\b",
        "memory": r"\b(?:memory|ram|flash|emmc)\b",
        "power": r"\bpower\b",
        "gpio": r"\bgpio",
    }
    return [tag for tag, pattern in tag_keywords.items() if re.search(pattern, text, flags=re.IGNORECASE)]


def extract_faq_pairs(text: str) -> list[dict]:
    pattern = re.compile(
        r"(?ims)^\s*(?:Q|Question)\s*[:：]\s*(?P<question>.*?)"
        r"^\s*(?:A|Answer)\s*[:：]\s*(?P<answer>.*?)(?=^\s*(?:Q|Question)\s*[:：]|\Z)"
    )
    pairs = []
    for match in pattern.finditer(text):
        question = normalize_text(match.group("question"))
        answer = normalize_text(match.group("answer"))
        if not question or not answer:
            continue
        pair_text = f"Q: {question}\nA: {answer}"
        pairs.append(
            {
                "text": pair_text,
                "question": question,
                "answer": answer,
                "start": match.start(),
                "end": match.end(),
            }
        )
    return pairs


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


def find_pages_for_span(start: int, end: int, page_spans: list[dict]) -> tuple[int, int]:
    matched_pages = [
        span["page"]
        for span in page_spans
        if span["start"] < end and span["end"] > start
    ]
    if not matched_pages:
        nearest = min(page_spans, key=lambda span: abs(span["start"] - start))
        matched_pages = [nearest["page"]]
    return matched_pages[0], matched_pages[-1]


def build_parent_child_spans(
    source: str,
    document_text: str,
    page_spans: list[dict],
    child_chunk_size: int,
    child_overlap: int,
    parent_chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
    parent_overlap: int = DEFAULT_PARENT_CHUNK_OVERLAP,
) -> list[dict]:
    child_spans = []
    parent_index = 0

    for section in split_document_sections(document_text):
        parents = chunk_text(
            section["text"],
            chunk_size=parent_chunk_size,
            overlap=parent_overlap,
            start_offset=section["start"],
        )

        for parent in parents:
            parent_index += 1
            parent_start_page, parent_end_page = find_pages_for_span(parent["start"], parent["end"], page_spans)
            parent_id = f"{source}-p{parent_start_page}-{parent_end_page}-parent-{parent_index}"
            children = chunk_text(
                parent["text"],
                chunk_size=child_chunk_size,
                overlap=child_overlap,
                start_offset=parent["start"],
            )

            for child_index, child in enumerate(children, start=1):
                child["parent_id"] = parent_id
                child["parent_index"] = parent_index
                child["parent_child_index"] = child_index
                child["parent_child_count"] = len(children)
                child["parent_text"] = parent["text"]
                child["parent_char_start"] = parent["start"]
                child["parent_char_end"] = parent["end"]
                child["parent_start_page"] = parent_start_page
                child["parent_end_page"] = parent_end_page
                child["parent_section"] = section.get("section", "")
                child["parent_heading_level"] = section.get("heading_level", 0)
                child_spans.append(child)

    return child_spans


def build_chunk_record(
    source: str,
    file_path: str,
    idx: int,
    chunk: dict,
    page_spans: list[dict],
    doc_type: str,
) -> dict:
    start_page, end_page = find_pages_for_span(chunk["start"], chunk["end"], page_spans)
    context_text = chunk.get("parent_text", chunk["text"])
    title, detected_section = extract_title_and_section(chunk["text"])
    parent_title, parent_section = extract_title_and_section(context_text)
    section = detected_section or chunk.get("parent_section", "") or parent_section
    title = detected_section or section or title or parent_title
    metadata_text = f"{chunk['text']}\n\n{context_text}"
    product_identity = extract_product_identity(source, metadata_text)
    specs_payload = extract_specs_from_text(metadata_text)
    version = extract_version(metadata_text)
    heading_level = extract_heading_level(chunk["text"]) or chunk.get("parent_heading_level", 0)
    list_structure = extract_list_structure(chunk["text"])
    table_context = extract_table_context(chunk["text"])

    record = {
        "chunk_id": f"{source}-p{start_page}-{end_page}-c{idx}",
        "source": source,
        "file_path": file_path,
        "page": start_page,
        "start_page": start_page,
        "end_page": end_page,
        "char_start": chunk["start"],
        "char_end": chunk["end"],
        "chunk_index": idx,
        "chunk_role": "child",
        "token_count": estimate_token_count(chunk["text"]),
        "section": section,
        "title": title,
        "doc_type": doc_type,
        "product": product_identity["product"],
        "product_codes": product_identity["product_codes"],
        "text_product_codes": product_identity["text_product_codes"],
        "source_product_codes": product_identity["source_product_codes"],
        "chip_models": product_identity["chip_models"],
        "vendors": product_identity["vendors"],
        "version": version,
        "heading_level": heading_level,
        "list_structure": list_structure,
        "table_context": table_context,
        "language": detect_language(chunk["text"]),
        "tags": extract_tags(chunk["text"]),
        "specs": specs_payload["specs"],
        "spec_evidence": specs_payload["spec_evidence"],
        "text": chunk["text"],
        "parent_id": chunk.get("parent_id", f"{source}-p{start_page}-{end_page}-parent-{idx}"),
        "parent_index": chunk.get("parent_index", idx),
        "parent_child_index": chunk.get("parent_child_index", 1),
        "parent_child_count": chunk.get("parent_child_count", 1),
        "parent_text": context_text,
        "parent_token_count": estimate_token_count(context_text),
        "parent_start_page": chunk.get("parent_start_page", start_page),
        "parent_end_page": chunk.get("parent_end_page", end_page),
        "parent_char_start": chunk.get("parent_char_start", chunk["start"]),
        "parent_char_end": chunk.get("parent_char_end", chunk["end"]),
    }

    if "question" in chunk:
        record["question"] = chunk["question"]
        record["answer"] = chunk["answer"]

    return record


def build_chunks(
    pages: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    parent_chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
    parent_overlap: int = DEFAULT_PARENT_CHUNK_OVERLAP,
) -> list[dict]:
    chunk_records = []
    pages_by_source: dict[str, list[dict]] = {}

    for page in pages:
        pages_by_source.setdefault(page["source"], []).append(page)

    for source, source_pages in pages_by_source.items():
        source_pages = sorted(source_pages, key=lambda item: item["page"])
        document_text, page_spans = build_document_text(source_pages)
        file_path = source_pages[0].get("file_path", "")
        doc_type = infer_doc_type(file_path)
        chunks = extract_faq_pairs(document_text) if doc_type == "faq" else []
        if chunks:
            for idx, chunk in enumerate(chunks, start=1):
                start_page, end_page = find_pages_for_span(chunk["start"], chunk["end"], page_spans)
                chunk["parent_id"] = f"{source}-p{start_page}-{end_page}-faq-parent-{idx}"
                chunk["parent_index"] = idx
                chunk["parent_child_index"] = 1
                chunk["parent_child_count"] = 1
                chunk["parent_text"] = chunk["text"]
                chunk["parent_char_start"] = chunk["start"]
                chunk["parent_char_end"] = chunk["end"]
                chunk["parent_start_page"] = start_page
                chunk["parent_end_page"] = end_page
                chunk["parent_section"] = ""
                chunk["parent_heading_level"] = 0
        else:
            chunks = build_parent_child_spans(
                source,
                document_text,
                page_spans,
                child_chunk_size=chunk_size,
                child_overlap=overlap,
                parent_chunk_size=parent_chunk_size,
                parent_overlap=parent_overlap,
            )

        for idx, chunk in enumerate(chunks, start=1):
            chunk_records.append(build_chunk_record(source, file_path, idx, chunk, page_spans, doc_type))

        source_chunk_records = chunk_records[-len(chunks):]
        for idx, record in enumerate(source_chunk_records):
            record["prev_chunk_id"] = source_chunk_records[idx - 1]["chunk_id"] if idx > 0 else ""
            record["next_chunk_id"] = source_chunk_records[idx + 1]["chunk_id"] if idx + 1 < len(source_chunk_records) else ""

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

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from rag_utils import CHUNKS_PATH


DEFAULT_OUTPUT = CHUNKS_PATH.parent / "debug" / "chunk_preview.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an interactive HTML preview for PDF chunks")
    parser.add_argument("--chunks", default=str(CHUNKS_PATH), help="Path to pdf_chunks.jsonl")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output HTML path")
    parser.add_argument("--source", default=None, help="Only include chunks from one source filename")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of chunks to include")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def page_label(record: dict[str, Any]) -> str:
    start_page = record.get("start_page", record.get("page"))
    end_page = record.get("end_page", record.get("page"))
    if start_page == end_page:
        return str(start_page)
    return f"{start_page}-{end_page}"


def compact_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def non_empty_specs(specs: Any) -> dict[str, Any]:
    if not isinstance(specs, dict):
        return {}

    cleaned: dict[str, Any] = {}
    for key, value in specs.items():
        if value is None:
            continue
        if isinstance(value, list) and not value:
            continue
        cleaned[key] = value
    return cleaned


def evidence_count(evidence: Any) -> int:
    if not isinstance(evidence, dict):
        return 0
    total = 0
    for value in evidence.values():
        if isinstance(value, list):
            total += len(value)
        elif value:
            total += 1
    return total


def build_payload(records: list[dict[str, Any]], chunk_path: Path, output_path: Path) -> dict[str, Any]:
    output_dir = output_path.parent
    source_counts = Counter(str(record.get("source", "")) for record in records)
    tag_counts = Counter(tag for record in records for tag in compact_list(record.get("tags")))
    section_counts = Counter(str(record.get("section") or record.get("title") or "") for record in records)
    token_counts = [int(record.get("token_count") or 0) for record in records]
    parent_counts = Counter(str(record.get("parent_id", "")) for record in records if record.get("parent_id"))

    chunks = []
    for index, record in enumerate(records):
        specs = non_empty_specs(record.get("specs"))
        file_path = str(record.get("file_path") or "")
        pdf_href = ""
        if file_path:
            path = Path(file_path)
            if path.exists():
                pdf_href = Path(os.path.relpath(path.resolve(), output_dir.resolve())).as_posix()

        chunks.append(
            {
                "index": index,
                "chunk_id": record.get("chunk_id", ""),
                "source": record.get("source", ""),
                "pdf_href": pdf_href,
                "page": record.get("page"),
                "page_label": page_label(record),
                "start_page": record.get("start_page", record.get("page")),
                "end_page": record.get("end_page", record.get("page")),
                "char_start": record.get("char_start"),
                "char_end": record.get("char_end"),
                "chunk_index": record.get("chunk_index"),
                "chunk_role": record.get("chunk_role", ""),
                "token_count": record.get("token_count") or 0,
                "section": record.get("section") or "",
                "title": record.get("title") or "",
                "doc_type": record.get("doc_type") or "",
                "product": record.get("product") or "",
                "product_codes": compact_list(record.get("product_codes")),
                "text_product_codes": compact_list(record.get("text_product_codes")),
                "source_product_codes": compact_list(record.get("source_product_codes")),
                "chip_models": compact_list(record.get("chip_models")),
                "vendors": compact_list(record.get("vendors")),
                "version": record.get("version") or "",
                "heading_level": record.get("heading_level"),
                "list_structure": record.get("list_structure") or "",
                "table_context": record.get("table_context") or "",
                "language": record.get("language") or "",
                "tags": compact_list(record.get("tags")),
                "specs": specs,
                "spec_evidence": record.get("spec_evidence") if isinstance(record.get("spec_evidence"), dict) else {},
                "evidence_count": evidence_count(record.get("spec_evidence")),
                "text": record.get("text") or "",
                "parent_id": record.get("parent_id") or "",
                "parent_index": record.get("parent_index"),
                "parent_child_index": record.get("parent_child_index"),
                "parent_child_count": record.get("parent_child_count"),
                "parent_text": record.get("parent_text") or "",
                "parent_token_count": record.get("parent_token_count") or 0,
                "prev_chunk_id": record.get("prev_chunk_id") or "",
                "next_chunk_id": record.get("next_chunk_id") or "",
            }
        )

    stats = {
        "chunk_count": len(records),
        "source_count": len(source_counts),
        "parent_count": len(parent_counts),
        "tag_count": len(tag_counts),
        "token_min": min(token_counts) if token_counts else 0,
        "token_max": max(token_counts) if token_counts else 0,
        "token_mean": round(mean(token_counts), 1) if token_counts else 0,
        "token_median": round(median(token_counts), 1) if token_counts else 0,
    }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "chunk_path": str(chunk_path),
        "stats": stats,
        "sources": source_counts.most_common(),
        "tags": tag_counts.most_common(),
        "sections": section_counts.most_common(),
        "chunks": chunks,
    }


def json_script(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def render_html(payload: dict[str, Any]) -> str:
    data = json_script(payload)
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chunk Preview</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --panel-soft: #f8fafc;
      --ink: #17202f;
      --muted: #667085;
      --line: #d9e0eb;
      --accent: #1677ff;
      --accent-2: #18a058;
      --warn: #b7791f;
      --chip: #edf2f7;
      --shadow: 0 14px 35px rgba(31, 42, 68, 0.09);
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}

    button, input, select {{
      font: inherit;
    }}

    button {{
      cursor: pointer;
    }}

    .app {{
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }}

    header {{
      padding: 18px 24px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
      position: sticky;
      top: 0;
      z-index: 10;
      backdrop-filter: blur(12px);
    }}

    .topline {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }}

    h1 {{
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
      font-weight: 760;
    }}

    .subtle {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}

    .stats {{
      display: grid;
      grid-template-columns: repeat(6, minmax(96px, 1fr));
      gap: 10px;
    }}

    .metric {{
      background: var(--panel-soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      min-width: 0;
    }}

    .metric b {{
      display: block;
      font-size: 18px;
      line-height: 1.1;
      margin-bottom: 4px;
    }}

    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    main {{
      display: grid;
      grid-template-columns: minmax(330px, 420px) minmax(0, 1fr);
      min-height: 0;
    }}

    aside {{
      min-height: 0;
      border-right: 1px solid var(--line);
      background: #fbfcff;
      display: grid;
      grid-template-rows: auto auto 1fr;
    }}

    .filters {{
      display: grid;
      gap: 10px;
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }}

    .search {{
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      min-height: 42px;
      padding: 0 12px;
      outline: none;
    }}

    .search:focus, select:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(22, 119, 255, 0.13);
    }}

    .filter-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}

    select {{
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 0 10px;
      color: var(--ink);
    }}

    .source-chart {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 7px;
      max-height: 230px;
      overflow: auto;
    }}

    .source-row {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 44px;
      gap: 8px;
      align-items: center;
      padding: 5px 6px;
      border: 0;
      background: transparent;
      color: var(--ink);
      text-align: left;
      border-radius: 7px;
    }}

    .source-row:hover, .source-row.active {{
      background: #eef5ff;
    }}

    .source-name {{
      min-width: 0;
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    .source-count {{
      color: var(--muted);
      font-size: 12px;
      text-align: right;
    }}

    .bar {{
      grid-column: 1 / -1;
      height: 4px;
      border-radius: 999px;
      background: #e4e9f2;
      overflow: hidden;
    }}

    .bar i {{
      display: block;
      height: 100%;
      width: var(--w);
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }}

    .list-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 14px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }}

    .chunk-list {{
      min-height: 0;
      overflow: auto;
      padding: 10px;
    }}

    .chunk-item {{
      width: 100%;
      border: 1px solid transparent;
      background: transparent;
      border-radius: 8px;
      padding: 10px;
      text-align: left;
      color: inherit;
      display: grid;
      gap: 7px;
    }}

    .chunk-item:hover {{
      background: #f0f5ff;
    }}

    .chunk-item.active {{
      background: #fff;
      border-color: #b9d7ff;
      box-shadow: var(--shadow);
    }}

    .chunk-title {{
      display: flex;
      gap: 8px;
      align-items: baseline;
      min-width: 0;
    }}

    .chunk-title b {{
      min-width: 0;
      font-size: 13px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    .page-pill {{
      flex: 0 0 auto;
      padding: 2px 7px;
      border-radius: 999px;
      background: #e8f3ff;
      color: #1356a3;
      font-size: 11px;
      font-weight: 700;
    }}

    .chunk-meta {{
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    .mini-tags {{
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
    }}

    .chip {{
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 7px;
      border-radius: 999px;
      background: var(--chip);
      color: #334155;
      font-size: 12px;
      max-width: 100%;
    }}

    .chip.good {{
      background: #e8f8ef;
      color: #147243;
    }}

    .chip.warn {{
      background: #fff4dc;
      color: var(--warn);
    }}

    .detail {{
      min-height: 0;
      overflow: auto;
      padding: 20px 24px 28px;
    }}

    .detail-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 300px;
      gap: 18px;
      align-items: start;
    }}

    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}

    .panel-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }}

    .panel-head h2 {{
      margin: 0 0 6px;
      font-size: 19px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }}

    .actions {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}

    .action {{
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 0 10px;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
    }}

    .action:hover {{
      border-color: var(--accent);
      color: #0d5bc4;
    }}

    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-soft);
    }}

    .kv {{
      min-width: 0;
    }}

    .kv span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 3px;
    }}

    .kv b {{
      display: block;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}

    .body {{
      padding: 16px;
    }}

    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 13px;
      line-height: 1.62;
      color: #263244;
    }}

    mark {{
      background: #fff0a8;
      color: inherit;
      padding: 0 2px;
      border-radius: 3px;
    }}

    .side {{
      display: grid;
      gap: 14px;
    }}

    .side-section {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}

    .side-section h3 {{
      margin: 0;
      padding: 12px 13px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }}

    .side-body {{
      padding: 12px 13px;
      display: grid;
      gap: 8px;
      font-size: 13px;
    }}

    .spec-row {{
      display: grid;
      grid-template-columns: 105px minmax(0, 1fr);
      gap: 8px;
      align-items: start;
      padding-bottom: 8px;
      border-bottom: 1px dashed #dce3ef;
    }}

    .spec-row:last-child {{
      border-bottom: 0;
      padding-bottom: 0;
    }}

    .spec-key {{
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}

    .spec-value {{
      overflow-wrap: anywhere;
    }}

    .empty {{
      color: var(--muted);
      font-size: 13px;
    }}

    .toggle-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 12px;
    }}

    .switch {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }}

    .switch input {{
      width: 18px;
      height: 18px;
      margin: 0;
    }}

    @media (max-width: 1100px) {{
      .stats {{
        grid-template-columns: repeat(3, minmax(96px, 1fr));
      }}

      main {{
        grid-template-columns: 360px minmax(0, 1fr);
      }}

      .detail-grid {{
        grid-template-columns: 1fr;
      }}

      .side {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}

    @media (max-width: 820px) {{
      header {{
        position: static;
        padding: 16px;
      }}

      .topline {{
        display: grid;
      }}

      .stats, .meta-grid, .filter-grid {{
        grid-template-columns: 1fr 1fr;
      }}

      main {{
        grid-template-columns: 1fr;
      }}

      aside {{
        border-right: 0;
        border-bottom: 1px solid var(--line);
        max-height: 72vh;
      }}

      .detail {{
        padding: 16px;
      }}

      .side {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="topline">
        <div>
          <h1>Chunk Preview</h1>
          <div class="subtle" id="generated"></div>
        </div>
        <div class="subtle" id="chunkPath"></div>
      </div>
      <section class="stats" id="stats"></section>
    </header>

    <main>
      <aside>
        <section class="filters">
          <input class="search" id="search" type="search" placeholder="搜尋 chunk、產品、規格或內文">
          <div class="filter-grid">
            <select id="sourceSelect" aria-label="source"></select>
            <select id="sectionSelect" aria-label="section"></select>
            <select id="tagSelect" aria-label="tag"></select>
            <select id="specSelect" aria-label="spec filter">
              <option value="all">全部 chunks</option>
              <option value="with_specs">有 specs</option>
              <option value="evidence">有 evidence</option>
              <option value="table">表格 chunk</option>
              <option value="has_lvds">has_lvds</option>
              <option value="has_mipi_dsi">has_mipi_dsi</option>
              <option value="has_mipi_csi">has_mipi_csi</option>
              <option value="ram_gb">ram_gb</option>
              <option value="flash_gb">flash_gb</option>
              <option value="uart_count">uart_count</option>
              <option value="i2c_count">i2c_count</option>
              <option value="spi_count">spi_count</option>
              <option value="can_count">can_count</option>
              <option value="usb_count">usb_count</option>
              <option value="ethernet_count">ethernet_count</option>
            </select>
          </div>
        </section>

        <section class="source-chart" id="sourceChart"></section>
        <div class="list-head">
          <span id="resultCount"></span>
          <span>最多顯示 300 筆</span>
        </div>
        <section class="chunk-list" id="chunkList"></section>
      </aside>

      <section class="detail" id="detail"></section>
    </main>
  </div>

  <script id="chunk-data" type="application/json">{data}</script>
  <script>
    const payload = JSON.parse(document.getElementById('chunk-data').textContent);
    const chunks = payload.chunks;
    const byId = new Map(chunks.map((chunk) => [chunk.chunk_id, chunk]));
    const state = {{
      search: '',
      source: 'all',
      section: 'all',
      tag: 'all',
      spec: 'all',
      selectedId: chunks[0]?.chunk_id || '',
      showParent: false,
    }};

    const el = {{
      generated: document.getElementById('generated'),
      chunkPath: document.getElementById('chunkPath'),
      stats: document.getElementById('stats'),
      search: document.getElementById('search'),
      sourceSelect: document.getElementById('sourceSelect'),
      sectionSelect: document.getElementById('sectionSelect'),
      tagSelect: document.getElementById('tagSelect'),
      specSelect: document.getElementById('specSelect'),
      sourceChart: document.getElementById('sourceChart'),
      resultCount: document.getElementById('resultCount'),
      chunkList: document.getElementById('chunkList'),
      detail: document.getElementById('detail'),
    }};

    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, (char) => ({{
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;',
      }}[char]));
    }}

    function escapeAttr(value) {{
      return escapeHtml(value);
    }}

    function escapeRegExp(value) {{
      return value.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
    }}

    function highlight(value) {{
      const safe = escapeHtml(value);
      const query = state.search.trim();
      if (!query) return safe;
      const terms = query.split(/\\s+/).filter(Boolean).slice(0, 6).map(escapeRegExp);
      if (!terms.length) return safe;
      const pattern = new RegExp(`(${{terms.join('|')}})`, 'gi');
      return safe.replace(pattern, '<mark>$1</mark>');
    }}

    function displayValue(value) {{
      if (Array.isArray(value)) return value.join(', ');
      if (typeof value === 'boolean') return value ? 'true' : 'false';
      if (value === null || value === undefined || value === '') return '';
      return String(value);
    }}

    function optionList(select, label, rows) {{
      const options = [`<option value="all">${{label}}</option>`];
      rows
        .filter(([name]) => name)
        .forEach(([name, count]) => {{
          options.push(`<option value="${{escapeAttr(name)}}">${{escapeHtml(name)}} (${{count}})</option>`);
        }});
      select.innerHTML = options.join('');
    }}

    function renderStats() {{
      const stats = payload.stats;
      const items = [
        ['Chunks', stats.chunk_count],
        ['Sources', stats.source_count],
        ['Parents', stats.parent_count],
        ['Tags', stats.tag_count],
        ['Token mean', stats.token_mean],
        ['Token range', `${{stats.token_min}} - ${{stats.token_max}}`],
      ];
      el.generated.textContent = `generated_at: ${{payload.generated_at}}`;
      el.chunkPath.textContent = payload.chunk_path;
      el.stats.innerHTML = items.map(([label, value]) => `
        <div class="metric"><b>${{escapeHtml(value)}}</b><span>${{escapeHtml(label)}}</span></div>
      `).join('');
    }}

    function renderSourceChart() {{
      const max = Math.max(1, ...payload.sources.map(([, count]) => count));
      const allActive = state.source === 'all' ? ' active' : '';
      const rows = [`
        <button class="source-row${{allActive}}" data-source="all" type="button">
          <span class="source-name">All sources</span>
          <span class="source-count">${{chunks.length}}</span>
          <span class="bar"><i style="--w:100%"></i></span>
        </button>
      `];
      payload.sources.forEach(([source, count]) => {{
        const active = state.source === source ? ' active' : '';
        const width = Math.max(4, Math.round((count / max) * 100));
        rows.push(`
          <button class="source-row${{active}}" data-source="${{escapeAttr(source)}}" type="button" title="${{escapeAttr(source)}}">
            <span class="source-name">${{escapeHtml(source)}}</span>
            <span class="source-count">${{count}}</span>
            <span class="bar"><i style="--w:${{width}}%"></i></span>
          </button>
        `);
      }});
      el.sourceChart.innerHTML = rows.join('');
      el.sourceChart.querySelectorAll('[data-source]').forEach((button) => {{
        button.addEventListener('click', () => {{
          state.source = button.dataset.source;
          el.sourceSelect.value = state.source;
          render();
        }});
      }});
    }}

    function searchableText(chunk) {{
      return [
        chunk.chunk_id,
        chunk.source,
        chunk.section,
        chunk.title,
        chunk.product,
        chunk.product_codes.join(' '),
        chunk.text_product_codes.join(' '),
        chunk.source_product_codes.join(' '),
        chunk.chip_models.join(' '),
        chunk.vendors.join(' '),
        chunk.tags.join(' '),
        JSON.stringify(chunk.specs),
        JSON.stringify(chunk.spec_evidence),
        chunk.text,
      ].join('\\n').toLowerCase();
    }}

    function matchesSpec(chunk) {{
      if (state.spec === 'all') return true;
      if (state.spec === 'with_specs') return Object.keys(chunk.specs).length > 0;
      if (state.spec === 'evidence') return chunk.evidence_count > 0;
      if (state.spec === 'table') return Boolean(chunk.table_context);
      return chunk.specs[state.spec] !== undefined && chunk.specs[state.spec] !== null && chunk.specs[state.spec] !== false;
    }}

    function filteredChunks() {{
      const query = state.search.trim().toLowerCase();
      const terms = query ? query.split(/\\s+/).filter(Boolean) : [];
      return chunks.filter((chunk) => {{
        if (state.source !== 'all' && chunk.source !== state.source) return false;
        if (state.section !== 'all' && (chunk.section || chunk.title) !== state.section) return false;
        if (state.tag !== 'all' && !chunk.tags.includes(state.tag)) return false;
        if (!matchesSpec(chunk)) return false;
        if (!terms.length) return true;
        const haystack = searchableText(chunk);
        return terms.every((term) => haystack.includes(term));
      }});
    }}

    function smallChips(chunk) {{
      const chips = [];
      if (Object.keys(chunk.specs).length) chips.push('<span class="chip good">specs</span>');
      if (chunk.evidence_count) chips.push(`<span class="chip good">evidence ${{chunk.evidence_count}}</span>`);
      if (chunk.table_context) chips.push('<span class="chip warn">table</span>');
      chunk.tags.slice(0, 4).forEach((tag) => chips.push(`<span class="chip">${{escapeHtml(tag)}}</span>`));
      return chips.join('');
    }}

    function renderList(results) {{
      el.resultCount.textContent = `${{results.length}} chunks`;
      const visible = results.slice(0, 300);
      if (!results.some((chunk) => chunk.chunk_id === state.selectedId)) {{
        state.selectedId = results[0]?.chunk_id || '';
      }}
      el.chunkList.innerHTML = visible.map((chunk) => {{
        const active = chunk.chunk_id === state.selectedId ? ' active' : '';
        const label = chunk.section || chunk.title || chunk.chunk_id;
        return `
          <button class="chunk-item${{active}}" data-id="${{escapeAttr(chunk.chunk_id)}}" type="button">
            <span class="chunk-title">
              <span class="page-pill">p.${{escapeHtml(chunk.page_label)}}</span>
              <b>${{highlight(label)}}</b>
            </span>
            <span class="chunk-meta">${{escapeHtml(chunk.source)}} · #${{escapeHtml(chunk.chunk_index)}} · ${{escapeHtml(chunk.token_count)}} tokens</span>
            <span class="mini-tags">${{smallChips(chunk)}}</span>
          </button>
        `;
      }}).join('') || '<div class="empty">沒有符合條件的 chunk。</div>';

      el.chunkList.querySelectorAll('[data-id]').forEach((button) => {{
        button.addEventListener('click', () => {{
          state.selectedId = button.dataset.id;
          render();
        }});
      }});
    }}

    function metadataGrid(chunk) {{
      const rows = [
        ['source', chunk.source],
        ['page', chunk.page_label],
        ['tokens', chunk.token_count],
        ['chars', `${{chunk.char_start ?? ''}} - ${{chunk.char_end ?? ''}}`],
        ['product', chunk.product],
        ['chips', chunk.chip_models.join(', ')],
        ['vendors', chunk.vendors.join(', ')],
        ['parent', chunk.parent_child_count ? `${{chunk.parent_child_index}} / ${{chunk.parent_child_count}}` : ''],
      ];
      return rows.map(([key, value]) => `
        <div class="kv"><span>${{escapeHtml(key)}}</span><b>${{escapeHtml(displayValue(value) || '-')}}</b></div>
      `).join('');
    }}

    function chipBlock(chunk) {{
      const items = [
        ...chunk.tags.map((tag) => ['chip', tag]),
        ...chunk.product_codes.map((code) => ['chip good', code]),
        ...chunk.text_product_codes.map((code) => ['chip good', code]),
        ...chunk.source_product_codes.map((code) => ['chip good', code]),
      ];
      if (chunk.table_context) items.unshift(['chip warn', chunk.table_context]);
      if (chunk.language) items.unshift(['chip', chunk.language]);
      return items.map(([klass, label]) => `<span class="${{klass}}">${{escapeHtml(label)}}</span>`).join('');
    }}

    function specRows(specs) {{
      const entries = Object.entries(specs);
      if (!entries.length) return '<div class="empty">這個 chunk 沒有抽到 specs。</div>';
      return entries.map(([key, value]) => `
        <div class="spec-row">
          <div class="spec-key">${{escapeHtml(key)}}</div>
          <div class="spec-value">${{escapeHtml(displayValue(value))}}</div>
        </div>
      `).join('');
    }}

    function evidenceRows(evidence) {{
      const entries = Object.entries(evidence || {{}}).filter(([, value]) => Array.isArray(value) ? value.length : value);
      if (!entries.length) return '<div class="empty">這個 chunk 沒有 evidence。</div>';
      return entries.map(([key, value]) => {{
        const lines = Array.isArray(value) ? value : [value];
        return `
          <div class="spec-row">
            <div class="spec-key">${{escapeHtml(key)}}</div>
            <div class="spec-value">${{lines.map((line) => highlight(line)).join('<br>')}}</div>
          </div>
        `;
      }}).join('');
    }}

    function neighborButton(label, id) {{
      if (!id || !byId.has(id)) return '';
      return `<button class="action" type="button" data-jump="${{escapeAttr(id)}}">${{label}}</button>`;
    }}

    function renderDetail() {{
      const chunk = byId.get(state.selectedId);
      if (!chunk) {{
        el.detail.innerHTML = '<div class="empty">選一個 chunk 來預覽。</div>';
        return;
      }}
      const label = chunk.section || chunk.title || chunk.chunk_id;
      const text = state.showParent && chunk.parent_text ? chunk.parent_text : chunk.text;
      const parentNote = state.showParent && chunk.parent_text ? `parent_text · ${{chunk.parent_token_count}} tokens` : 'child text';
      el.detail.innerHTML = `
        <div class="detail-grid">
          <article class="panel">
            <div class="panel-head">
              <div>
                <h2>${{escapeHtml(label)}}</h2>
                <div class="subtle">${{escapeHtml(chunk.chunk_id)}}</div>
              </div>
              <div class="actions">
                ${{neighborButton('Prev', chunk.prev_chunk_id)}}
                ${{neighborButton('Next', chunk.next_chunk_id)}}
                ${{chunk.pdf_href ? `<a class="action" href="${{escapeAttr(chunk.pdf_href)}}" target="_blank" rel="noreferrer">PDF</a>` : ''}}
              </div>
            </div>
            <div class="meta-grid">${{metadataGrid(chunk)}}</div>
            <div class="body">
              <div class="toggle-row">
                <div class="mini-tags">${{chipBlock(chunk)}}</div>
                <label class="switch">
                  <input id="parentToggle" type="checkbox" ${{state.showParent ? 'checked' : ''}}>
                  parent
                </label>
              </div>
              <div class="subtle" style="margin-bottom:10px">${{escapeHtml(parentNote)}}</div>
              <pre>${{highlight(text)}}</pre>
            </div>
          </article>

          <aside class="side">
            <section class="side-section">
              <h3>Specs</h3>
              <div class="side-body">${{specRows(chunk.specs)}}</div>
            </section>
            <section class="side-section">
              <h3>Evidence</h3>
              <div class="side-body">${{evidenceRows(chunk.spec_evidence)}}</div>
            </section>
          </aside>
        </div>
      `;

      el.detail.querySelectorAll('[data-jump]').forEach((button) => {{
        button.addEventListener('click', () => {{
          state.selectedId = button.dataset.jump;
          render();
        }});
      }});
      const toggle = document.getElementById('parentToggle');
      if (toggle) {{
        toggle.addEventListener('change', () => {{
          state.showParent = toggle.checked;
          renderDetail();
        }});
      }}
    }}

    function render() {{
      const results = filteredChunks();
      renderSourceChart();
      renderList(results);
      renderDetail();
    }}

    function init() {{
      renderStats();
      optionList(el.sourceSelect, '全部來源', payload.sources);
      optionList(el.sectionSelect, '全部章節', payload.sections);
      optionList(el.tagSelect, '全部 tags', payload.tags);

      el.search.addEventListener('input', () => {{
        state.search = el.search.value;
        render();
      }});
      el.sourceSelect.addEventListener('change', () => {{
        state.source = el.sourceSelect.value;
        render();
      }});
      el.sectionSelect.addEventListener('change', () => {{
        state.section = el.sectionSelect.value;
        render();
      }});
      el.tagSelect.addEventListener('change', () => {{
        state.tag = el.tagSelect.value;
        render();
      }});
      el.specSelect.addEventListener('change', () => {{
        state.spec = el.specSelect.value;
        render();
      }});
      render();
    }}

    init();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    chunk_path = Path(args.chunks)
    output_path = Path(args.output)

    records = load_jsonl(chunk_path)
    if args.source:
        records = [record for record in records if record.get("source") == args.source]
    if args.limit is not None:
        records = records[: max(args.limit, 0)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(records, chunk_path, output_path)
    output_path.write_text(render_html(payload), encoding="utf-8")
    print(f"Chunk preview written to: {output_path}")


if __name__ == "__main__":
    main()

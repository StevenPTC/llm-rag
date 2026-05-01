from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


NUMERIC_SPEC_FIELDS = [
    "uart_count",
    "i2c_count",
    "spi_count",
    "can_count",
    "usb_count",
    "ethernet_count",
    "ram_gb",
    "flash_gb",
]
BOOLEAN_SPEC_FIELDS = ["has_lvds", "has_mipi_dsi", "has_mipi_csi"]
LIST_SPEC_FIELDS = ["cpu_soc", "os_list", "power_input"]
SPEC_FIELDS = NUMERIC_SPEC_FIELDS + BOOLEAN_SPEC_FIELDS + LIST_SPEC_FIELDS
FALLBACK_VENDOR_NAMES = ["NXP", "Rockchip", "ST", "Realtek", "Renesas"]

# RAG 階段：資料前處理 / Retrieval
# 這個檔案負責把非結構化產品文件萃取成規格欄位，並把使用者問題解析成可比對條件。
# 對規格比較題而言，這層 deterministic logic 可減少 LLM 直接讀文字時的數值幻覺。

COUNT_ALIASES = {
    "uart_count": [r"UART"],
    "i2c_count": [r"I2C", r"IIC"],
    "spi_count": [r"SPI"],
    "can_count": [r"FDCAN", r"CAN"],
    "usb_count": [r"USB(?:\s*\d(?:\.\d)?|[0-9.]+)?"],
}

FIELD_ALIASES = {
    "uart_count": ["uart", "serial", "serial port"],
    "i2c_count": ["i2c", "iic"],
    "spi_count": ["spi"],
    "can_count": ["can", "fdcan"],
    "usb_count": ["usb"],
    "ethernet_count": ["ethernet", "lan", "gbe", "rgmii"],
    "ram_gb": ["ram", "memory", "ddr", "lpddr", "sdram", "記憶體"],
    "flash_gb": ["flash", "emmc", "rom", "storage", "儲存"],
    "has_lvds": ["lvds"],
    "has_mipi_dsi": ["mipi dsi", "dsi"],
    "has_mipi_csi": ["mipi csi", "csi"],
    "cpu_soc": ["cpu", "soc", "processor", "處理器"],
    "os_list": ["os", "yocto", "android", "linux", "作業系統"],
    "power_input": ["power input", "power", "dc", "電源"],
}

QUERY_FIELD_ALIASES = {
    "uart_count": [r"UART", r"串口", r"序列埠"],
    "i2c_count": [r"I2C", r"IIC"],
    "spi_count": [r"SPI"],
    "can_count": [r"FDCAN", r"CAN"],
    "usb_count": [r"USB"],
    "ethernet_count": [r"Ethernet", r"LAN", r"網路", r"乙太網路", r"RGMII", r"GbE"],
    "ram_gb": [r"RAM", r"Memory", r"記憶體"],
    "flash_gb": [r"Flash", r"eMMC", r"ROM", r"儲存"],
    "has_lvds": [r"LVDS"],
    "has_mipi_dsi": [r"MIPI\s*DSI", r"DSI"],
    "has_mipi_csi": [r"MIPI\s*CSI", r"CSI"],
    "os_list": [r"OS", r"Yocto", r"Android", r"Linux", r"作業系統"],
    "power_input": [r"Power\s*Input", r"Power", r"DC", r"電源"],
}

COMPARATOR_PATTERNS = {
    ">=": r"(?:以上|至少|不少於|大於等於|>=|≧)",
    ">": r"(?:超過|大於|>)",
    "<=": r"(?:以下|至多|不超過|小於等於|<=|≦)",
    "<": r"(?:少於|小於|<)",
    "==": r"(?:等於|剛好|正好|=)",
}

SUPPORT_WORD = r"(?:支援|支持|support|supports|with|具有|有|包含|include|includes)"
NEGATIVE_SUPPORT_WORD = r"(?:不支援|不支持|without|no)"


def empty_specs() -> dict[str, Any]:
    # RAG 階段：資料前處理
    # 統一 specs schema，讓每個產品即使缺資料也有穩定欄位。None 表示未知，
    # 不等於 False 或 0，避免後續條件比對誤判「沒有支援」。
    specs: dict[str, Any] = {}
    for field in NUMERIC_SPEC_FIELDS:
        specs[field] = None
    for field in BOOLEAN_SPEC_FIELDS:
        specs[field] = None
    for field in LIST_SPEC_FIELDS:
        specs[field] = []
    return specs


def unique_values(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        normalized = re.sub(r"\s+", " ", str(value)).strip(" ,;/")
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def normalize_vendor_names(vendors: list[str] | None) -> list[str]:
    return sorted(unique_values(vendors or FALLBACK_VENDOR_NAMES), key=len, reverse=True)


def ascii_keyword_pattern(value: str) -> str:
    return rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])"


def detect_vendor_mentions(text: str, known_vendors: list[str] | None = None) -> list[str]:
    # RAG 階段：Retrieval
    # vendor 偵測用於 metadata 補召回與 structured filter。額外的 typo recovery
    # 是為了處理查詢輸入的小拼字錯誤，提升產品查詢的容錯性。
    vendors = normalize_vendor_names(known_vendors)
    matches = []
    for vendor in vendors:
        if re.search(ascii_keyword_pattern(vendor), text, flags=re.IGNORECASE):
            matches.append(vendor)

    # Lenient typo recovery for longer vendor names, e.g. "Renzsas" -> "Renesas".
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9./-]*", text)
    for token in tokens:
        if len(token) < 4:
            continue
        for vendor in vendors:
            if len(vendor) < 4:
                continue
            similarity = SequenceMatcher(None, token.lower(), vendor.lower()).ratio()
            if similarity >= 0.84:
                matches.append(vendor)

    return unique_values(matches)


def normalize_for_specs(text: str) -> str:
    text = text.replace("\u00d7", "x").replace("\u00b2", "2")
    text = text.replace("\uff1a", ":")
    return re.sub(r"[ \t]+", " ", text)


def clean_evidence(text: str, limit: int = 180) -> str:
    # RAG 階段：資料前處理 / LLM 推論
    # evidence 需要短而可讀，否則 structured context 會被表格雜訊塞滿。
    # 清理 Row/Column 標記後仍保留原始規格片段，供回答時引用來源。
    text = normalize_for_specs(text)
    text = re.sub(r"\bColumn\s+\d+\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bRow\s+\d+\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" :")
    if len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


def evidence_key(text: str) -> str:
    return clean_evidence(text, limit=500).lower()


def add_evidence(evidence: dict[str, list[str]], field: str, snippet: str) -> None:
    # RAG 階段：資料前處理
    # 規格值必須附證據，否則 structured filter 雖然能篩選，LLM 卻無法說明依據。
    # 這裡同時去重，避免同一段表格文字反覆進入 prompt。
    snippet = clean_evidence(snippet)
    if not snippet:
        return
    existing = {item.lower() for item in evidence.setdefault(field, [])}
    if snippet.lower() not in existing:
        evidence[field].append(snippet)


def iter_windows(text: str, size: int = 3) -> list[str]:
    # RAG 階段：資料前處理
    # 規格常跨多行，例如欄名在前一行、數值在下一行。用短視窗合併相鄰行，
    # 可以提高規則式抽取的召回率，同時控制誤配範圍。
    lines = [line.strip() for line in normalize_for_specs(text).splitlines() if line.strip()]
    windows = []
    for idx, line in enumerate(lines):
        windows.append(line)
        if size > 1:
            window = " ".join(lines[idx : idx + size])
            if window != line:
                windows.append(window)
    return windows


def extract_interface_count(text: str, aliases: list[str], field: str, evidence: dict[str, list[str]]) -> int | None:
    # RAG 階段：資料前處理
    # 從產品文字中抽取 UART/I2C/SPI/CAN/USB 等介面數量。排除 M.2/PCIe 等情境，
    # 是避免把擴充槽描述誤算成板載介面數。
    total = 0
    matched_keys = set()
    alias_pattern = r"(?:%s)" % "|".join(aliases)
    lines = [line.strip() for line in normalize_for_specs(text).splitlines() if line.strip()]

    for window in lines:
        if not re.search(alias_pattern, window, flags=re.IGNORECASE):
            continue
        if field in {"uart_count", "i2c_count", "spi_count", "can_count"} and re.search(
            r"\b(?:E-Key|M\.2|Mini-PCIe|PCIe|PCI\s*Express)\b",
            window,
            flags=re.IGNORECASE,
        ):
            continue

        patterns = [
            rf"(?P<count>\d+)\s*(?:x|X|個|組|路|port|ports)?\s*(?P<alias>{alias_pattern})\b",
            rf"\b(?P<alias>{alias_pattern})\s*(?:port|ports)?\s*[:=]\s*(?P<count>\d+)\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, window, flags=re.IGNORECASE):
                count = int(match.group("count"))
                tail = window[match.start() :]
                next_match = re.search(
                    rf"\s+\d+\s*(?:x|X|個|組|路|port|ports)?\s*{alias_pattern}\b",
                    tail[1:],
                    flags=re.IGNORECASE,
                )
                matched_fragment = tail[: next_match.start() + 1] if next_match else tail
                matched_fragment = matched_fragment.split(":", 1)[0]
                key = (count, evidence_key(matched_fragment))
                if key in matched_keys:
                    continue
                matched_keys.add(key)
                total += count
                add_evidence(evidence, field, window)

    return total or None


def extract_ethernet_count(text: str, evidence: dict[str, list[str]]) -> int | None:
    explicit = extract_interface_count(text, [r"Ethernet", r"GbE", r"RGMII"], "ethernet_count", evidence)
    if explicit:
        return explicit

    for window in iter_windows(text, size=2):
        if re.search(r"\b(?:Ethernet|LAN)\b", window, flags=re.IGNORECASE) and re.search(
            r"(?:10/100|1000|Gbps|Mbps|RGMII)", window, flags=re.IGNORECASE
        ):
            add_evidence(evidence, "ethernet_count", window)
            return 1
    return None


def gb_value(number: str, unit: str) -> float:
    value = float(number)
    if unit.lower() == "mb":
        return value / 1024
    return value


def extract_capacity_gb(text: str, field: str, evidence: dict[str, list[str]]) -> float | None:
    # RAG 階段：資料前處理
    # RAM 與 Flash 都會出現 GB/MB，但語境不同。include/exclude pattern 用來避免
    # 將 storage、SD card 或最大外接容量誤當成板載記憶體。
    if field == "ram_gb":
        include = r"\b(?:RAM|DDR|LPDDR|SDRAM|Memory)\b"
        exclude = r"\b(?:ROM|Flash|eMMC|Storage|Micro\s*SD|SD\s*Card)\b"
    else:
        include = r"\b(?:Flash|eMMC|ROM|On Board Flash|On Board ROM)\b"
        exclude = r"\b(?:External Storage|Micro\s*SD|SD\s*Card|max)\b"

    best_value = None
    best_window = ""
    for window in iter_windows(text, size=3):
        if not re.search(include, window, flags=re.IGNORECASE):
            continue
        if re.search(exclude, window, flags=re.IGNORECASE):
            continue
        match = re.search(r"(\d+(?:\.\d+)?)\s*(GB|G|MB)\b", window, flags=re.IGNORECASE)
        if not match:
            continue
        value = gb_value(match.group(1), "GB" if match.group(2).lower() == "g" else match.group(2))
        if best_value is None or value > best_value:
            best_value = value
            best_window = window

    if best_value is not None:
        add_evidence(evidence, field, best_window)
    return best_value


def extract_cpu_soc(text: str, evidence: dict[str, list[str]]) -> list[str]:
    values = [
        match.upper()
        for match in re.findall(
            r"\b(?:RK\d{4}[A-Z0-9]*|STM32[A-Z0-9]+|RTL\d+[A-Z0-9]*|RZ/[A-Z0-9]+)\b",
            text,
            flags=re.IGNORECASE,
        )
    ]
    values.extend(re.findall(r"\bi\.MX\s*[0-9A-Za-z]+(?:\s+\w+)?\b", text, flags=re.IGNORECASE))
    values = unique_values(values)
    if values:
        for window in iter_windows(text, size=3):
            if re.search(r"\b(?:CPU|Processor|SoC|RISC)\b", window, flags=re.IGNORECASE) and any(
                value.lower().replace(" ", "") in window.lower().replace(" ", "") for value in values
            ):
                add_evidence(evidence, "cpu_soc", window)
                break
    return values


def extract_os_list(text: str, evidence: dict[str, list[str]]) -> list[str]:
    values = []
    patterns = [
        r"\bAndroid(?:\s+\d+(?:\.\d+)*)?\b",
        r"\bYocto(?:\s+Linux)?\b",
        r"\bUbuntu(?:\s+\d+(?:\.\d+)*)?\b",
        r"\bDebian(?:\s+\d+(?:\.\d+)*)?\b",
        r"\bLinux\b",
        r"\bWindows(?:\s+CE)?\b",
    ]
    for pattern in patterns:
        values.extend(match.group(0) for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    values = unique_values(values)
    if values:
        for window in iter_windows(text, size=3):
            if any(value.lower() in window.lower() for value in values):
                add_evidence(evidence, "os_list", window)
                break
    return values


def extract_power_input(text: str, evidence: dict[str, list[str]]) -> list[str]:
    values = []
    for window in iter_windows(text, size=4):
        if not re.search(r"(?:Power Input|Power|DC)", window, flags=re.IGNORECASE):
            continue
        voltage = re.search(
            r"(?:DC\s*)?\d+(?:\.\d+)?\s*V(?:\s*(?:~|-|to)\s*\d+(?:\.\d+)?\s*V)?",
            window,
            flags=re.IGNORECASE,
        )
        if voltage:
            value = voltage.group(0)
            if re.search(r"\bDC\b", window, flags=re.IGNORECASE) and not value.upper().startswith("DC"):
                value = f"DC {value}"
            values.append(value)
            add_evidence(evidence, "power_input", window)
    return unique_values(values)


def extract_specs_from_text(text: str) -> dict[str, Any]:
    # RAG 階段：資料前處理
    # 這是規格萃取入口，將 chunk/parent 文字轉成 specs + spec_evidence。
    # 萃取結果會寫入 chunk metadata，供 retrieval、structured filter 與 LLM evidence 使用。
    specs = empty_specs()
    evidence: dict[str, list[str]] = {}
    normalized = normalize_for_specs(text or "")

    for field, aliases in COUNT_ALIASES.items():
        specs[field] = extract_interface_count(normalized, aliases, field, evidence)
    specs["ethernet_count"] = extract_ethernet_count(normalized, evidence)
    specs["ram_gb"] = extract_capacity_gb(normalized, "ram_gb", evidence)
    specs["flash_gb"] = extract_capacity_gb(normalized, "flash_gb", evidence)
    specs["cpu_soc"] = extract_cpu_soc(normalized, evidence)
    specs["os_list"] = extract_os_list(normalized, evidence)
    specs["power_input"] = extract_power_input(normalized, evidence)

    boolean_patterns = {
        "has_lvds": r"\bLVDS\b",
        "has_mipi_dsi": r"\bMIPI\s*DSI\b|\bDSI\b",
        "has_mipi_csi": r"\bMIPI\s*CSI\b|\bCSI\b",
    }
    for field, pattern in boolean_patterns.items():
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        specs[field] = bool(match) if match else None
        if match:
            for window in iter_windows(normalized, size=2):
                if re.search(pattern, window, flags=re.IGNORECASE):
                    add_evidence(evidence, field, window)
                    break

    return {"specs": specs, "spec_evidence": evidence}


def source_product_label(record: dict) -> str:
    # RAG 階段：Retrieval
    # 產品彙整需要穩定 label。優先使用來源/文字中的產品代號，否則回退到檔名或 chunk 資訊，
    # 避免同一產品因不同 chunk metadata 稍有差異而無法聚合。
    codes = record.get("source_product_codes") or record.get("product_codes") or record.get("text_product_codes") or []
    if codes:
        return codes[0]
    source = record.get("source", "")
    return Path(source).stem if source else record.get("product") or record.get("chunk_id", "unknown")


def product_key(record: dict) -> str:
    return source_product_label(record).upper()


def merge_numeric(current: int | float | None, incoming: int | float | None) -> int | float | None:
    if incoming is None:
        return current
    if current is None:
        return incoming
    return max(current, incoming)


def merge_product_record(product: dict, record: dict) -> None:
    # RAG 階段：Retrieval
    # 同一產品的規格可能散落在多個 chunks。聚合時 numeric 取最大值、boolean 保留明確 True、
    # list 去重合併，目的是建立可被 structured filter 穩定查詢的產品級資料。
    product["records"].append(record)
    product["sources"].add(record.get("source", ""))
    for value in record.get("product_codes", []) + record.get("text_product_codes", []) + record.get("source_product_codes", []):
        product["product_codes"].add(value)
    for value in record.get("chip_models", []):
        product["specs"]["cpu_soc"] = unique_values(product["specs"]["cpu_soc"] + [value])
    for value in record.get("vendors", []):
        product["vendors"].add(value)

    extracted = record.get("specs_payload")
    if not extracted and (record.get("specs") or record.get("spec_evidence")):
        extracted = {"specs": record.get("specs", {}), "spec_evidence": record.get("spec_evidence", {})}
    if not extracted:
        text = "\n\n".join(value for value in [record.get("parent_text", ""), record.get("text", "")] if value)
        extracted = extract_specs_from_text(text)
    specs = extracted.get("specs", {})
    spec_evidence = extracted.get("spec_evidence", {})

    for field in NUMERIC_SPEC_FIELDS:
        product["specs"][field] = merge_numeric(product["specs"].get(field), specs.get(field))
    for field in BOOLEAN_SPEC_FIELDS:
        incoming = specs.get(field)
        if incoming is True:
            product["specs"][field] = True
        elif product["specs"].get(field) is None and incoming is False:
            product["specs"][field] = False
    for field in LIST_SPEC_FIELDS:
        incoming_values = specs.get(field) or []
        if isinstance(incoming_values, str):
            incoming_values = [incoming_values]
        product["specs"][field] = unique_values(product["specs"].get(field, []) + list(incoming_values))

    for field, snippets in spec_evidence.items():
        for snippet in snippets:
            product["evidence"].setdefault(field, [])
            entry = {
                "source": record.get("source", ""),
                "page": record.get("page", ""),
                "snippet": clean_evidence(snippet),
            }
            existing_keys = {(item["source"], item["page"], item["snippet"].lower()) for item in product["evidence"][field]}
            if (entry["source"], entry["page"], entry["snippet"].lower()) not in existing_keys:
                product["evidence"][field].append(entry)


def build_product_specs(records: list[dict]) -> list[dict]:
    # RAG 階段：Retrieval
    # 將 chunk-level metadata 彙整成 product-level specs。產品比較與列舉題不應只看單一 chunk，
    # 否則規格欄位容易因分散在不同頁面而被漏掉。
    products: dict[str, dict] = {}
    order = []
    for record in records:
        key = product_key(record)
        if key not in products:
            products[key] = {
                "key": key,
                "label": source_product_label(record),
                "product_codes": set(),
                "vendors": set(),
                "sources": set(),
                "records": [],
                "specs": empty_specs(),
                "evidence": {},
            }
            order.append(key)
        merge_product_record(products[key], record)

    output = []
    for key in order:
        product = products[key]
        product["product_codes"] = sorted(product["product_codes"])
        product["vendors"] = sorted(product["vendors"])
        product["sources"] = sorted(source for source in product["sources"] if source)
        output.append(product)
    return output


def comparator_from_text(text: str) -> str | None:
    for op, pattern in COMPARATOR_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            return op
    return None


def add_condition(plan: dict, field: str, op: str, value: Any, text: str) -> None:
    # RAG 階段：Retrieval
    # 將自然語言中的條件轉成標準化 predicate，並去除重複條件。
    # text 保留原始片段，方便 debug 使用者問題如何被解析。
    condition = {"field": field, "op": op, "value": value, "text": clean_evidence(text, limit=100)}
    key = (field, op, str(value).lower())
    existing = {(item["field"], item["op"], str(item["value"]).lower()) for item in plan["conditions"]}
    if key not in existing:
        plan["conditions"].append(condition)


def parse_numeric_conditions(question: str, plan: dict) -> None:
    # RAG 階段：Retrieval
    # 將「至少 2 個 UART」「RAM 4GB 以上」等描述轉成可比較的 numeric conditions。
    # 預設使用 >=，符合多數規格篩選題的語意。
    q = normalize_for_specs(question)
    comparator_union = "|".join(COMPARATOR_PATTERNS.values())
    unit = r"(?:個|組|路|ports?|x|X)?"
    for field in ["uart_count", "i2c_count", "spi_count", "can_count", "usb_count", "ethernet_count"]:
        alias_union = "|".join(QUERY_FIELD_ALIASES[field])
        patterns = [
            rf"(?P<num>\d+)\s*{unit}\s*(?P<alias>{alias_union})\s*(?P<cmp>{comparator_union})?",
            rf"(?P<alias>{alias_union})\s*(?:至少|有|需|需要|>=|大於等於)?\s*(?P<num>\d+)\s*{unit}\s*(?P<cmp>{comparator_union})?",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, q, flags=re.IGNORECASE):
                op = comparator_from_text(match.group("cmp") or "") or ">="
                add_condition(plan, field, op, int(match.group("num")), match.group(0))

    for field in ["ram_gb", "flash_gb"]:
        alias_union = "|".join(QUERY_FIELD_ALIASES[field])
        patterns = [
            rf"(?P<alias>{alias_union}).{{0,12}}?(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>GB|G|MB)\s*(?P<cmp>{comparator_union})?",
            rf"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>GB|G|MB).{{0,12}}?(?P<alias>{alias_union})\s*(?P<cmp>{comparator_union})?",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, q, flags=re.IGNORECASE):
                op = comparator_from_text(match.group("cmp") or "") or ">="
                add_condition(plan, field, op, gb_value(match.group("num"), match.group("unit")), match.group(0))


def parse_boolean_conditions(question: str, plan: dict) -> None:
    # RAG 階段：Retrieval
    # LVDS/MIPI 等支援性問題需區分肯定與否定語意；若直接交給 dense retrieval，
    # 「support」與「without」附近文字可能仍被視為相似，造成錯誤推薦。
    for field in BOOLEAN_SPEC_FIELDS:
        alias_union = "|".join(QUERY_FIELD_ALIASES[field])
        negative = re.search(rf"{NEGATIVE_SUPPORT_WORD}.{{0,8}}(?:{alias_union})|(?:{alias_union}).{{0,8}}{NEGATIVE_SUPPORT_WORD}", question, flags=re.IGNORECASE)
        positive = re.search(rf"{SUPPORT_WORD}.{{0,8}}(?:{alias_union})|(?:{alias_union}).{{0,8}}{SUPPORT_WORD}", question, flags=re.IGNORECASE)
        if negative:
            add_condition(plan, field, "==", False, negative.group(0))
        elif positive:
            add_condition(plan, field, "==", True, positive.group(0))


def parse_contains_conditions(question: str, plan: dict, known_vendors: list[str] | None = None) -> None:
    # RAG 階段：Retrieval
    # OS、SoC、vendor、power input 等欄位適合用 contains 而不是數值比較。
    # 這讓 structured filter 可處理「支援 Yocto」「NXP 產品」這類包含式需求。
    os_patterns = [
        r"Yocto(?:\s+Linux)?",
        r"Android(?:\s+\d+(?:\.\d+)*)?",
        r"Ubuntu(?:\s+\d+(?:\.\d+)*)?",
        r"Debian(?:\s+\d+(?:\.\d+)*)?",
        r"Linux",
        r"Windows(?:\s+CE)?",
    ]
    for pattern in os_patterns:
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            add_condition(plan, "os_list", "contains", match.group(0), match.group(0))

    chip_patterns = [
        r"\b(?:RK\d{4}[A-Z0-9]*|STM32[A-Z0-9]+|RTL\d+[A-Z0-9]*|RZ/[A-Z0-9]+)\b",
        r"\bi\.MX\s*[0-9A-Za-z]+(?:\s+\w+)?\b",
    ]
    for pattern in chip_patterns:
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            add_condition(plan, "cpu_soc", "contains", match.group(0), match.group(0))

    for vendor in detect_vendor_mentions(question, known_vendors=known_vendors):
        add_condition(plan, "vendor", "contains", vendor, vendor)

    power_match = re.search(r"(?:DC\s*)?\d+(?:\.\d+)?\s*V(?:\s*(?:~|-|to)\s*\d+(?:\.\d+)?\s*V)?", question, flags=re.IGNORECASE)
    if power_match and re.search(r"(?:power|電源|input|輸入|DC)", question, flags=re.IGNORECASE):
        add_condition(plan, "power_input", "contains", power_match.group(0), power_match.group(0))


def parse_support_count_conditions(question: str, plan: dict) -> None:
    # RAG 階段：Retrieval
    # 當問題只問「是否支援 UART」而沒有數量時，轉成 count >= 1。
    # 這能讓支援性問題與數量型 specs schema 共用同一套比對邏輯。
    for field in ["uart_count", "i2c_count", "spi_count", "can_count", "usb_count", "ethernet_count"]:
        if any(condition["field"] == field for condition in plan["conditions"]):
            continue
        alias_union = "|".join(QUERY_FIELD_ALIASES[field])
        if re.search(rf"{SUPPORT_WORD}.{{0,8}}(?:{alias_union})|(?:{alias_union}).{{0,8}}{SUPPORT_WORD}", question, flags=re.IGNORECASE):
            add_condition(plan, field, ">=", 1, field)


def plan_specs_query(question: str, known_vendors: list[str] | None = None) -> dict:
    # RAG 階段：Retrieval
    # 將自然語言問題規劃成 structured query plan。若成功產生 conditions，
    # query.py 會走 deterministic product filtering，再交由 LLM 格式化回答。
    plan = {"conditions": [], "referenced_fields": []}
    parse_numeric_conditions(question, plan)
    parse_boolean_conditions(question, plan)
    parse_contains_conditions(question, plan, known_vendors=known_vendors)
    parse_support_count_conditions(question, plan)
    plan["referenced_fields"] = unique_values([condition["field"] for condition in plan["conditions"] if condition["field"] != "vendor"])
    plan["is_structured"] = bool(plan["conditions"])
    return plan


def compare_numeric(actual: Any, op: str, expected: int | float) -> bool:
    if actual is None:
        return False
    if op == ">=":
        return actual >= expected
    if op == ">":
        return actual > expected
    if op == "<=":
        return actual <= expected
    if op == "<":
        return actual < expected
    return actual == expected


def contains_value(actual: Any, expected: str) -> bool:
    if actual is None:
        return False
    expected_norm = str(expected).lower().replace(" ", "")
    values = actual if isinstance(actual, list) else [actual]
    return any(expected_norm in str(value).lower().replace(" ", "") for value in values)


def product_matches_plan(product: dict, plan: dict) -> bool:
    # RAG 階段：Retrieval
    # 對產品級 specs 套用 query plan。所有條件都必須通過，確保推薦清單符合使用者約束，
    # 而不是只因某個 chunk 語意相似就被納入。
    specs = product.get("specs", {})
    for condition in plan.get("conditions", []):
        field = condition["field"]
        op = condition["op"]
        expected = condition["value"]
        if field == "vendor":
            actual = product.get("vendors", [])
        else:
            actual = specs.get(field)

        if op in {">=", ">", "<=", "<", "=="} and field in NUMERIC_SPEC_FIELDS:
            if not compare_numeric(actual, op, expected):
                return False
        elif op == "==" and field in BOOLEAN_SPEC_FIELDS:
            if actual is not expected:
                return False
        elif op == "contains":
            if not contains_value(actual, expected):
                return False
        else:
            if actual != expected:
                return False
    return True


def condition_fields(plan: dict) -> list[str]:
    fields = []
    for condition in plan.get("conditions", []):
        field = condition["field"]
        if field != "vendor":
            fields.append(field)
    if "cpu_soc" not in fields:
        fields.insert(0, "cpu_soc")
    return unique_values(fields)


def format_spec_value(field: str, value: Any) -> str:
    if value is None or value == []:
        return "unknown"
    if field in {"ram_gb", "flash_gb"}:
        return f"{value:g}GB"
    if field in BOOLEAN_SPEC_FIELDS:
        return "yes" if value is True else "no"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def format_product_spec_line(product: dict, fields: list[str]) -> str:
    specs = product.get("specs", {})
    parts = [f"{field}={format_spec_value(field, specs.get(field))}" for field in fields]
    if product.get("vendors"):
        parts.append(f"vendor={', '.join(product['vendors'])}")
    return f"{product['label']}: " + ", ".join(parts)


def format_evidence_lines(product: dict, fields: list[str], max_per_field: int = 2) -> list[str]:
    lines = []
    for field in fields:
        entries = product.get("evidence", {}).get(field, [])[:max_per_field]
        for entry in entries:
            page = f"p.{entry['page']}" if entry.get("page") != "" else "page unknown"
            lines.append(f"- {field}: {entry.get('source', '')} {page}: {entry.get('snippet', '')}")
    return lines


def format_structured_context(plan: dict, products: list[dict], max_products: int = 20) -> str:
    # RAG 階段：LLM 推論
    # 把 deterministic filter 的結果轉成 LLM 可讀 context。LLM 只負責整理與說明，
    # 不重新判斷產品是否符合條件，降低規格問答的風險。
    fields = condition_fields(plan)
    condition_lines = [
        f"- {condition['field']} {condition['op']} {condition['value']}"
        for condition in plan.get("conditions", [])
    ]
    lines = ["Structured query plan:", *condition_lines, "", "Filtered matching products:"]
    if not products:
        lines.append("- No product passed all structured conditions.")
        return "\n".join(lines)

    for idx, product in enumerate(products[:max_products], start=1):
        lines.append(f"{idx}. {format_product_spec_line(product, fields)}")
        lines.extend(format_evidence_lines(product, fields))
    return "\n".join(lines)


def sort_products_by_retrieval(products: list[dict], retrieved_results: list[dict]) -> list[dict]:
    # RAG 階段：Retrieval
    # structured filter 決定「是否符合條件」，retrieval rank 決定「呈現順序」。
    # 這讓結果同時保有硬條件正確性與與使用者問題的相關度。
    rank_by_key = {}
    for rank, item in enumerate(retrieved_results, start=1):
        key = product_key(item)
        rank_by_key.setdefault(key, rank)
    return sorted(products, key=lambda product: (rank_by_key.get(product["key"], 10_000), product["label"]))

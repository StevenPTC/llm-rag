# PDF RAG Demo

這個專案示範一條最小可用的 RAG 流程，從讀取 PDF、切分文字、產生向量、建立索引，到最後用本機 LLM 問答。

目前專案預設：
- PDF 放在 `docs/`
- 中間產物放在 `data/`
- 向量檢索預設使用 `FAISS`
- 最終回答預設使用本機 `Ollama`，並在查詢時選擇本機已安裝模型
- PDF 前處理會保留頁碼、章節、char range、產品、版本、heading level、list/table context、tags 與 specs metadata
- 查詢流程支援 `Dense Retrieval -> Hybrid Search -> Reranker -> Structured Specs Filter -> LLM`
- 對產品規格、型號推薦、條件查詢，本專案採用 `RAG + 規格資料庫`：PDF chunk 用來找上下文，specs metadata 用來做精準判斷，LLM 用來產生自然語言答案

## 這個專案在做什麼

RAG 的核心流程是：

1. 讀取 PDF 文字
2. 把長文字切成較小的 chunks
3. 把 chunks 轉成向量 embeddings
4. 把向量存進索引
5. 查詢時先找出最相關的 chunks
6. 把這些 chunks 當作 context 交給 LLM 回答

這樣做的好處是，LLM 回答時不是只靠模型內部記憶，而是能參考你自己的 PDF 文件內容。

## 專案結構

```text
rag-project/
├─ docs/                  # 原始 PDF
├─ data/                  # 前處理、embedding、index 的輸出
├─ pyPDF.py               # PDF 文字抽取與 chunk 切分
├─ embedding.py           # 將 chunks 轉成 embeddings
├─ index.py               # 建立向量索引（FAISS / Chroma）
├─ query.py               # 查詢 RAG
├─ specs.py               # 規格 metadata 抽取、query planner、structured filter
├─ eval_retrieval.py      # 離線檢索評估
├─ eval/                  # 評估資料集範例
├─ rag_utils.py           # 共用工具函式
├─ requirements.txt       # Python 套件需求
└─ README.md
```

## 各檔案用途

### `pyPDF.py`

負責：
- 掃描 `docs/*.pdf`
- 逐頁抽取文字
- 清理多餘空白、頁碼、重複頁首頁尾與常見浮水印文字
- 修正 PDF 斷字，例如 `auto-\nmatically`
- 先合併同一份 PDF，再做可跨頁的語意切塊
- 以 token estimate 切 child chunk，預設 `chunk_size=320`、`overlap=50`
- 同時產生 parent chunk，預設 `parent_chunk_size=850`、`parent_overlap=100`
- 先在文件層偵測高層章節，再讓 child chunk 繼承 `section`
- 表格會轉成 row-level 文字，例如 `Row 1:`、`Voltage: 12V`
- 記錄 `char_start` / `char_end`，讓 chunk 可回查原文位置
- 偵測 `section` / `title` / `doc_type` / `product` / `product_codes` / `text_product_codes` / `source_product_codes` / `chip_models` / `vendors` / `version` / `tags` / `parent_id`
- 從 chunk/table/parent context 抽出 specs metadata，例如 `cpu_soc`、`uart_count`、`i2c_count`、`spi_count`、`can_count`、`usb_count`、`ethernet_count`、`ram_gb`、`flash_gb`、`has_lvds`、`has_mipi_dsi`、`has_mipi_csi`、`os_list`、`power_input`
- FAQ 文件若包含 `Q:` / `A:` 或 `Question:` / `Answer:`，會優先以 QA pair 作為 chunk

輸出：
- `data/pdf_pages.jsonl`
- `data/pdf_chunks.jsonl`

### `embedding.py`

負責：
- 讀取 `data/pdf_chunks.jsonl`
- 使用 `sentence-transformers` 產生每個 chunk 的向量
- embedding input 會包含 `title` / `section` / `product` / `product_codes` / `text_product_codes` / `source_product_codes` / `chip_models` / `vendors` / `version` / `tags` / `doc_type` / `table_context` 與 child `text`

輸出：
- `data/embeddings.npy`
- `data/embedding_metadata.jsonl`

### `index.py`

負責：
- 讀取 embeddings
- 建立向量索引

可選 backend：
- `faiss`
- `chroma`

輸出：
- FAISS:
  - `data/faiss.index`
  - `data/faiss_metadata.jsonl`
- Chroma:
  - `data/chroma_db/`

### `query.py`

負責：
- 對問題做 embedding
- 從向量索引中找出最相關的 child chunks
- 用 `prev_chunk_id` / `next_chunk_id` 補相鄰 child chunks 後再 rerank
- 將命中的 child chunks 合併成 compact parent contexts
- 若問題包含規格條件，query planner 會先把自然語言轉成條件，例如 `UART >= 3`、`has_lvds = true`、`ram_gb >= 2`、`os_list contains Yocto`
- structured filter 會用程式逐一比較產品級 specs metadata，只保留明確符合條件的產品
- LLM 只接收符合條件的產品摘要與規格證據，負責中文整理、條列、推薦語氣與補充說明，不負責判斷誰符合條件

預設：
- 檢索 backend: `faiss`
- LLM provider: `ollama`
- Chat model: 若未指定，會列出本機 Ollama 模型讓使用者選擇

### `eval_retrieval.py`

負責：
- 讀取標註好的離線評估集
- 比較 `dense` / `hybrid` / `rerank` / `final(compact parent context)` 各階段表現
- 計算 `Hit@K`、`MRR`、`mean first relevant rank`
- 輸出 rerank 相對於 dense baseline 的改善幅度

## 安裝需求

建議先啟用虛擬環境，再安裝套件：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` 包含：
- `PyMuPDF`
- `pdfplumber`
- `numpy`
- `sentence-transformers`
- `faiss-cpu`
- `chromadb`
- `openai`

## Ollama 使用方式

你目前是使用本機 Ollama，這個專案已經支援。

先確認模型已安裝：

```bash
ollama list
```

確認 Ollama server 有在回應：

```bash
curl http://localhost:11434/api/tags
```

確認模型目前是否已經載入記憶體：

```bash
ollama ps
```

如果你平常用類似這樣的指令：

```bash
ollama run qwen3.5:9b
```

那通常代表：
- 模型已經存在本機
- Ollama 服務也會一起被用到

本專案的 `query.py` 會直接呼叫 Ollama API：

```text
http://localhost:11434/api/chat
```

## 使用流程

建議使用虛擬環境中的 Python 執行：

```bash
source .venv/bin/activate
```

### 1. 抽取 PDF 文字

把 PDF 放進 `docs/` 之後執行：

```bash
.venv/bin/python pyPDF.py
```

輸出：
- `data/pdf_pages.jsonl`
- `data/pdf_chunks.jsonl`

### 2. 產生 embeddings

```bash
.venv/bin/python embedding.py
```

輸出：
- `data/embeddings.npy`
- `data/embedding_metadata.jsonl`

如果你想指定不同的 embedding model：

```bash
.venv/bin/python embedding.py --model sentence-transformers/all-MiniLM-L6-v2
```

### 3. 建立索引

預設使用 FAISS：

```bash
.venv/bin/python index.py --backend faiss
```

如果想改用 Chroma：

```bash
.venv/bin/python index.py --backend chroma
```

### 4. 查詢

使用預設設定查詢：

```bash
.venv/bin/python query.py "這份文件的 CPU 是什麼？"
```

如果想連續測試多個問題，可以進入互動式 CLI：

```bash
.venv/bin/python query.py --interactive
```

或直接不帶問題，也會進入互動模式：

```bash
.venv/bin/python query.py
```

在互動模式中輸入 `exit`、`quit` 或 `q` 可結束。

這會：
- 先做 dense retrieval（預設 top 20）
- 再做 lexical hybrid merge
- 若問題是在列出特定 vendor/product，會補入符合 `vendors` metadata 的候選並加權
- 再用 metadata-aware 文字做 reranker 排序
- 取 top 5 child hits 後，才用 `prev_chunk_id` / `next_chunk_id` 補相鄰 child chunks，去重並轉成 compact parent contexts
- 如果 query planner 偵測到規格條件，會把所有 metadata 彙整成產品級 specs database，再用程式做 structured filter
- 印出 retrieved contexts
- 規格條件型問題只會把符合條件的產品摘要與證據交給 LLM，不會把一堆 raw chunks 全塞進 prompt
- 列出本機 Ollama 模型讓你選擇
- 再把 context 交給選到的模型回答

如果已經知道要用哪個 Ollama 模型，可以直接指定：

```bash
.venv/bin/python query.py "這份文件的 CPU 是什麼？" --chat-model qwen3.5:9b
```

若模型支援 thinking，預設會關閉 thinking：

```bash
.venv/bin/python query.py "這份文件的 CPU 是什麼？" --think false
```

如果只想看檢索結果，不呼叫 LLM：

```bash
.venv/bin/python query.py "這份文件的 CPU 是什麼？" --retrieval-only
```

如果想把某一題的 retrieval 過程留成 debug 檔：

```bash
.venv/bin/python query.py "Which product uses RK3568?" --retrieval-only --debug-retrieval
```

輸出會寫到 `data/debug/retrieval_*.json`，並同步更新 `data/debug/retrieval_latest.json`。如果只想留檔、不想讓 terminal 印出 retrieved contexts：

```bash
.venv/bin/python query.py "Which product uses RK3568?" --retrieval-only --debug-retrieval --no-print-results
```

如果想檢查 chunk 切分本身：

```bash
.venv/bin/python inspect_chunks.py
```

預設輸出 `data/debug/chunk_inspection.md`。也可以只看單一 PDF 或輸出 JSONL：

```bash
.venv/bin/python inspect_chunks.py --source HRBS10007ZHC21DB01.pdf
.venv/bin/python inspect_chunks.py --format jsonl --output data/debug/chunk_inspection.jsonl
```

如果想手動調整檢索策略：

```bash
.venv/bin/python query.py "Which product uses RK3568?" --dense-top-k 20 --top-k 5 --adjacent-window 1
```

如果想控制最後交給 LLM 的 parent window 大小：

```bash
.venv/bin/python query.py "Which product uses RK3568?" --parent-context-tokens 700
```

產品清單型問題會按產品去重後多收幾個 context，預設最多 20 個：

```bash
.venv/bin/python query.py "請幫我尋找ST的產品有哪些，幫我條列出來" --product-list-top-k 20
```

規格條件型問題會先做 structured filter，再交給 LLM 格式化：

```bash
.venv/bin/python query.py "請找出 3 個 UART 以上且支援 Yocto 的產品"
.venv/bin/python query.py "有哪些產品支援 LVDS 且 RAM 2GB 以上？"
```

如果要暫時關閉 specs planner/filter：

```bash
.venv/bin/python query.py "請找出 3 個 UART 以上的產品" --disable-structured-filter
```

如果想暫時關閉 hybrid 或 reranker 做比較：

```bash
.venv/bin/python query.py "Which product uses RK3568?" --disable-hybrid --retrieval-only
.venv/bin/python query.py "Which product uses RK3568?" --disable-rerank --retrieval-only
.venv/bin/python query.py "Which product uses RK3568?" --disable-hybrid --disable-rerank --retrieval-only
```

### 5. 離線檢索評估

先準備一份 JSONL 評估集，可參考：

```bash
eval/retrieval_eval.sample.jsonl
```

每一行格式：

```json
{"query":"Which product uses RK3568?","targets":[{"product":"RK3568"}]}
```

`targets` 支援的欄位包含：
- `chunk_id`
- `source`
- `page`
- `start_page`
- `end_page`
- `product`
- `product_codes`
- `text_product_codes`
- `source_product_codes`
- `chip_models`
- `vendors`
- `version`
- `section`
- `tags`
- `text_contains`

執行評估：

```bash
.venv/bin/python eval_retrieval.py --dataset eval/retrieval_eval.sample.jsonl
```

如果要輸出完整報表 JSON：

```bash
.venv/bin/python eval_retrieval.py --dataset eval/retrieval_eval.sample.jsonl --output data/retrieval_eval_report.json
```

輸出會比較：
- `dense`
- `hybrid`
- `rerank`
- `final`，也就是補相鄰 chunk 並轉成 compact parent context 之後的結果

並顯示：
- `Hit@1 / Hit@3 / Hit@5`
- `MRR`
- `mean first relevant rank`
- `rerank` 與 `final` 相對 `dense` 的改善幅度

建議先用 20 到 50 題建立最小評估集，讓每題至少標一個 `targets` 條件。若 `dense` 沒 hit，優先看 chunk / embedding；若 `dense` hit 但 `rerank` 掉了，檢查 reranker；若 `rerank` hit 但 `final` 或 LLM 回答不對，通常是 parent context 或 prompt/生成問題。

如果想切到 Chroma：

```bash
.venv/bin/python query.py "這份文件的 CPU 是什麼？" --backend chroma
```

如果 Ollama API 不在預設位置：

```bash
.venv/bin/python query.py "這份文件的 CPU 是什麼？" --ollama-base-url http://localhost:11434
```

如果要改用 OpenAI：

```bash
.venv/bin/python query.py "這份文件的 CPU 是什麼？" --llm-provider openai --chat-model gpt-4.1-mini
```

這時需要設定：

```bash
export OPENAI_API_KEY=your_api_key
```

## 完整重建流程

當 `docs/` 裡的 PDF 有新增、刪除或內容更新時，建議從前處理開始完整重建：

```bash
.venv/bin/python pyPDF.py
.venv/bin/python embedding.py
.venv/bin/python index.py --backend faiss
```

只檢查檢索結果，不呼叫 LLM：

```bash
.venv/bin/python query.py "Which product uses RK3568?" --backend faiss --retrieval-only
```

確認檢索正常後，再呼叫 LLM：

```bash
.venv/bin/python query.py "Which product uses RK3568?" --backend faiss --chat-model qwen3.5:9b
```

如果使用 Chroma，索引與查詢都要指定同一個 backend：

```bash
.venv/bin/python index.py --backend chroma
.venv/bin/python query.py "Which product uses RK3568?" --backend chroma --retrieval-only
```

## 主要輸出檔案說明

### `data/pdf_pages.jsonl`

每行一筆頁級資料，例如：

```json
{"source":"P05D00107-00.pdf","file_path":"/home/steven/rag-project/docs/P05D00107-00.pdf","page":1,"text":"..."}
```

用途：
- 保留逐頁原文
- 方便除錯與回查頁碼

### `data/pdf_chunks.jsonl`

每行一筆 chunk 級資料，例如：

```json
{"chunk_id":"P05D00107-00.pdf-p1-1-c1","source":"P05D00107-00.pdf","page":1,"start_page":1,"end_page":1,"char_start":0,"char_end":696,"chunk_index":1,"chunk_role":"child","token_count":320,"section":"Hardware","parent_id":"P05D00107-00.pdf-p1-2-parent-1","parent_token_count":980,"parent_start_page":1,"parent_end_page":2,"text":"...","parent_text":"..."}
```

用途：
- `text` 是 child chunk；embedding 會把 `title` / `section` / `product` / `product_codes` / `text_product_codes` / `source_product_codes` / `chip_models` / `vendors` / `version` / `tags` / `specs` / `doc_type` / `table_context` 一起組進輸入
- `parent_text` 是 compact parent context 的來源，不會整段無條件塞進 LLM
- 是 RAG 最核心的中間資料
- 透過 `start_page` / `end_page` / `char_start` / `char_end` 回查來源位置
- 透過 `product` / `product_codes` / `text_product_codes` / `source_product_codes` / `chip_models` / `vendors` / `version` / `tags` 支援後續 filter 或 hybrid search
- 透過 `specs` / `spec_evidence` 支援 CPU/SoC、RAM、Flash/eMMC、UART、I2C、SPI、CAN、Ethernet、USB、LVDS、MIPI DSI、MIPI CSI、OS、Power Input 等常見 FAE 條件查詢

### `data/embeddings.npy`

NumPy 格式的向量矩陣。

用途：
- 每一列對應一個 chunk 的 embedding
- 供 `index.py` 建立向量索引

### `data/embedding_metadata.jsonl`

保留和向量對應的 chunk metadata。

用途：
- 讓檢索到的向量可以對回原始文字、來源 PDF、頁碼

### `data/faiss.index`

FAISS 向量索引本體。

用途：
- 查詢時快速找相似 chunks

### `data/faiss_metadata.jsonl`

FAISS 檢索結果對應的 metadata。

用途：
- 根據向量索引位置還原 chunk 文字與來源資訊

## 常見問題

### 1. `ModuleNotFoundError`

代表套件還沒安裝，先執行：

```bash
pip install -r requirements.txt
```

### 2. `query.py` 沒有輸出最終答案

可能原因：
- Ollama 沒有啟動
- 選擇的 Ollama 模型沒有安裝
- 使用的是 `openai` 但未設定 `OPENAI_API_KEY`

可以先檢查：

```bash
ollama list
ollama ps
curl http://localhost:11434/api/tags
```

### 3. 查得到內容，但答案不夠準

可以調整：
- `pyPDF.py` 的 `chunk_size`
- `pyPDF.py` 的 `overlap`
- `query.py` 的 `--top-k`
- embedding model
- `eval_retrieval.py` 的離線標註集來量測真實改善幅度

### 4. `python3 pyPDF.py` 找不到 `fitz`

通常代表目前 shell 沒有使用專案虛擬環境。請改用：

```bash
.venv/bin/python pyPDF.py
```

## 建議的最小操作順序

```bash
.venv/bin/python pyPDF.py
.venv/bin/python embedding.py
.venv/bin/python index.py --backend faiss
.venv/bin/python query.py "這份文件的 CPU 是什麼？"
```

如果這四步能順利跑完，就代表這個專案的第一版 RAG 流程已經通了。

# PDF RAG Demo

這個專案示範一條最小可用的 RAG 流程，從讀取 PDF、切分文字、產生向量、建立索引，到最後用本機 LLM 問答。

目前專案預設：
- PDF 放在 `docs/`
- 中間產物放在 `data/`
- 向量檢索預設使用 `FAISS`
- 最終回答預設使用本機 `Ollama`，並在查詢時選擇本機已安裝模型
- PDF 前處理會保留頁碼、章節、char range、產品、版本、heading level、list/table context 與 tags 等 metadata
- 查詢流程支援 `Dense Retrieval -> Hybrid Search -> Reranker -> Top K -> Adjacent Chunks -> LLM`

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
- 依 `chunk_size=800`、`overlap=120` 進行 span-based overlap
- 記錄 `char_start` / `char_end`，讓 chunk 可回查原文位置
- 偵測 `section` / `title` / `doc_type` / `product` / `version` / `tags`
- FAQ 文件若包含 `Q:` / `A:` 或 `Question:` / `Answer:`，會優先以 QA pair 作為 chunk

輸出：
- `data/pdf_pages.jsonl`
- `data/pdf_chunks.jsonl`

### `embedding.py`

負責：
- 讀取 `data/pdf_chunks.jsonl`
- 使用 `sentence-transformers` 產生每個 chunk 的向量

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
- 從向量索引中找出最相關的 chunks
- 將檢索結果交給 LLM 生成答案

預設：
- 檢索 backend: `faiss`
- LLM provider: `ollama`
- Chat model: 若未指定，會列出本機 Ollama 模型讓使用者選擇

### `eval_retrieval.py`

負責：
- 讀取標註好的離線評估集
- 比較 `dense` / `hybrid` / `rerank` / `final(with adjacent chunks)` 各階段表現
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

這會：
- 先做 dense retrieval（預設 top 20）
- 再做 lexical hybrid merge
- 再做 reranker 排序
- 取 top 5 主 chunk，並自動補入相鄰 chunk
- 印出 retrieved contexts
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

如果想手動調整檢索策略：

```bash
.venv/bin/python query.py "Which product uses RK3568?" --dense-top-k 20 --top-k 5 --adjacent-window 1
```

如果想暫時關閉 hybrid 或 reranker 做比較：

```bash
.venv/bin/python query.py "Which product uses RK3568?" --disable-hybrid --retrieval-only
.venv/bin/python query.py "Which product uses RK3568?" --disable-rerank --retrieval-only
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
- `final`

並顯示：
- `Hit@1 / Hit@3 / Hit@5`
- `MRR`
- `mean first relevant rank`
- `rerank` 與 `final` 相對 `dense` 的改善幅度

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
{"chunk_id":"P05D00107-00.pdf-p1-1-c1","source":"P05D00107-00.pdf","file_path":"/home/steven/rag-project/docs/P05D00107-00.pdf","page":1,"start_page":1,"end_page":1,"char_start":0,"char_end":696,"chunk_index":1,"section":"Hardware","title":"Hardware","doc_type":"datasheet","product":"P05D00107-00","version":"Yocto 4.0","language":"en","tags":["hardware","software","ethernet"],"text":"..."}
```

用途：
- 直接作為 embedding 的輸入
- 是 RAG 最核心的中間資料
- 透過 `start_page` / `end_page` / `char_start` / `char_end` 回查來源位置
- 透過 `product` / `version` / `tags` 支援後續 filter 或 hybrid search

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

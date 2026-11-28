"""
建立向量索引：比照老師 rag.ipynb 的流程
    PDF 逐頁抽文字 → 切塊(500/50) → nomic-embed-text-v1.5 向量化(正規化、加任務前綴) → 寫入 Qdrant
並把 app.py 的 6 個構面檢索問句也先轉成向量，存成 query_vectors.json（網站查詢時直接使用，
所以網站本身不需要載入 embedding 模型）。

用法（建議在 Google Colab 執行，免費 GPU 較快）：
    !pip install -q pypdf qdrant-client sentence-transformers einops pandas requests certifi
    import os
    os.environ["QDRANT_URL"] = "你的 Qdrant 叢集網址"
    os.environ["QDRANT_API_KEY"] = "你的 Qdrant API key"
    !python build_index.py --years 2023            # 只建 2023 年
    !python build_index.py --years 2021 2022 2023 2024 2025
    !python build_index.py --years 2023 --tickers 2884 2881   # 只建指定公司
    !python build_index.py --years 2023 --force    # 重建（先刪除舊資料）

需要放在同一個資料夾：app.py（讀取構面問句）、reports_index.csv（報告書連結）、reports/（本機 PDF，選用）。
跑完會產生 query_vectors.json，請把它上傳到 GitHub（和 app.py 同一層）。
"""
import argparse
import ast
import io
import json
import os
import re
import sys
import unicodedata
import uuid
from urllib.parse import urlparse

import pandas as pd
import requests
from pypdf import PdfReader

COLLECTION = os.getenv("QDRANT_COLLECTION", "esg_report_chunks")
MODEL = os.getenv("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
REFERER_OVERRIDE = {"todayir.com": "https://ir.ctbcholding.com/"}


# ---------- 與 app.py 相同的文字清理 / 切塊 ----------
_BAD_CHARS = re.compile(r"[^\t\n\x20-\x7e\xa0-\xff\u2010-\u2027\u2030-\u203a\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff]")


def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = _BAD_CHARS.sub(" ", text)
    return re.sub(r"\s+", " ", text)


def chunk_pages(pages, size=500, overlap=50):
    """逐頁切塊，回傳 [(頁碼, 文字), ...]；頁碼是 PDF 的第幾頁（從 1 開始）。"""
    out, step = [], max(1, size - overlap)
    for pno, page in enumerate(pages, start=1):
        page = clean_text(page)
        for i in range(0, len(page), step):
            piece = page[i:i + size]
            if not piece.strip() or (i > 0 and len(piece) <= overlap):
                continue
            out.append((pno, piece))
    return out


# ---------- 讀 app.py 裡的設定（不重複維護） ----------
def _literal_from_app(app_path, name):
    tree = ast.parse(open(app_path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"app.py 裡找不到 {name}")


# ---------- 報告書清單（與 app.py 的 load_library 邏輯一致，本機檔案優先） ----------
def load_library(base, companies):
    frames = []
    p = os.path.join(base, "reports_index.csv")
    if os.path.exists(p):
        df = pd.read_csv(p, dtype=str, encoding="utf-8-sig").fillna("")
        df = df.rename(columns={"股票代碼/名稱": "代碼", "股票代碼": "代碼", "代碼名稱": "代碼"})
        for c in ["代碼", "年度", "PDF來源", "備註"]:
            if c not in df.columns:
                df[c] = ""
        frames.append(df[["代碼", "年度", "PDF來源", "備註"]])
    rd = os.path.join(base, "reports")
    if os.path.isdir(rd):
        rows = []
        for f in sorted(os.listdir(rd)):
            m = re.match(r"(\d{4})[_\-](\d{4})", f)
            if f.lower().endswith(".pdf") and m:
                rows.append({"代碼": m.group(1), "年度": m.group(2), "PDF來源": f, "備註": "本機檔案"})
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=["代碼", "年度", "PDF來源", "備註"])
    df = pd.concat(frames[::-1], ignore_index=True)
    df["代碼"] = df["代碼"].map(lambda x: (re.match(r"\d{4}", str(x).strip()) or [None])[0])
    df = df[df["代碼"].notna()]
    df["代碼"] = df["代碼"] + ".TW"
    df["年度"] = df["年度"].astype(str).str.extract(r"(\d{4})")[0]
    df = df[df["代碼"].isin(companies) & df["年度"].notna() & (df["PDF來源"].str.strip() != "")]
    return df.drop_duplicates(["代碼", "年度"], keep="first").sort_values(["年度", "代碼"]).reset_index(drop=True)


def normalize_url(u):
    u = str(u).strip()
    m = re.search(r"drive\.google\.com/file/d/([\w-]+)", u)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    if "dropbox.com" in u:
        return u.replace("dl=0", "dl=1") if "dl=" in u else u + ("&" if "?" in u else "?") + "dl=1"
    return u


def fetch_pdf(src, base):
    if not src.lower().startswith("http"):
        with open(os.path.join(base, "reports", src), "rb") as f:
            return f.read()
    url = normalize_url(src)
    u = urlparse(url)
    ref = next((v for k, v in REFERER_OVERRIDE.items() if u.netloc.endswith(k)), f"{u.scheme}://{u.netloc}/")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Referer": ref}
    try:
        r = requests.get(url, headers=headers, timeout=120)
    except requests.exceptions.SSLError as e:
        if "Subject Key Identifier" not in str(e):  # 只針對憑證格式問題放寬
            raise
        r = requests.get(url, headers=headers, timeout=120, verify=False)
    r.raise_for_status()
    if r.content[:4] != b"%PDF":
        raise ValueError("下載的內容不是 PDF")
    return r.content


# ---------- Embedding（比照老師：sentence-transformers + 任務前綴 + 正規化） ----------
def prefixes(model_name):
    m = model_name.lower()
    if "nomic" in m:
        return "search_query: ", "search_document: "
    if "e5" in m:
        return "query: ", "passage: "
    return "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", help="要建索引的年度，例如 2023 2024；不填＝清單內全部")
    ap.add_argument("--tickers", nargs="*", help="只處理指定公司代碼，例如 2884 2881")
    ap.add_argument("--force", action="store_true", help="已有資料也重建（先刪除舊的）")
    ap.add_argument("--base", default=".", help="app.py / reports_index.csv 所在資料夾")
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    for k in ("QDRANT_URL", "QDRANT_API_KEY"):
        if not os.getenv(k):
            sys.exit(f"請先設定環境變數 {k}")

    from qdrant_client import QdrantClient
    from qdrant_client.models import (Distance, FieldCondition, Filter, FilterSelector, MatchValue,
                                      PayloadSchemaType, PointStruct, VectorParams)
    from sentence_transformers import SentenceTransformer

    app_path = os.path.join(args.base, "app.py")
    companies = _literal_from_app(app_path, "COMPANIES")
    rubric = _literal_from_app(app_path, "RUBRIC")

    print(f"載入 embedding 模型：{MODEL}")
    model = SentenceTransformer(MODEL, trust_remote_code=True)
    dim = model.get_sentence_embedding_dimension()
    q_prefix, d_prefix = prefixes(MODEL)

    def encode(texts):
        return model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)

    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=60)
    if not client.collection_exists(COLLECTION):
        client.create_collection(COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        print(f"已建立 collection：{COLLECTION}（{dim} 維、cosine）")
    for field in ("ticker", "year"):  # 過濾欄位要建索引，Qdrant Cloud 才允許用來過濾
        client.create_payload_index(COLLECTION, field_name=field, field_schema=PayloadSchemaType.KEYWORD)

    # 1) 構面檢索問句 → 向量（網站直接使用）
    qv = {dim_name: {"query": cfg["query"], "vector": encode([q_prefix + cfg["query"]])[0].tolist()}
          for dim_name, cfg in rubric.items()}
    with open(os.path.join(args.base, "query_vectors.json"), "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "query_prefix": q_prefix, "queries": qv}, f, ensure_ascii=False)
    print("已輸出 query_vectors.json（請上傳到 GitHub）")

    # 2) 逐份報告書建索引
    lib = load_library(args.base, companies)
    if args.years:
        lib = lib[lib["年度"].isin([str(y) for y in args.years])]
    if args.tickers:
        lib = lib[lib["代碼"].str[:4].isin([str(t)[:4] for t in args.tickers])]
    print(f"預計處理 {len(lib)} 份報告書")

    for r in lib.itertuples():
        flt = Filter(must=[FieldCondition(key="ticker", match=MatchValue(value=r.代碼)),
                           FieldCondition(key="year", match=MatchValue(value=r.年度))])
        label = f"{companies[r.代碼]} {r.年度}"
        existing = client.count(COLLECTION, count_filter=flt, exact=True).count
        if existing and not args.force:
            print(f"[略過] {label} 已有 {existing} 筆（要重建請加 --force）")
            continue
        try:
            reader = PdfReader(io.BytesIO(fetch_pdf(r.PDF來源, args.base)))
            pages = [(p.extract_text() or "") for p in reader.pages]
        except Exception as e:
            print(f"[失敗] {label}：{type(e).__name__}: {str(e)[:150]}")
            continue
        chunks = chunk_pages(pages, args.chunk, args.overlap)
        if len(chunks) < 20:
            print(f"[失敗] {label}：抽不到足夠文字（可能是掃描檔）")
            continue
        if existing:
            client.delete(COLLECTION, points_selector=FilterSelector(filter=flt))
        total = 0
        for i in range(0, len(chunks), args.batch):
            part = chunks[i:i + args.batch]
            vecs = encode([d_prefix + text for _, text in part])
            points = [PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{r.代碼}|{r.年度}|{i + j}")),
                vector=vecs[j].tolist(),
                payload={"ticker": r.代碼, "company": companies[r.代碼], "year": r.年度,
                         "page": pno, "chunk_idx": i + j, "text": text, "source": r.PDF來源, "model": MODEL})
                for j, (pno, text) in enumerate(part)]
            client.upsert(COLLECTION, points=points)
            total += len(points)
        print(f"[完成] {label}：{len(pages)} 頁 → {total} 個片段")
    print("全部完成。")


if __name__ == "__main__":
    main()

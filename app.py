import json, os, re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="臺灣金融業永續報告書評分比較平台", page_icon="🌱", layout="wide")

# ---------------- 設定 ----------------
def secret(key, default=""):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

COMPANIES = {
    "2880.TW": "華南金", "2881.TW": "富邦金", "2882.TW": "國泰金", "2883.TW": "凱基金",
    "2884.TW": "玉山金", "2885.TW": "元大金", "2886.TW": "兆豐金", "2887.TW": "台新新光金",
    "2890.TW": "永豐金", "2891.TW": "中信金", "2892.TW": "第一金", "5880.TW": "合庫金",
}

# 評分構面：query 用於語意檢索，keywords 用於判斷揭露深度
RUBRIC = {
    "PCAF 方法學": {"query": "依循 PCAF 碳核算金融聯盟方法學 資料品質分數",
                  "keywords": ["PCAF", "資料品質", "data quality", "歸因", "第二版"]},
    "投融資碳排揭露(範疇三類別15)": {"query": "投融資組合碳排放 範疇三 類別15 financed emissions tCO2e",
                  "keywords": ["範疇三", "投融資", "financed emissions", "類別15", "tCO2e", "噸CO2e", "授信", "投資組合"]},
    "盤查覆蓋率": {"query": "投融資碳盤查 資產覆蓋率 涵蓋比例",
                  "keywords": ["覆蓋率", "涵蓋", "占比", "%", "資產類別"]},
    "第三方確信": {"query": "會計師有限確信 第三方查證 ISO 14064 確信報告",
                  "keywords": ["確信", "查證", "assurance", "ISO 14064", "會計師"]},
    "減碳目標": {"query": "SBTi 科學基礎減碳目標 2050 淨零 投融資組合減碳",
                  "keywords": ["SBTi", "科學基礎", "淨零", "2050", "減碳目標", "net zero"]},
    "轉型機制": {"query": "氣候議合 撤資 內部碳定價 煤炭 排除 高碳排授信限制",
                  "keywords": ["議合", "撤資", "碳定價", "煤", "排除", "轉型", "授信政策"]},
}

# ---------------- 文本處理 ----------------
def extract_pdf(data: bytes) -> str:
    import io
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((p.extract_text() or "") for p in reader.pages)

def chunk_text(text, size=500, overlap=50):
    text = re.sub(r"\s+", " ", text)
    step = max(1, size - overlap)
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]

def fetch_pdf(url):
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.content

# ---------------- RAG 檢索 + 評分 ----------------
def retrieve(chunks, query, top_k=5):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
    mat = vec.fit_transform(chunks + [query])
    sims = cosine_similarity(mat[-1], mat[:-1]).ravel()
    idx = sims.argsort()[::-1][:top_k]
    return [chunks[i] for i in idx]

def rule_score(top_chunks, keywords):
    joined = " ".join(top_chunks).lower()
    hit = [k for k in keywords if k.lower() in joined]
    if not hit:
        return 0.0, [], ""
    score = min(5.0, 1 + 4 * len(hit) / len(keywords))
    best = max(top_chunks, key=lambda c: sum(k.lower() in c.lower() for k in keywords))
    return round(score, 1), hit, best

def llm_review(name, dim, evidence, endpoint, token, model):
    """只根據檢索到的原文評分，找不到證據就給 0 分。失敗回傳 None。"""
    prompt = (f"以下是 {name} 永續報告書中與『{dim}』相關的原文片段。\n"
              "只能依據這些片段評分，片段中沒有證據請給 0 分。\n"
              "請只回傳 JSON：{\"score\": 0-5 的數字, \"reason\": \"一句話理由\"}\n\n"
              + "\n---\n".join(evidence))
    try:
        r = requests.post(endpoint, timeout=40,
                          headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                          json={"model": model, "temperature": 0,
                                "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        j = json.loads(re.search(r"\{.*\}", txt, re.S).group())
        return float(j["score"]), str(j.get("reason", ""))
    except Exception:
        return None


def guess_company(text):
    text = str(text)
    for t, n in COMPANIES.items():
        aliases = {n, n.replace("金", ""), t[:4]}
        if n == "台新新光金":
            aliases |= {"台新", "新光"}
        if any(a and a in text for a in aliases):
            return f"{t} {n}"
    return ""

# 各金控官網網域（5880 合庫、2891 中信已確認；其餘請上線後抽查，若搜不到官網結果可在此修正）
OFFICIAL = {
    "2880.TW": ("華南金控", "hnfhc.com.tw"),
    "2881.TW": ("富邦金控", "fubon.com"),
    "2882.TW": ("國泰金控", "cathayholdings.com"),
    "2883.TW": ("凱基金控", "kgi.com.tw"),
    "2884.TW": ("玉山金控", "esunfhc.com"),
    "2885.TW": ("元大金控", "yuanta.com.tw"),
    "2886.TW": ("兆豐金控", "megaholding.com"),
    "2887.TW": ("台新金控", "taishinholdings.com.tw"),
    "2890.TW": ("永豐金控", "sinopac.com"),
    "2891.TW": ("中信金控", "ctbcholding.com"),
    "2892.TW": ("第一金控", "firstholding.com.tw"),
    "5880.TW": ("合庫金控", "tcfhc.com.tw"),
}

def resolve_company(kw):
    kw = kw.strip()
    if len(kw) < 2:
        return ""
    for t, n in COMPANIES.items():
        short = OFFICIAL[t][0]
        if kw in (t, t[:4]) or kw in n or n in kw or short in kw or n.replace("金", "") in kw:
            return t
    return ""

def _raw_search(q, n=10):
    key = secret("SERPER_API_KEY")
    if key:
        r = requests.post("https://google.serper.dev/search", timeout=20,
                          headers={"X-API-KEY": key, "Content-Type": "application/json"},
                          json={"q": q, "gl": "tw", "hl": "zh-tw", "num": n})
        r.raise_for_status()
        return [{"title": x.get("title", ""), "url": x.get("link", ""), "snippet": x.get("snippet", "")}
                for x in r.json().get("organic", [])]
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    with DDGS() as d:
        hits = d.text(q, region="tw-tzh", max_results=n)
    return [{"title": h.get("title", ""), "url": h.get("href", ""), "snippet": h.get("body", "")} for h in hits]

def search_reports(keyword, year, n=10):
    """先鎖定該公司官網網域搜尋；找不到才退回一般搜尋並標示『非官網』。"""
    from urllib.parse import urlparse
    t = resolve_company(keyword)
    topic = f"永續報告書 {year}"
    out = []
    if t:
        short, domain = OFFICIAL[t]
        for q in (f"site:{domain} {short} {topic} filetype:pdf", f"site:{domain} {short} {topic}"):
            for h in _raw_search(q, n):
                if domain in urlparse(h["url"]).netloc:
                    h["來源"] = "官網"
                    out.append(h)
            if out:
                break
        if not out:
            full = short.replace("金控", "金融控股公司")
            for h in _raw_search(f"{full} {topic} filetype:pdf", n):
                blob = h["title"] + h["url"] + h["snippet"]
                if COMPANIES[t][:2] in blob or short in blob:
                    h["來源"] = "非官網"
                    out.append(h)
    else:
        for h in _raw_search(f"{keyword} {topic} filetype:pdf", n):
            h["來源"] = "非官網"
            out.append(h)
    seen, uniq = set(), []
    for h in out:
        if h["url"] not in seen:
            seen.add(h["url"])
            h["company"] = f"{t} {COMPANIES[t]}" if t else ""
            uniq.append(h)
    uniq.sort(key=lambda h: (h["來源"] != "官網", year not in (h["url"] + h["title"]),
                             not h["url"].lower().endswith(".pdf")))
    return uniq

def load_report(url):
    data = fetch_pdf(url)
    if data[:4] != b"%PDF":
        raise ValueError("這個連結不是 PDF 檔")
    text = extract_pdf(data)
    if len(text) < 1000:
        raise ValueError("PDF 幾乎抽不到文字（可能是掃描檔）")
    return text

def run_scoring(tickers):
    rows, evidence = [], []
    bar = st.progress(0.0)
    for i, t in enumerate(tickers):
        chunks = chunk_text(st.session_state["docs"][t], chunk_size, overlap)
        row = {"代碼": t, "公司": COMPANIES[t]}
        for dim, cfg in RUBRIC.items():
            top = retrieve(chunks, cfg["query"], top_k)
            sc_, hit, best = rule_score(top, cfg["keywords"])
            reason = f"命中關鍵詞：{', '.join(hit) or '無'}"
            if use_llm and token and hit:
                r = llm_review(COMPANIES[t], dim, top, endpoint, token, model)
                if r:
                    sc_, reason = r
            row[dim] = sc_
            evidence.append({"公司": COMPANIES[t], "構面": dim, "分數": sc_, "說明": reason, "原文佐證": best[:300]})
        row["總分(平均)"] = round(float(np.mean([row[d] for d in RUBRIC])), 2)
        rows.append(row)
        bar.progress((i + 1) / len(tickers))
    st.session_state["scores"] = pd.DataFrame(rows)
    st.session_state["evidence"] = pd.DataFrame(evidence)


def get_prices(tickers, start, end):
    """優先用 yfinance(還原權息價)；失敗或空值時改用 FinMind 日收盤價。"""
    try:
        df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame(tickers[0])
        if not df.dropna(how="all").empty:
            return df, "yfinance"
    except Exception:
        pass
    fm = secret("FINMIND_TOKEN")
    cols = {}
    for t in tickers:
        r = requests.get("https://api.finmindtrade.com/api/v4/data", timeout=30,
                         headers={"Authorization": f"Bearer {fm}"},
                         params={"dataset": "TaiwanStockPrice", "data_id": t.split(".")[0],
                                 "start_date": str(start), "end_date": str(end)})
        r.raise_for_status()
        d = pd.DataFrame(r.json().get("data", []))
        if d.empty:
            raise ValueError(f"FinMind 查無 {t} 的股價")
        cols[t] = d.set_index(pd.to_datetime(d["date"]))["close"]
    return pd.DataFrame(cols), "FinMind(未還原權息)"

# ---------------- 回測 ----------------
def metrics(ret, cum):
    years = max(len(ret) / 252, 1e-9)
    cagr = cum.iloc[-1] ** (1 / years) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (cagr - 0.015) / vol if vol else np.nan
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    var95 = np.percentile(ret, 5)
    return cagr * 100, vol * 100, sharpe, mdd * 100, var95 * 100

# ---------------- 側邊欄 ----------------
st.sidebar.title("🌿 控制面板")
picked = st.sidebar.multiselect("選擇金控", list(COMPANIES), default=list(COMPANIES)[:5],
                                format_func=lambda t: f"{t} {COMPANIES[t]}")
top_k = st.sidebar.slider("每構面檢索片段數 (top-k)", 3, 10, 5)
chunk_size = st.sidebar.slider("Chunk 字數", 200, 1000, 500)
overlap = st.sidebar.slider("Overlap 字數", 0, 200, 50)

st.sidebar.subheader("🔑 LLM 複核（選用）")
use_llm = st.sidebar.checkbox("啟用 LLM 依檢索證據複核評分", value=False)
endpoint = st.sidebar.text_input("API Endpoint", secret("API_ENDPOINT", "https://api.openai.com/v1/chat/completions"))
token = st.sidebar.text_input("API Token", secret("CUSTOM_API_TOKEN"), type="password")
model = st.sidebar.text_input("模型名稱", secret("LLM_MODEL", "gpt-4o-mini"))

if st.sidebar.button("🔌 測試 API 連線"):
    try:
        r = requests.post(endpoint, timeout=20,
                          headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                          json={"model": model, "max_tokens": 5,
                                "messages": [{"role": "user", "content": "回覆OK"}]})
        if r.ok:
            st.sidebar.success("連線成功")
        else:
            st.sidebar.error(f"失敗 {r.status_code}：{r.text[:200]}")
    except Exception as e:
        st.sidebar.error(f"無法連線：{e}")

st.session_state.setdefault("docs", {})

st.title("🌱 臺灣金融業永續報告書：檢索、評分與比較")
t1, t2, t3, t4 = st.tabs(["📂 1 取得報告書", "🤖 2 RAG 評分", "📊 3 差異比較", "📈 4 因子回測"])

# ---------------- Tab1 ----------------
with t1:
    c1, c2, c3 = st.columns([3, 1, 1])
    kw = c1.text_input("輸入關鍵字（公司名稱或代碼，例如：玉山金、2884、富邦）")
    year = c2.selectbox("年度", [str(y) for y in range(datetime.now().year - 1, 2020, -1)])
    c3.write("")
    if c3.button("🔎 搜尋", type="primary") and kw.strip():
        try:
            with st.spinner("搜尋永續報告書中…"):
                res = search_reports(kw.strip(), year)
            df = pd.DataFrame(res)
            if df.empty:
                st.warning("沒有找到結果，換個關鍵字試試。")
                st.session_state.pop("results", None)
            else:
                df["歸屬公司"] = [r.company or guess_company(f"{r.title} {r.url}") for r in df.itertuples()]
                df = df.drop(columns=["company"])
                df.insert(0, "選取", False)
                df.loc[0, "選取"] = bool(df.loc[0, "來源"] == "官網")
                if not (df["來源"] == "官網").any():
                    st.warning("沒有找到該公司官網的結果，以下為一般搜尋結果，請自行確認是否為正確報告書。")
                st.session_state["results"] = df
        except Exception as e:
            st.error(f"搜尋失敗：{e}")

    if "results" in st.session_state:
        st.caption("勾選要分析的報告書，確認『歸屬公司』正確後按下方按鈕。")
        opts = [""] + [f"{t} {n}" for t, n in COMPANIES.items()]
        edited = st.data_editor(
            st.session_state["results"], use_container_width=True, hide_index=True,
            column_config={
                "url": st.column_config.LinkColumn("報告書連結", display_text="開啟 PDF"),
                "歸屬公司": st.column_config.SelectboxColumn("歸屬公司", options=opts),
                "title": "標題", "snippet": "摘要"},
            disabled=["title", "url", "snippet", "來源"])
        if st.button("📥 載入選取的報告書並評分", type="primary"):
            for r in edited[edited["選取"]].itertuples():
                if not r.歸屬公司:
                    st.warning(f"請先指定歸屬公司：{r.title[:30]}")
                    continue
                try:
                    with st.spinner(f"下載並解析：{r.歸屬公司}"):
                        st.session_state["docs"][r.歸屬公司[:7]] = load_report(r.url)
                except Exception as e:
                    st.error(f"{r.歸屬公司} 失敗：{e}")
            if st.session_state["docs"]:
                run_scoring(list(st.session_state["docs"]))
                st.success("評分完成！請到「步驟 2」看明細、「步驟 3」看公司比較。")

    with st.expander("搜尋不到？手動上傳 PDF"):
        ups = st.file_uploader("檔名請含代碼，例如 2884_玉山金.pdf", type="pdf", accept_multiple_files=True)
        for f in ups or []:
            m = re.search(r"(\d{4})", f.name)
            if m and f"{m.group(1)}.TW" in COMPANIES:
                st.session_state["docs"][f"{m.group(1)}.TW"] = extract_pdf(f.read())
        if ups and st.button("對上傳檔案評分"):
            run_scoring(list(st.session_state["docs"]))
    st.write("已載入：", {COMPANIES.get(k, k): f"{len(v):,} 字" for k, v in st.session_state["docs"].items()})

# ---------------- Tab2 ----------------
with t2:
    st.caption("評分只依據報告書原文：先語意檢索相關段落，再依揭露深度打分（可選 LLM 複核）；找不到證據即為 0 分。")
    if not st.session_state["docs"]:
        st.info("請先在步驟 1 搜尋並載入報告書。")
    else:
        if st.button("🔄 重新評分（調整參數後使用）"):
            run_scoring(list(st.session_state["docs"]))
        if "scores" in st.session_state:
            st.dataframe(st.session_state["scores"], use_container_width=True, hide_index=True)
            with st.expander("🔍 原文佐證"):
                st.dataframe(st.session_state["evidence"], use_container_width=True, hide_index=True)

# ---------------- Tab3 ----------------
with t3:
    if "scores" not in st.session_state:
        st.info("請先完成步驟 2。")
    else:
        sc = st.session_state["scores"].sort_values("總分(平均)", ascending=False)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(sc, x="公司", y="總分(平均)", color="總分(平均)",
                                   color_continuous_scale="Greens", range_y=[0, 5]), use_container_width=True)
        with c2:
            fig = go.Figure()
            for _, r in sc.iterrows():
                fig.add_trace(go.Scatterpolar(r=[r[d] for d in RUBRIC], theta=list(RUBRIC), fill="toself", name=r["公司"]))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 5])))
            st.plotly_chart(fig, use_container_width=True)
        heat = sc.set_index("公司")[list(RUBRIC)]
        st.plotly_chart(px.imshow(heat, text_auto=True, color_continuous_scale="Greens", aspect="auto"),
                        use_container_width=True)
        st.download_button("下載評分 CSV", sc.to_csv(index=False).encode("utf-8-sig"), "esg_scores.csv")

# ---------------- Tab4 ----------------
with t4:
    if "scores" not in st.session_state:
        st.info("請先完成步驟 2。")
    else:
        d1, d2 = st.columns(2)
        start = d1.date_input("起始日", datetime(2022, 1, 1))
        end = d2.date_input("結束日", datetime.now())
        st.warning("注意：用『最新』報告書分數回測過去績效存在前視偏誤(look-ahead bias)，結果僅供概念展示，不可視為可實際獲利的策略。")
        if st.button("執行回測", type="primary"):
            sc = st.session_state["scores"].set_index("代碼")["總分(平均)"]
            tickers = list(sc.index)
            try:
                px_df, src = get_prices(tickers, start, end)
                st.caption(f"股價來源：{src}")
                rets = px_df.pct_change().dropna(how="all").fillna(0)
                w_c = sc.reindex(rets.columns).fillna(0)
                w_c = w_c / w_c.sum()
                w_e = pd.Series(1 / len(rets.columns), index=rets.columns)
                r_c, r_e = (rets * w_c).sum(axis=1), (rets * w_e).sum(axis=1)
                cum = pd.DataFrame({"碳揭露評分加權": (1 + r_c).cumprod(), "等權重": (1 + r_e).cumprod()})
                mc, me = metrics(r_c, cum.iloc[:, 0]), metrics(r_e, cum.iloc[:, 1])
                names = ["CAGR %", "年化波動 %", "Sharpe", "最大回撤 %", "95% 日VaR %"]
                st.dataframe(pd.DataFrame({"評分加權": mc, "等權重": me}, index=names).round(2), use_container_width=True)
                st.plotly_chart(px.line(cum), use_container_width=True)
                dd = (cum - cum.cummax()) / cum.cummax() * 100
                st.plotly_chart(px.area(dd, labels={"value": "回撤 %"}), use_container_width=True)
            except Exception as e:
                st.error(f"回測失敗：{e}")

st.divider()
st.caption("評分為文本揭露程度之輔助指標，非投資建議。")

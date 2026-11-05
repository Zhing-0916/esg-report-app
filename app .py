import hashlib, json, os, re, unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# 關鍵詞規則：
#  - 一般中文詞：子字串比對；純英數詞（PCAF、2050）：前後不能緊鄰英數字，避免誤中（例如 PCAF 不會中 PCAFX）
#  - "re:顯示名稱::正規表示式"：進階規則，例如「覆蓋率後面要接著數字 %」，單純出現「%」不算命中
RUBRIC = {
    "PCAF 方法學": {"query": "依循 PCAF 碳核算金融聯盟方法學 資料品質分數",
                  "keywords": ["PCAF", "碳核算金融聯盟", "資料品質", "data quality", "第二版", "歸因因子"]},
    "投融資碳排揭露(範疇三類別15)": {"query": "投融資組合碳排放 範疇三 類別15 financed emissions tCO2e",
                  "keywords": ["範疇三", "投融資", "financed emissions", "re:類別15::(類別|category)\\s*15",
                               "tCO2e", "re:碳排數字(公噸CO2e)::\\d[\\d,.]*\\s*(公噸|噸)\\s*CO2e", "投資組合"]},
    "盤查覆蓋率": {"query": "投融資組合財務碳排放盤查 盤查覆蓋率 資產類別 涵蓋比例",
                  "keywords": ["re:覆蓋率+數字%::覆蓋率[^。；\\n]{0,25}\\d+(\\.\\d+)?\\s*%", "盤查覆蓋率", "盤查範圍", "資產類別",
                               "re:碳盤查涵蓋比例::(投融資|碳盤查|碳排)[^。；\\n]{0,30}(涵蓋|覆蓋)[^。；\\n]{0,10}\\d+(\\.\\d+)?\\s*%"]},
    "第三方確信": {"query": "會計師有限確信 第三方查證 ISO 14064 確信報告",
                  "keywords": ["確信", "查證", "assurance", "ISO 14064", "會計師", "第三方"]},
    "減碳目標": {"query": "SBTi 科學基礎減碳目標 2050 淨零 投融資組合減碳",
                  "keywords": ["SBTi", "科學基礎", "淨零", "2050", "減碳目標", "net zero"]},
    "轉型機制": {"query": "氣候議合 撤資 內部碳定價 煤炭 排除 高碳排授信限制",
                  "keywords": ["議合", "撤資", "碳定價", "煤炭", "轉型成熟度", "轉型計畫"]},
}

# ---------------- 網路請求（放寬 TLS 嚴格檢查） ----------------
# Python 3.13 預設啟用 VERIFY_X509_STRICT，部分台灣企業官網憑證缺少 Subject Key Identifier 會被拒絕。
# 這裡只關閉「嚴格格式檢查」，憑證鏈與網域仍然會驗證。
import ssl
import certifi
import urllib3
from requests.adapters import HTTPAdapter

class _LaxTLSAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context(cafile=certifi.where())
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)

SESSION = requests.Session()
SESSION.mount("https://", _LaxTLSAdapter())

# 某些網站（例如中信金的 IR 平台）只接受從自家網頁連過去的下載，需指定來源頁
REFERER_OVERRIDE = {"todayir.com": "https://ir.ctbcholding.com/"}

def referer_for(url):
    from urllib.parse import urlparse
    u = urlparse(url)
    for dom, ref in REFERER_OVERRIDE.items():
        if u.netloc.endswith(dom):
            return ref
    return f"{u.scheme}://{u.netloc}/"

_NET = {"skip_ssl": False, "relaxed": set()}  # 由主執行緒設定；背景執行緒只讀取，不呼叫 st 函式

def http_get(url, **kw):
    if _NET["skip_ssl"]:
        urllib3.disable_warnings()
        return requests.get(url, verify=False, **kw)
    try:
        return SESSION.get(url, **kw)
    except requests.exceptions.SSLError as e:
        # 只針對「憑證格式不符(缺少 Subject Key Identifier)」重試；其他憑證錯誤(過期、網域不符等)仍然報錯。
        if "Subject Key Identifier" not in str(e):
            raise
        from urllib.parse import urlparse
        _NET["relaxed"].add(urlparse(url).netloc)
        urllib3.disable_warnings()
        return requests.get(url, verify=False, **kw)

# ---------------- 文本處理 ----------------
def extract_pdf(data: bytes) -> str:
    import io
    reader = PdfReader(io.BytesIO(data))
    return "\f".join((p.extract_text() or "") for p in reader.pages)  # \f 分隔頁面，供後續標註頁碼

# 只保留：英數、常見標點、全形/半形符號、中日韓文字；其餘（亂碼、私用區字元、泰文/藏文等）一律移除
_BAD_CHARS = re.compile(r"[^\t\n\x20-\x7e\xa0-\xff\u2010-\u2027\u2030-\u203a\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff]")

def clean_text(text):
    """NFKC 正規化（全形英數→半形、下標 CO₂→CO2、康熙部首→一般漢字），再移除亂碼字元。"""
    text = unicodedata.normalize("NFKC", text)
    text = _BAD_CHARS.sub(" ", text)
    return re.sub(r"\s+", " ", text)

def chunk_text(text, size=500, overlap=50):
    """逐頁切塊：每塊開頭帶 [pN] 標籤（PDF 第 N 頁），之後佐證可以標示出處頁碼。"""
    chunks, step = [], max(1, size - overlap)
    for pno, page in enumerate(text.split("\f"), start=1):
        page = clean_text(page)
        for i in range(0, len(page), step):
            piece = page[i:i + size]
            if not piece.strip() or (i > 0 and len(piece) <= overlap):
                continue
            chunks.append(f"[p{pno}] {piece}")
    return chunks

_PAGE_TAG = re.compile(r"^\[p(\d+)\]\s*")

def fetch_pdf(url):
    from urllib.parse import urlparse
    u = urlparse(url)
    r = http_get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                                               "Referer": referer_for(url)})
    r.raise_for_status()
    return r.content

# ---------------- RAG 檢索 + 評分 ----------------
def _strip_punct(s):
    s = _PAGE_TAG.sub("", s)  # 頁碼標籤不參與相似度計算
    return re.sub(r"[\W_]+", " ", s.lower())  # 檢索比對時去掉標點符號，只看文字本身

def retrieve(chunks, query, top_k=5):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), preprocessor=_strip_punct)
    mat = vec.fit_transform(chunks + [query])
    sims = cosine_similarity(mat[-1], mat[:-1]).ravel()
    idx = [i for i in sims.argsort()[::-1][:top_k] if sims[i] >= globals().get("min_sim", 0.0)]
    return [chunks[i] for i in idx]  # 低於相似度門檻的段落直接捨棄（門檻 0 = 不限制）

def _kw_parts(k):
    if k.startswith("re:"):
        label, pat = k[3:].split("::", 1)
        return label, re.compile(pat, re.I)
    if re.fullmatch(r"[A-Za-z0-9 .\-]+", k):
        pat = re.escape(k).replace(r"\ ", r"\s*")
        return k, re.compile(r"(?<![A-Za-z0-9])" + pat + r"(?![A-Za-z0-9])", re.I)
    return k, re.compile(re.escape(k))

def rule_score(top_chunks, keywords):
    joined = " ".join(_PAGE_TAG.sub("", c) for c in top_chunks)
    pats = [_kw_parts(k) for k in keywords]
    hit = [lab for lab, p in pats if p.search(joined)]
    if not hit:
        return 0.0, [], ""
    score = min(5.0, 1 + 4 * len(hit) / len(keywords))
    best = max(top_chunks, key=lambda c: sum(bool(p.search(_PAGE_TAG.sub("", c))) for _, p in pats))
    pm = _PAGE_TAG.match(best)
    page_label = f"（PDF 第 {pm.group(1)} 頁）" if pm else ""
    best = _PAGE_TAG.sub("", best)
    m = next((p.search(best) for _, p in pats if p.search(best)), None)
    if m is None:
        snippet = best[:200]
    else:
        a, b = max(0, m.start() - 100), min(len(best), m.end() + 100)
        snippet = ("…" if a else "") + best[a:m.start()] + "【" + m.group() + "】" + best[m.end():b] + ("…" if b < len(best) else "")
    return round(score, 1), hit, page_label + snippet

@st.cache_resource
def _llm_cache():
    return {}  # (公司, 構面, 模型, 檢索原文的雜湊) -> (分數, 理由)；同樣的原文不會重複呼叫 LLM

def _llm_key(t, dim, top):
    return (COMPANIES[t], dim, model, hashlib.md5("\n".join(top).encode("utf-8")).hexdigest())

def make_tfidf(chunks):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), preprocessor=_strip_punct)
    return vec, vec.fit_transform(chunks)

def retrieve_fast(chunks, vec, mat, query, k):
    """用已建好的 TF-IDF 索引檢索（6 個構面共用同一份索引，不用重複建立）。"""
    sims = cosine_similarity(vec.transform([query]), mat).ravel()
    idx = [i for i in sims.argsort()[::-1][:k] if sims[i] >= globals().get("min_sim", 0.0)]
    return [chunks[i] for i in idx]

def llm_review(name, dim, evidence, endpoint, token, model):
    """只根據檢索到的原文評分，找不到證據就給 0 分。回傳 ((分數, 理由) 或 None, 失敗原因)。"""
    prompt = (f"你是金融業永續報告書評鑑助理。以下是 {name} 永續報告書中與『{dim}』相關的原文片段，"
              "每段開頭的 [pN] 代表 PDF 第 N 頁。\n"
              "規則：\n"
              "1. 只能根據提供的片段評分，不可使用片段以外的知識或自行推測。\n"
              "2. 片段中沒有相關證據時，score 必須是 0，reason 寫「根據目前資料無法判斷」。\n"
              "3. 評分標準：0=完全沒有證據；1-2=只有概略提及；3-4=有具體做法或數據；5=完整揭露，且有數據、方法與查證。\n"
              "4. reason 用一句話說明依據，並在 pages 列出你引用的頁碼。\n"
              "請只回傳 JSON：{\"score\": 0-5 的數字, \"reason\": \"一句話理由\", \"pages\": [頁碼數字]}\n\n"
              + "\n---\n".join(evidence))
    import random
    import time
    last = "未知錯誤"
    for attempt in range(4):
        try:
            r = requests.post(endpoint, timeout=60,
                              headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                              json={"model": model, "temperature": 0,
                                    "messages": [{"role": "user", "content": prompt}]})
        except requests.exceptions.Timeout:
            last = "連線逾時"
            time.sleep(2 * (attempt + 1))
            continue
        except Exception as e:
            last = f"連線錯誤 {type(e).__name__}"
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 429:  # 被限流
            m = re.search(r"\((TPM|TPD|RPM|RPD)\)", r.text or "")
            kind = m.group(1) if m else ""
            try:
                wait = float(r.headers.get("retry-after", ""))
            except ValueError:
                wait = 4 * (2 ** attempt)
            if kind in ("TPD", "RPD"):  # 每日額度用完，重試沒有意義
                return None, f"HTTP 429：今日額度已用完（{kind}）"
            if wait > 60:  # 要等太久，不在這裡空等
                return None, f"HTTP 429：需等待約 {int(wait)} 秒（{kind or '額度'}）"
            last = "HTTP 429 被限流" + (f"（{kind}）" if kind else "")
            time.sleep(min(wait, 30) + random.random())
            continue
        if r.status_code >= 500:
            last = f"HTTP {r.status_code} 伺服器錯誤"
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code >= 400:  # 金鑰、模型名稱、網址錯誤等：重試沒有用，直接回報
            return None, f"HTTP {r.status_code}：{r.text[:80]}"
        try:
            txt = r.json()["choices"][0]["message"]["content"] or ""
            j = json.loads(re.search(r"\{.*\}", txt, re.S).group())
            pages = [str(p) for p in (j.get("pages") or []) if str(p).isdigit()]
            reason = str(j.get("reason", "")) + (f"（依據 PDF 第 {', '.join(pages)} 頁）" if pages else "")
            return (max(0.0, min(5.0, float(j["score"]))), reason), ""
        except Exception:
            last = "回傳格式無法解析"
    return None, last


def guess_company(text):
    text = str(text)
    for t, n in COMPANIES.items():
        aliases = {n, n.replace("金", ""), t[:4]}
        if n == "台新新光金":
            aliases |= {"台新", "新光"}
        if any(a and a in text for a in aliases):
            return f"{t} {n}"
    return ""

# 各金控 (簡稱, 官網網域, 官網報告書下載頁)。有填『下載頁』的公司會直接從官網頁面抓 PDF 連結；
# 沒填的公司改用搜尋。5880 合庫、2891 中信網域已確認；其餘請自行抽查，下載頁網址可隨時補上。
OFFICIAL = {
    "2880.TW": ("華南金控", "hnfhc.com.tw", "https://esg.hnfhc.com.tw/WebPortalESG/csr/annual.do"),
    "2881.TW": ("富邦金控", "fubon.com", "https://www.fubon.com/financialholdings/citizenship/downloadlist/downloadlist_report.html"),
    "2882.TW": ("國泰金控", "cathayholdings.com", "https://www.cathayholdings.com/holdings/esg/reports/reports_download"),
    "2883.TW": ("凱基金控", "kgi.com", "https://www.kgi.com/zh-tw/esg/download"),
    "2884.TW": ("玉山金控", "esunfhc.com", "https://www.esunfhc.com/zh-tw/esg/resource-center/report"),
    "2885.TW": ("元大金控", "yuanta.com", "https://www.yuanta.com/TW/ESG/ESG-Report"),
    "2886.TW": ("兆豐金控", "megaholdings.com.tw", "https://esg.megaholdings.com.tw/frontend/download"),
    "2887.TW": ("台新金控", "tsholdings.com.tw", "https://www.tsholdings.com.tw/tsh/responsibility/report/"),
    "2890.TW": ("永豐金控", "sinopac.com", "https://www.sinopac.com/esg/tw/sustainability-report/"),
    "2891.TW": ("中信金控", "ctbcholding.com", "https://ir.ctbcholding.com/c/esg_reports"),
    "2892.TW": ("第一金控", "firstholding.com.tw", "https://csr.firstholding.com.tw/tc/csr_report.html"),
    "5880.TW": ("合庫金控", "tcfhc.com.tw", "https://esg.tcfhc.com.tw/report-download/"),
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

from html.parser import HTMLParser

class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self._cur = [], None
        self._row, self._head_buf, self._in_head, self._head = None, "", False, ""
    def handle_starttag(self, tag, attrs):
        if tag in ("tr", "li") and self._row is None:
            self._row = {"text": "", "links": []}
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._in_head, self._head_buf = True, ""
        elif tag == "a":
            a = dict(attrs)
            self._cur = {"href": a.get("href") or "", "title": a.get("title") or "", "text": "",
                         "row": "", "head": self._head}
            if self._row is not None:
                self._row["links"].append(self._cur)
    def handle_data(self, data):
        if self._cur is not None:
            self._cur["text"] += data
        if self._row is not None:
            self._row["text"] += data
        if self._in_head:
            self._head_buf += data
    def handle_endtag(self, tag):
        if tag == "a" and self._cur is not None:
            self.links.append(self._cur)
            self._cur = None
        elif tag in ("tr", "li") and self._row is not None:
            txt = " ".join(self._row["text"].split())
            for l in self._row["links"]:
                l["row"] = txt
            self._row = None
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6") and self._in_head:
            self._head = " ".join(self._head_buf.split())
            self._in_head = False

def crawl_report_page(url, year):
    """直接讀企業官網的報告書下載頁，收集 PDF 連結（含藏在 script 裡的、以及非 .pdf 結尾的下載連結）。"""
    from urllib.parse import urljoin, urlparse, unquote
    r = http_get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                                               "Accept-Language": "zh-TW,zh;q=0.9"})
    r.encoding = r.apparent_encoding or "utf-8"
    html = r.text.replace("\\/", "/")
    first_seg = "/" + urlparse(url).path.strip("/").split("/")[0].lower()
    roc = int(year) - 1911
    year_re = re.compile(rf"(?<!\d){year}(?!\d)|(?<!\d){roc}\s*年")
    p = _LinkParser()
    p.feed(html)
    out, seen = [], set()

    def add(raw_href, title_attr="", text="", loose=False, row="", head=""):
        raw_href = raw_href.strip()
        if not raw_href or raw_href.lower().startswith(("#", "javascript", "mailto", "tel")):
            return
        href = urljoin(url, raw_href).split("#")[0]
        if href in seen:
            return
        is_pdf = urlparse(href).path.lower().endswith(".pdf")
        if not is_pdf:
            label = f"{title_attr} {text}"
            if not (loose and re.search(r"download|file|attach|upload|\.do\?|\.aspx?|\.php", href, re.I)
                    and re.search(r"報告書|report|csr|永續", label, re.I)):
                return
        seen.add(href)
        fname = unquote(href.rsplit("/", 1)[-1].split("?")[0])
        title = (title_attr or text).replace("下載", "").replace("(開新視窗)", "").replace("(PDF)", "")
        title = re.sub(r"\s+", " ", title).strip()
        if len(title) <= 8:
            title = f"{title} {row[:30]} {fname}".strip()
        blob = f"{title} {href} {row} {head}"
        near = f"{title} {href} {row}"
        is_sus = bool(re.search(r"永續|CSR|ESG|sustainab", blob, re.I)) and not re.search(r"TCFD|TNFD|氣候", near, re.I)
        english = bool(re.search(r"[_\-]EN(?![A-Za-z])|English|英文", blob, re.I))
        parent = urlparse(href).path.lower().startswith(first_seg)
        note = ("永續報告書" if is_sus else "其他報告書（氣候/TCFD 等）") + ("｜英文版" if english else "")
        note += "｜來自官網報告書下載頁" + ("" if is_pdf else "｜連結非 .pdf 結尾，載入時會驗證是否為 PDF")
        out.append({"title": title, "url": href, "來源": "官網", "snippet": note,
                    "_sus": is_sus, "_year": bool(year_re.search(blob)), "_en": english,
                    "_parent": parent, "_loose": not is_pdf})

    for l in p.links:
        add(l["href"], l["title"], l["text"], loose=True, row=l["row"], head=l["head"])
    for m in re.findall(r'(?:https?:)?//[^\s"\'<>\\)]+?\.pdf|/[^\s"\'<>\\)]+?\.pdf', html, re.I):
        add(m if not m.startswith("//") else "https:" + m)
    # 有些網站的清單是 JavaScript 動態載入：改掃同網站的 script 檔，找裡面的 PDF 連結
    host = urlparse(url).netloc
    labels = host.split(".")
    root = ".".join(labels[-3:] if len(labels) >= 3 and labels[-2] in ("com", "org", "net", "gov", "edu") else labels[-2:])
    script_urls = [urljoin(url, x) for x in re.findall(r'<script[^>]+src=["\']([^"\']+)', html, re.I)]
    same_site = [x for x in script_urls if urlparse(x).netloc.endswith(root)]
    if not out:
        for sj in same_site[:8]:
            try:
                body = http_get(sj, timeout=15, headers={"User-Agent": "Mozilla/5.0"}).text.replace("\\/", "/")
            except Exception:
                continue
            for m in re.findall(r'(?:https?:)?//[^\s"\'<>\\)]+?\.pdf|[\w./-]+?\.pdf', body, re.I):
                add(m if not m.startswith("//") else "https:" + m)
    out.sort(key=lambda h: (h["_loose"], not h["_year"], not h["_sus"], h["_en"], not h["_parent"]))

    st.session_state["diag"] = {
        "頁面網址": url, "HTTP 狀態": r.status_code, "網頁內容長度": len(html),
        "頁面連結總數": len(p.links), "抓到的候選報告連結數": len(out),
        "前 15 個連結文字": [(l["text"] or l["title"]).strip()[:25] for l in p.links[:15]],
        "頁面載入的 script 檔": script_urls[:12],
        "HTML 開頭片段": " ".join(html[:400].split()),
    }
    return out

def search_reports(keyword, year, n=10):
    """1) 貼網址：直接讀該頁 PDF；2) 有官網下載頁：直接讀；3) 否則用官網網域搜尋；4) 最後才一般搜尋。"""
    from urllib.parse import urlparse
    kw = keyword.strip()
    st.session_state["crawl_err"] = None
    out, t = [], ""
    if kw.lower().startswith("http"):
        host = urlparse(kw).netloc
        t = next((k for k, v in OFFICIAL.items() if v[1] in host), "")
        if urlparse(kw).path.lower().endswith(".pdf"):
            out = [{"title": kw.rsplit("/", 1)[-1], "url": kw, "snippet": "你直接貼上的 PDF 連結", "來源": "官網" if t else "手動貼上"}]
        else:
            out = crawl_report_page(kw, year)
    else:
        t = resolve_company(kw)
        if t and OFFICIAL[t][2]:
            try:
                out = crawl_report_page(OFFICIAL[t][2], year)
            except Exception as e:
                out = []
                st.session_state["crawl_err"] = f"{type(e).__name__}: {str(e)[:250]}"
        if not out and t:
            short, domain, _ = OFFICIAL[t]
            topic = f"永續報告書 {year}"
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
        elif not t:
            for h in _raw_search(f"{kw} 永續報告書 {year} filetype:pdf", n):
                h["來源"] = "非官網"
                out.append(h)
    seen, uniq = set(), []
    for h in out:
        if h["url"] not in seen:
            seen.add(h["url"])
            h.pop("_sus", None)
            h.pop("_year", None)
            h.pop("_en", None)
            h.pop("_parent", None)
            h.pop("_loose", None)
            h["company"] = f"{t} {COMPANIES[t]}" if t else ""
            uniq.append(h)
    return uniq

def load_report(url):
    data = fetch_pdf(url)
    if data[:4] != b"%PDF":
        raise ValueError("這個連結不是 PDF 檔")
    text = extract_pdf(data)
    if len(text) < 1000:
        raise ValueError("PDF 幾乎抽不到文字（可能是掃描檔）")
    return text

# ---------------- 報告書資料庫 ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_COLS = ["代碼", "年度", "PDF來源", "備註"]

def normalize_url(u):
    """把 Google Drive / Dropbox 分享連結轉成可直接下載的連結。"""
    u = str(u).strip()
    m = re.search(r"drive\.google\.com/file/d/([\w-]+)", u)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    if "dropbox.com" in u:
        return u.replace("dl=0", "dl=1") if "dl=" in u else u + ("&" if "?" in u else "?") + "dl=1"
    return u

def _norm_code(x):
    m = re.match(r"\d{4}", str(x).strip())
    return f"{m.group()}.TW" if m else ""

def load_library():
    """合併 reports_index.csv（連結清單）與 reports/ 資料夾（檔名如 2884_2023.pdf）。"""
    frames = []
    p = os.path.join(BASE_DIR, "reports_index.csv")
    if os.path.exists(p):
        try:
            df = pd.read_csv(p, dtype=str, encoding="utf-8-sig").fillna("")
            df = df.rename(columns={"股票代碼/名稱": "代碼", "股票代碼": "代碼", "代碼名稱": "代碼"})
            for c in LIB_COLS:
                if c not in df.columns:
                    df[c] = ""
            frames.append(df[LIB_COLS])
        except Exception:
            pass
    rd = os.path.join(BASE_DIR, "reports")
    if os.path.isdir(rd):
        rows = []
        for f in sorted(os.listdir(rd)):
            m = re.match(r"(\d{4})[_\-](\d{4})", f)
            if f.lower().endswith(".pdf") and m:
                rows.append({"代碼": m.group(1), "年度": m.group(2), "PDF來源": f, "備註": "本機檔案"})
        if rows:
            frames.append(pd.DataFrame(rows, columns=LIB_COLS))
    if not frames:
        return pd.DataFrame(columns=LIB_COLS)
    df = pd.concat(frames[::-1], ignore_index=True)  # 本機檔案排前面，同代碼同年度時優先使用
    df["代碼"] = df["代碼"].map(_norm_code)
    df["年度"] = df["年度"].astype(str).str.extract(r"(\d{4})")[0]
    df = df[df["代碼"].isin(COMPANIES) & df["年度"].notna() & (df["PDF來源"].str.strip() != "")]
    return df.drop_duplicates(["代碼", "年度"], keep="first").sort_values(["年度", "代碼"], ascending=[False, True]).reset_index(drop=True)

def check_link(src):
    """只讀檔頭前幾個位元組，確認連結真的是 PDF（不會下載整份）。"""
    try:
        if not str(src).lower().startswith("http"):
            ok = os.path.exists(os.path.join(BASE_DIR, "reports", src))
            return "✅ 本機檔案" if ok else "❌ 找不到檔案"
        from urllib.parse import urlparse
        u = urlparse(normalize_url(src))
        r = http_get(normalize_url(src), timeout=20, stream=True,
                     headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                              "Referer": referer_for(normalize_url(src))})
        head = next(r.iter_content(16), b"")
        status = r.status_code
        r.close()
        if status >= 400:
            return f"❌ HTTP {status}"
        return "✅ PDF" if head[:4] == b"%PDF" else "⚠️ 不是 PDF（可能是網頁或需登入）"
    except Exception as e:
        return f"❌ {type(e).__name__}"

@st.cache_resource
def _text_cache():
    return {}  # 跨使用者共用的報告書全文快取（同一份報告書不會重複下載、解析）

def get_text(src, cache=None):
    """取得報告書全文。不呼叫任何 st 函式，所以可以安全地在背景執行緒中執行。"""
    cache = _text_cache() if cache is None else cache
    if src in cache:
        return cache[src]
    if str(src).lower().startswith("http"):
        text = load_report(normalize_url(src))
    else:
        with open(os.path.join(BASE_DIR, "reports", src), "rb") as f:
            text = extract_pdf(f.read())
        if len(text) < 1000:
            raise ValueError("PDF 幾乎抽不到文字（可能是掃描檔）")
    if len(cache) >= 40:
        cache.pop(next(iter(cache)))
    cache[src] = text
    return text

def load_texts_parallel(items, workers=3):
    """同時下載並解析多份報告書。items = [(代碼, 來源), ...]；回傳 {代碼: 全文 或 Exception}。"""
    cache = _text_cache()  # 在主執行緒取得，再交給背景執行緒使用

    def one(it):
        t, src = it
        try:
            return t, get_text(src, cache)
        except Exception as e:
            return t, e

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(one, items))

QDRANT_COLLECTION = secret("QDRANT_COLLECTION", "esg_report_chunks")

@st.cache_resource
def get_qdrant():
    from qdrant_client import QdrantClient
    return QdrantClient(url=secret("QDRANT_URL"), api_key=secret("QDRANT_API_KEY"), timeout=30)

@st.cache_data(show_spinner=False)
def load_query_vectors():
    """6 個構面的檢索問句向量（由 build_index.py 事先算好），網站不需要載入 embedding 模型。"""
    p = os.path.join(BASE_DIR, "query_vectors.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def _qdrant_filter(ticker, year):
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    return Filter(must=[FieldCondition(key="ticker", match=MatchValue(value=ticker)),
                        FieldCondition(key="year", match=MatchValue(value=str(year)))])

def vector_retrieve(ticker, year, dim, qvecs):
    """比照老師範例：query_points + score_threshold + 以 ticker / year 做 metadata 過濾。"""
    res = get_qdrant().query_points(
        collection_name=QDRANT_COLLECTION, query=qvecs["queries"][dim]["vector"],
        query_filter=_qdrant_filter(ticker, year), limit=vec_top_k, score_threshold=vec_threshold,
        with_payload=True)
    pts = res.points
    return [f"[p{p.payload.get('page', 0)}] {p.payload.get('text', '')}" for p in pts], (pts[0].score if pts else None)

def index_count(ticker, year):
    return get_qdrant().count(collection_name=QDRANT_COLLECTION, count_filter=_qdrant_filter(ticker, year),
                              exact=True).count

def run_scoring(tickers, mode="basic"):
    """mode='basic'：TF-IDF 檢索（需已載入報告書全文）；mode='vector'：Qdrant 向量檢索。成功回傳 True。"""
    if not token:
        st.error("尚未設定 LLM（CUSTOM_API_TOKEN / API_ENDPOINT / LLM_MODEL），無法進行 RAG 評分。請先到 Streamlit 的 Secrets 設定。")
        return False
    qvecs = None
    if mode == "vector":
        qvecs = load_query_vectors()
        if not qvecs:
            st.error("找不到 query_vectors.json。請先執行 build_index.py，並把產生的 query_vectors.json 上傳到 GitHub。")
            return False
        bad = [d for d, c in RUBRIC.items() if qvecs.get("queries", {}).get(d, {}).get("query") != c["query"]]
        if bad:
            st.error("query_vectors.json 的檢索問句和目前的 app.py 不一致（" + "、".join(bad)
                     + "）。請重新執行 build_index.py 更新。")
            return False
    year = st.session_state.get("lib_year", "")
    bar = st.progress(0.0, text="檢索與規則評分中…")

    # 階段 1：檢索 + 規則評分（很快）
    base, jobs = {}, {}
    tcache = st.session_state.setdefault("tfidf_cache", {})
    for i, t in enumerate(tickers):
        if mode == "basic":
            key = (t, chunk_size, overlap, len(st.session_state["docs"][t]))
            if key not in tcache:
                if len(tcache) > 30:
                    tcache.clear()
                chunks = chunk_text(st.session_state["docs"][t], chunk_size, overlap)
                tcache[key] = (chunks, *make_tfidf(chunks))
            chunks, vec, mat = tcache[key]
        for dim, cfg in RUBRIC.items():
            sim = None
            if mode == "vector":
                try:
                    top, sim = vector_retrieve(t, year, dim, qvecs)
                except Exception as e:
                    st.error(f"向量檢索失敗：{type(e).__name__}: {str(e)[:200]}")
                    return False
            else:
                top = retrieve_fast(chunks, vec, mat, cfg["query"], top_k)
            sc_, hit, best = rule_score(top, cfg["keywords"])
            reason = f"命中關鍵詞：{', '.join(hit) or '無'}"
            if mode == "vector" and not top:
                reason = f"沒有段落達到相似度門檻（{vec_threshold}）"
            base[(t, dim)] = (sc_, reason, best, sim)
            if top:  # 只要檢索到段落就交給 LLM 判斷，不受關鍵詞命中與否限制
                jobs[(t, dim)] = top
        bar.progress(0.5 * (i + 1) / len(tickers), text="檢索與規則評分中…")

    # 階段 2：LLM 複核（多個請求同時送出；失敗的項目最後再單獨補跑一次）
    llm_res, llm_err = {}, {}
    lcache = _llm_cache()
    cached = 0
    pending = {}
    for k, top in jobs.items():
        ck = _llm_key(k[0], k[1], top)
        if ck in lcache:  # 先前已成功評過的相同原文：直接沿用，不消耗額度
            llm_res[k], llm_err[k] = lcache[ck], ""
            cached += 1
        else:
            pending[k] = top

    def _run(batch, workers, label):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(llm_review, COMPANIES[t], dim, top, endpoint, token, model): (t, dim)
                    for (t, dim), top in batch.items()}
            for n, f in enumerate(as_completed(futs), 1):
                llm_res[futs[f]], llm_err[futs[f]] = f.result()
                bar.progress(0.5 + 0.5 * n / len(batch), text=f"{label} {n}/{len(batch)}")

    if pending:
        _run(pending, llm_workers, "LLM 複核")
        retry = {k: v for k, v in pending.items()
                 if llm_res.get(k) is None
                 and not re.match(r"HTTP (400|401|403|404)|HTTP 429：", llm_err.get(k, ""))}
        if retry:
            _run(retry, 1, "補跑失敗項目")
        for k, top in pending.items():
            if llm_res.get(k) is not None:
                lcache[_llm_key(k[0], k[1], top)] = llm_res[k]
    bar.progress(1.0, text="完成")

    # 階段 3：整理結果
    rows, evidence, llm_failed, fail_reasons = [], [], 0, {}
    for t in tickers:
        row = {"代碼": t, "公司": COMPANIES[t], "年度": year}
        for dim in RUBRIC:
            sc_, reason, best, sim = base[(t, dim)]
            r = llm_res.get((t, dim))
            if r:
                sc_, reason = r
            elif (t, dim) in jobs:  # 有送出但 LLM 沒有成功回應
                err = llm_err.get((t, dim), "") or "原因不明"
                reason = f"⚠️ LLM 未回應（{err}），暫用關鍵詞分數｜" + reason
                llm_failed += 1
                fail_reasons[err[:30]] = fail_reasons.get(err[:30], 0) + 1
            row[dim] = sc_
            evidence.append({"公司": COMPANIES[t], "構面": dim, "分數": sc_, "說明": reason,
                             "最高相似度": round(sim, 3) if sim is not None else None, "原文佐證": best[:300]})
        row["總分(平均)"] = round(float(np.mean([row[d] for d in RUBRIC])), 2)
        rows.append(row)
    if llm_failed:
        summary = "、".join(f"{k} ×{v}" for k, v in fail_reasons.items())
        if "額度已用完" in summary or "需等待" in summary:
            hint = ("這是 Groq 額度用完，不是程式錯誤：已成功的項目會保留、重新評分時不會再消耗額度。"
                    "請等額度恢復後按「🔄 重新評分」，或改用較小的模型、調低檢索片段數 (top-k) 以減少 token 用量。")
        elif re.search(r"HTTP 40[0134]：", summary):
            hint = "請檢查 Secrets 的 CUSTOM_API_TOKEN / API_ENDPOINT / LLM_MODEL 是否正確。"
        else:
            hint = "常見原因是額度限流：請把「LLM 同時呼叫數」調成 1，等一分鐘後按「🔄 重新評分」（已成功的項目不會重跑）。"
        st.warning(f"有 {llm_failed} 個構面的 LLM 沒有回應（已自動重試），暫用關鍵詞分數。失敗原因：{summary}。{hint}")
    if cached:
        st.caption(f"ℹ️ 其中 {cached} 個構面沿用先前成功的 LLM 結果（原文相同，未消耗額度）。")
    st.session_state["scores"] = pd.DataFrame(rows)
    st.session_state["evidence"] = pd.DataFrame(evidence)
    st.session_state["scored_mode"] = mode
    st.session_state["scored_tickers"] = list(tickers)
    return True


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

def ranked(df):
    """依總分由高到低排序，加上『排名』欄（從 1 開始，同分並列同名次）。"""
    d = df.sort_values("總分(平均)", ascending=False).reset_index(drop=True)
    d.insert(0, "排名", d["總分(平均)"].rank(method="min", ascending=False).astype(int))
    return d

SCORE_NOTE = "📌 各構面與總分的**滿分為 5 分**（最低 0 分）；總分＝六個構面分數的平均；排名依總分由高到低，同分者並列同名次。"

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
top_k = st.sidebar.slider("每構面檢索片段數 (top-k)", 3, 10, 5)
chunk_size = st.sidebar.slider("Chunk 字數", 200, 1000, 500)
overlap = st.sidebar.slider("Overlap 字數", 0, 200, 50)
min_sim = st.sidebar.slider("相似度門檻（0＝不限制）", 0.0, 0.3, 0.0, 0.01,
                           help="低於此相似度的段落不會被採用。TF-IDF 相似度通常落在 0.05–0.4，建議 0.05 以內；設太高會讓構面找不到證據而變 0 分。")

st.sidebar.subheader("🔎 檢索模式")
retrieval_mode = st.sidebar.radio("檢索模式", ["基礎（TF-IDF，預設）", "向量檢索（Qdrant，比照老師範例）"],
                                  label_visibility="collapsed")
vector_mode = retrieval_mode.startswith("向量")
vec_top_k, vec_threshold = 10, 0.65  # 老師範例的設定：top_k=10、score_threshold=0.65
if vector_mode:
    vec_top_k = st.sidebar.slider("向量檢索 top-k", 3, 20, 10)
    vec_threshold = st.sidebar.slider("相似度門檻 score_threshold", 0.0, 0.95, 0.65, 0.01,
                                      help="老師範例用 0.65。門檻太高會找不到段落、構面變 0 分；太低則會混入不相關內容。")
    st.sidebar.caption("向量模式只用於「📚 報告書資料庫」，需先用 build_index.py 建好索引。")
    if not (secret("QDRANT_URL") and secret("QDRANT_API_KEY")):
        st.sidebar.warning("尚未在 Secrets 設定 QDRANT_URL / QDRANT_API_KEY")

st.sidebar.subheader("🧠 LLM 複核")
st.sidebar.caption("每個構面都會把檢索到的原文交給 LLM 判斷評分（完整 RAG 流程）。")
llm_workers = st.sidebar.slider("LLM 同時呼叫數", 1, 6, 2,
                                help="越大越快，但免費額度容易被限流（429）；該項目沒有回應時，會暫用關鍵詞分數並在說明欄標示。")
# LLM 連線設定一律從 Secrets 讀取，不在網頁上顯示或讓訪客輸入
endpoint = secret("API_ENDPOINT", "https://api.openai.com/v1/chat/completions")
model = secret("LLM_MODEL", "gpt-4o-mini")
_SECRET_TOKEN = secret("CUSTOM_API_TOKEN")
_ACCESS_CODE = secret("LLM_ACCESS_CODE")  # 選用：設了之後，要輸入密碼才能使用你的 key（有設定才會出現輸入框）
_allowed = bool(_SECRET_TOKEN)
if _SECRET_TOKEN and _ACCESS_CODE:
    _allowed = st.sidebar.text_input("LLM 功能使用密碼", "", type="password") == _ACCESS_CODE
token = _SECRET_TOKEN if _allowed else ""
if not token:
    st.sidebar.error("LLM 尚未設定，無法評分：請在 Secrets 設定 CUSTOM_API_TOKEN、API_ENDPOINT、LLM_MODEL"
                     "（若有設使用密碼，請輸入密碼）。")
_NET["skip_ssl"] = False

st.session_state.setdefault("docs", {})

st.title("🌱 臺灣金融業永續報告書：檢索、評分與比較")
t1, t2, t3, t4 = st.tabs(["📂 1 取得報告書", "🤖 2 RAG 評分", "📊 3 差異比較", "📈 4 因子回測"])

# ---------------- Tab1 ----------------
with t1:
    st.subheader("📚 報告書資料庫（優先使用）")
    lib = load_library()
    if lib.empty:
        st.info("資料庫目前是空的。請用下方「即時搜尋」找到報告書後加入清單，再下載 reports_index.csv 上傳到 GitHub；"
                "或直接把 PDF 放進 GitHub 的 reports 資料夾（檔名如 2884_2023.pdf）。")
    else:
        ly = st.selectbox("年度", sorted(lib["年度"].unique(), reverse=True), key="lib_year_sel")
        avail = lib[lib["年度"] == ly]
        names = {r.代碼: f"{r.代碼} {COMPANIES.get(r.代碼, '')}" for r in avail.itertuples()}
        picks = st.multiselect("選擇公司（同一年度才能公平比較）", list(names), default=list(names),
                               format_func=lambda x: names[x])
        if st.button("⚡ 從資料庫載入並評分", type="primary") and picks:
            if vector_mode:
                ready_ = []
                try:
                    for t in picks:
                        if index_count(t, ly) > 0:
                            ready_.append(t)
                        else:
                            st.warning(f"{COMPANIES[t]} {ly} 年尚未建立向量索引，已略過（請先執行 build_index.py）")
                except Exception as e:
                    st.error(f"連線 Qdrant 失敗：{type(e).__name__}: {str(e)[:200]}")
                    ready_ = []
                if ready_:
                    st.session_state["lib_year"] = ly
                    if run_scoring(ready_, mode="vector"):
                        st.success("完成（向量檢索）！請到「步驟 2」看明細、「步驟 3」看公司比較。")
            else:
                st.session_state["docs"] = {}
                items = [(r.代碼, r.PDF來源) for r in avail[avail["代碼"].isin(picks)].itertuples()]
                with st.spinner(f"同時下載並解析 {len(items)} 份 {ly} 年報告書…（第一次較久，之後會用快取）"):
                    loaded = load_texts_parallel(items, workers=3)
                for t, res in loaded.items():
                    if isinstance(res, Exception):
                        st.error(f"{COMPANIES[t]} 失敗：{res}")
                    else:
                        st.session_state["docs"][t] = res
                if st.session_state["docs"]:
                    st.session_state["lib_year"] = ly
                    run_scoring(list(st.session_state["docs"]))
                    st.success("完成！請到「步驟 2」看明細、「步驟 3」看公司比較。")
        if vector_mode and st.button("📊 查看這個年度的向量索引狀態"):
            try:
                st.dataframe(pd.DataFrame([{"公司": COMPANIES[r.代碼], "年度": ly, "已索引片段數": index_count(r.代碼, ly)}
                                           for r in avail.itertuples()]), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"連線 Qdrant 失敗：{type(e).__name__}: {str(e)[:200]}")
        if st.button("🔍 檢查這個年度的連結能不能下載"):
            res, bar = [], st.progress(0.0)
            rows_ = list(avail.itertuples())
            for i, r in enumerate(rows_):
                res.append({"公司": COMPANIES[r.代碼], "年度": ly, "結果": check_link(r.PDF來源)})
                bar.progress((i + 1) / len(rows_))
            st.dataframe(pd.DataFrame(res), use_container_width=True, hide_index=True)
        with st.expander(f"資料庫清單（共 {len(lib)} 筆）"):
            st.dataframe(lib, use_container_width=True, hide_index=True)
    st.divider()
    st.subheader("🔎 找不到？即時搜尋官網")
    c1, c2, c3 = st.columns([3, 1, 1])
    kw = c1.text_input("輸入關鍵字（公司名稱或代碼，如：合庫金、5880）；或直接貼上企業官網的報告書下載頁網址")
    year = c2.selectbox("年度", [str(y) for y in range(datetime.now().year - 1, 2020, -1)])
    c3.write("")
    if c3.button("🔎 搜尋", type="primary") and kw.strip():
        try:
            with st.spinner("搜尋永續報告書中…"):
                res = search_reports(kw.strip(), year)
            df = pd.DataFrame(res)
            if st.session_state.get("crawl_err"):
                st.warning("讀取官網報告書頁面失敗，已改用搜尋結果（可能不準確，請特別確認來源與檔名）："
                           + st.session_state["crawl_err"])
            if df.empty:
                st.warning("沒有找到結果。")
                if st.session_state.get("diag"):
                    with st.expander("🩺 診斷資訊（回報問題時請截圖這一塊）", expanded=True):
                        st.json(st.session_state["diag"])
                        d = st.session_state["diag"]
                        if d["頁面連結總數"] < 5:
                            st.info("頁面幾乎沒有連結，可能是網站用 JavaScript 動態載入內容，或是擋掉了雲端主機的存取。")
                st.session_state.pop("results", None)
            else:
                df["歸屬公司"] = [r.company or guess_company(f"{r.title} {r.url}") for r in df.itertuples()]
                df = df.drop(columns=["company"])
                df.insert(0, "選取", False)
                df.loc[0, "選取"] = bool(df.loc[0, "來源"] == "官網" and year in (df.loc[0, "title"] + df.loc[0, "url"]))
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
                st.session_state["lib_year"] = year
                run_scoring(list(st.session_state["docs"]))
                st.success("評分完成！請到「步驟 2」看明細、「步驟 3」看公司比較。")
                if _NET["relaxed"]:
                    st.info("提醒：以下網站的憑證格式不符規範（缺少 Subject Key Identifier），"
                            "已改用免憑證驗證下載公開報告書：" + "、".join(sorted(_NET["relaxed"])))

    if "results" in st.session_state:
        if st.button("➕ 把勾選的報告書加入資料庫清單"):
            new_rows = [{"代碼": r.歸屬公司[:7], "年度": year, "PDF來源": r.url, "備註": r.title[:50]}
                        for r in edited[edited["選取"]].itertuples() if r.歸屬公司]
            st.session_state.setdefault("lib_draft", []).extend(new_rows)
            st.success(f"已加入 {len(new_rows)} 筆，請往下查看並下載清單。")
    if st.session_state.get("lib_draft"):
        merged = pd.concat([pd.DataFrame(st.session_state["lib_draft"]), load_library()], ignore_index=True)
        merged = merged.drop_duplicates(["代碼", "年度"], keep="first").sort_values(["年度", "代碼"], ascending=[False, True])
        st.markdown("**📝 待更新的資料庫清單**（下載後上傳到 GitHub，覆蓋原本的 reports_index.csv）")
        st.dataframe(merged, use_container_width=True, hide_index=True)
        st.download_button("⬇️ 下載 reports_index.csv", merged.to_csv(index=False).encode("utf-8-sig"), "reports_index.csv")
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
    st.caption("RAG 評分流程：檢索相關段落 → LLM 只依檢索到的原文判斷並給分（0–5 分）；找不到證據即為 0 分。")
    if "scores" not in st.session_state and not st.session_state["docs"]:
        st.info("請先在步驟 1 搜尋並載入報告書。")
    else:
        _mode = st.session_state.get("scored_mode", "basic")
        st.caption("目前評分使用的檢索模式：" + ("向量檢索（Qdrant）" if _mode == "vector" else "基礎（TF-IDF）"))
        if st.button("🔄 重新評分（調整參數後使用）"):
            if _mode == "vector":
                run_scoring(st.session_state.get("scored_tickers", []), mode="vector")
            else:
                run_scoring(list(st.session_state["docs"]))
        if "scores" in st.session_state:
            st.info(SCORE_NOTE)
            st.dataframe(ranked(st.session_state["scores"]), use_container_width=True, hide_index=True)
            with st.expander("🔍 原文佐證（分數欄滿分為 5 分）"):
                st.dataframe(st.session_state["evidence"], use_container_width=True, hide_index=True)

# ---------------- Tab3 ----------------
with t3:
    if "scores" not in st.session_state:
        st.info("請先完成步驟 2。")
    else:
        sc = ranked(st.session_state["scores"])
        st.info(SCORE_NOTE)
        st.markdown("#### 🏆 總分排名")
        st.dataframe(sc[["排名", "公司", "總分(平均)"]].rename(columns={"總分(平均)": "總分（滿分 5 分）"}),
                     use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(sc, x="公司", y="總分(平均)", color="總分(平均)",
                                   color_continuous_scale="Greens", range_color=[0, 5], range_y=[0, 5],
                                   title="總分比較（滿分 5 分）",
                                   labels={"總分(平均)": "總分（滿分 5 分）"}), use_container_width=True)
        with c2:
            fig = go.Figure()
            for _, r in sc.iterrows():
                fig.add_trace(go.Scatterpolar(r=[r[d] for d in RUBRIC], theta=list(RUBRIC), fill="toself", name=r["公司"]))
            fig.update_layout(title="各構面雷達圖（外圈為滿分 5 分）", polar=dict(radialaxis=dict(range=[0, 5])))
            st.plotly_chart(fig, use_container_width=True)
        heat = sc.set_index("公司")[list(RUBRIC)]
        hfig = px.imshow(heat, text_auto=True, color_continuous_scale="Greens", zmin=0, zmax=5, aspect="auto",
                         title="各構面分數熱力圖（滿分 5 分）")
        st.plotly_chart(hfig, use_container_width=True)
        st.download_button("下載評分 CSV", sc.to_csv(index=False).encode("utf-8-sig"), "esg_scores.csv")

# ---------------- Tab4 ----------------
with t4:
    if "scores" not in st.session_state:
        st.info("請先完成步驟 2。")
    else:
        d1, d2 = st.columns(2)
        start = d1.date_input("起始日", datetime(2022, 1, 1))
        end = d2.date_input("結束日", datetime.now())
        st.warning("注意：回測使用的是同一組評分，並套用到整段期間；若報告書公布日期晚於起始日，就會有前視偏誤"
                   "(look-ahead bias)。持股只有已評分的公司，樣本很少，結果僅供概念展示，不可視為可實際獲利的策略。")
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
                wt = pd.DataFrame({"公司": [COMPANIES.get(t, t) for t in w_c.index],
                                   "總分(滿分5分)": sc.reindex(w_c.index).values,
                                   "評分加權的權重 %": (w_c * 100).round(1).values,
                                   "等權重的權重 %": (w_e * 100).round(1).values})
                st.markdown("**持股權重**（兩組權重越接近，兩條績效線就越重疊）")
                st.dataframe(wt, use_container_width=True, hide_index=True)
                st.markdown("**績效指標**")
                st.dataframe(pd.DataFrame({"評分加權": mc, "等權重": me}, index=names).round(2), use_container_width=True)
                st.plotly_chart(px.line(cum, title="累積淨值（起始 = 1）", labels={"value": "累積淨值", "variable": ""}),
                                use_container_width=True)
                dd = (cum - cum.cummax()) / cum.cummax() * 100
                fig_dd = go.Figure()  # 不用 px.area：它預設把兩組數列「疊加」，會讓回撤看起來變成兩倍
                for col, color in zip(dd.columns, ["#1f77b4", "#ff7f0e"]):
                    fig_dd.add_trace(go.Scatter(x=dd.index, y=dd[col], name=col, mode="lines", fill="tozeroy",
                                                line=dict(color=color, width=1.5), opacity=0.7))
                fig_dd.update_layout(title="水下回撤圖（與前高相比的跌幅 %）", yaxis_title="回撤 %", xaxis_title="日期",
                                     legend_title_text="")
                st.plotly_chart(fig_dd, use_container_width=True)
            except Exception as e:
                st.error(f"回測失敗：{e}")

st.divider()
st.caption("評分為文本揭露程度之輔助指標，非投資建議。")

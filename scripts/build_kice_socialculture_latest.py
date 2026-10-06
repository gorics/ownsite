import os, sys, time, re
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import fitz
import gdown

OUT = Path("KICE_socialculture_2014-2027_latest_first.pdf")
WORK = Path("_social_build")
WORK.mkdir(exist_ok=True)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36"
S = requests.Session()
S.headers.update({"User-Agent": UA, "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8"})

# Latest-first order. 2027 has no CSAT yet as of 2026-10-07.
current = [
    ("2027학년도 9월 모의평가", "https://legendstudy.com/1710"),
    ("2027학년도 6월 모의평가", "https://legendstudy.com/1706"),
]

# For 2021-2026, use the user's existing KICE social-culture archive;
# its problem section is already ordered 2026 CSAT -> ... -> 2021 June.
base_titles = []
for y in range(2026, 2020, -1):
    base_titles += [f"{y}학년도 수능", f"{y}학년도 9월 모의평가", f"{y}학년도 6월 모의평가"]

older = [
    ("2020학년도 수능", "https://legendstudy.com/1458"),
    ("2020학년도 9월 모의평가", "https://legendstudy.com/1455"),
    ("2020학년도 6월 모의평가", "https://legendstudy.com/1454"),
    ("2019학년도 수능", "https://legendstudy.com/1379"),
    ("2019학년도 9월 모의평가", "https://legendstudy.com/1353"),
    ("2019학년도 6월 모의평가", "https://legendstudy.com/1290"),
    ("2018학년도 수능", "https://legendstudy.com/1236"),
    ("2018학년도 9월 모의평가", "https://legendstudy.com/1217"),
    ("2018학년도 6월 모의평가", "https://legendstudy.com/1128"),
    ("2017학년도 수능", "https://legendstudy.com/1064"),
    ("2017학년도 9월 모의평가", "https://legendstudy.com/1038"),
    ("2017학년도 6월 모의평가", "https://legendstudy.com/971"),
    ("2016학년도 수능", "https://legendstudy.com/832"),
    ("2016학년도 9월 모의평가", "https://legendstudy.com/806"),
    ("2016학년도 6월 모의평가", "https://legendstudy.com/748"),
    ("2015학년도 수능", "https://legendstudy.com/689"),
    ("2015학년도 9월 모의평가", "https://legendstudy.com/687"),
    ("2015학년도 6월 모의평가", "https://legendstudy.com/602"),
    ("2014학년도 수능", "https://legendstudy.com/275"),
    ("2014학년도 9월 모의평가", "https://legendstudy.com/281"),
    ("2014학년도 6월 모의평가", "https://legendstudy.com/280"),
]

def get_article(url):
    for attempt in range(5):
        r = S.get(url, timeout=40)
        if r.ok and len(r.content) > 1000:
            r.encoding = r.apparent_encoding or "utf-8"
            return r.text
        time.sleep(2 + attempt)
    raise RuntimeError(f"Article fetch failed: {url} status={getattr(r,'status_code',None)}")

def find_society_pdf(article_url):
    html = get_article(article_url)
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    for a in soup.find_all("a", href=True):
        text = " ".join(a.stripped_strings)
        norm = re.sub(r"\s+", "", text)
        if ("사회문화" in norm and "문제" in norm and "정답" not in norm):
            href = a["href"]
            if href.startswith("//"):
                href = "https:" + href
            elif href.startswith("/"):
                href = requests.compat.urljoin(article_url, href)
            candidates.append((text, href))
    if not candidates:
        raise RuntimeError(f"No 사회문화 문제 PDF link found in {article_url}")
    # Prefer a PDF/CDN attachment, not navigation links.
    for text, href in candidates:
        if ("daumcdn" in href or "kakaocdn" in href or ".pdf" in href.lower() or "attachment" in href.lower()):
            return href, text
    return candidates[0][1], candidates[0][0]

def download_pdf(url, dest, referer):
    headers = {"Referer": referer, "User-Agent": UA}
    last = None
    for attempt in range(6):
        try:
            with S.get(url, headers=headers, timeout=90, stream=True, allow_redirects=True) as r:
                last = r.status_code
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1024*1024):
                        if chunk: f.write(chunk)
            # Validate
            d = fitz.open(dest)
            n = d.page_count
            d.close()
            if n < 1:
                raise RuntimeError("empty PDF")
            return n
        except Exception as e:
            if Path(dest).exists():
                Path(dest).unlink(missing_ok=True)
            if attempt == 5:
                raise RuntimeError(f"PDF download failed {url}: {e}; status={last}")
            time.sleep(2 + attempt * 2)

def fetch_exam(title, article, idx):
    href, label = find_society_pdf(article)
    dest = WORK / f"src_{idx:02d}_{title.replace(' ','_')}.pdf"
    pages = download_pdf(href, dest, article)
    print(f"OK {title}: {pages}p <- {href}")
    return dest

# 1) Download 2027.
current_files = []
for i,(title,article) in enumerate(current):
    current_files.append((title, fetch_exam(title, article, i)))

# 2) Download user Drive archive (2021-2026).
base = WORK / "9_사문.pdf"
base_id = "1PTpYJ_h56gNcIejtdtDYpX77nsKTuwWA"
print("Downloading base Drive archive...")
result = gdown.download(id=base_id, output=str(base), quiet=False, fuzzy=True)
if not result or not base.exists():
    raise RuntimeError("Could not download base 9 사문.pdf from Google Drive")
bd = fitz.open(base)
print("Base pages:", bd.page_count)
if bd.page_count < 74:
    raise RuntimeError(f"Base archive too short: {bd.page_count}")
bd.close()

# 3) Download 2014-2020.
older_files = []
for j,(title,article) in enumerate(older, start=len(current_files)):
    older_files.append((title, fetch_exam(title, article, j)))

# 4) Assemble.
out = fitz.open()
toc = []

def append_doc(title, path, from_page=None, to_page=None):
    src = fitz.open(path)
    start = out.page_count
    if from_page is None:
        out.insert_pdf(src)
        added = src.page_count
    else:
        out.insert_pdf(src, from_page=from_page, to_page=to_page)
        added = to_page - from_page + 1
    src.close()
    toc.append([1, title, start + 1])
    print("ADD", title, "start", start+1, "pages", added)

for title,path in current_files:
    append_doc(title,path)

# Base problem section: PDF pages 3-74 (0-based 2-73), 18 exams x 4 pages.
src = fitz.open(base)
for i,title in enumerate(base_titles):
    p0 = 2 + i*4
    start = out.page_count
    out.insert_pdf(src, from_page=p0, to_page=p0+3)
    toc.append([1,title,start+1])
    print("ADD", title, "base pages", p0+1, "-", p0+4)
src.close()

for title,path in older_files:
    append_doc(title,path)

out.set_toc(toc)
out.set_metadata({
    "title":"KICE 사회·문화 2014-2027 최신순 기출문제",
    "subject":"한국교육과정평가원 6월·9월 모의평가 및 수능 사회·문화 문제지 최신순 합본",
    "author":"KICE source compilation",
    "keywords":"사회문화, 수능, 평가원, 6월 모의평가, 9월 모의평가, 기출"
})
out.save(OUT, garbage=3, deflate=True)
out.close()

final = fitz.open(OUT)
print("FINAL_PAGES", final.page_count)
print("TOC_COUNT", len(final.get_toc()))
# Every included 사회문화 paper is expected to be 4 pages: 41 exams => 164 pages.
if final.page_count != 164:
    print("WARNING expected 164 pages but got", final.page_count)
if len(final.get_toc()) != 41:
    raise RuntimeError(f"Expected 41 bookmarks, got {len(final.get_toc())}")
final.close()
print("OUTPUT", OUT.resolve(), OUT.stat().st_size)

"""
    Changes: Remove task_done and only put one in the finally block
"""


import os
os.environ.pop("OPENSSL_CONF", None)
os.environ.pop("OPENSSL_MODULES", None)

import argparse
import asyncio
import json
import logging
import random
import re
import sqlite3
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from urllib.parse import urljoin, urlparse, quote, urlunparse

import aiohttp
import requests
import urllib3
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import fitz
except ImportError:
    raise ImportError("Please install PyMuPDF: pip install PyMuPDF")

@dataclass
class CrawlConfig:
    base_url: str = "https://congbobanan.toaan.gov.vn"
    search_path: str = "/0t15at1cvn/Tra-cu-ban-an"
    out_dir: Path = Path("./output_banan_async")
    pdf_dir: Path = Path("./output_banan_async/pdfs")
    keyword: str = ""
    court_level: str = "T"
    max_pages: int = 0
    max_items: int = 0
    sleep_min: float = 0.5
    sleep_max: float = 1.0
    retries: int = 5
    insecure: bool = True
    concurrent_requests: int = 5
    exclude_courts: Set[str] = None
    resume: bool = False

class CrawlerDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS judgments (
                    url TEXT PRIMARY KEY,
                    case_id TEXT,
                    pdf_url TEXT,
                    status TEXT NOT NULL,
                    text_length INTEGER DEFAULT 0,
                    pdf_filename TEXT,
                    court_value TEXT,
                    court_name TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_status ON judgments(status);
                CREATE INDEX IF NOT EXISTS idx_url ON judgments(url);

                CREATE TABLE IF NOT EXISTS crawl_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)

    def is_url_processed(self, url: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT 1 FROM judgments WHERE url = ? AND status IN ('downloaded', 'skipped_scanned')", (url,))
            return cur.fetchone() is not None

    def add_pending(self, url: str, case_id: str = "", court_value: str = "", court_name: str = ""):
        with sqlite3.connect(self.db_path) as conn:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute("""
                INSERT OR IGNORE INTO judgments (url, case_id, court_value, court_name, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'pending', ?, ?)
            """, (url, case_id, court_value, court_name, now, now))

    def update_status(self, url: str, status: str, text_length: int = 0, pdf_url: str = "", pdf_filename: str = "", case_id: str = ""):
        with sqlite3.connect(self.db_path) as conn:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute("""
                UPDATE judgments
                SET status = ?, text_length = ?, pdf_url = ?, pdf_filename = ?, case_id = ?, updated_at = ?
                WHERE url = ?
            """, (status, text_length, pdf_url, pdf_filename, case_id, now, url))

    def get_state(self, key: str, default: str = "") -> str:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT value FROM crawl_state WHERE key = ?", (key,))
            row = cur.fetchone()
            return row[0] if row else default

    def set_state(self, key: str, value: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("REPLACE INTO crawl_state (key, value) VALUES (?, ?)", (key, value))

    def get_pending_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM judgments WHERE status = 'pending'")
            return cur.fetchone()[0]

class AsyncCrawler:
    def __init__(self, cfg: CrawlConfig):
        self.cfg = cfg
        self.search_url = urljoin(cfg.base_url.rstrip("/") + "/", cfg.search_path.lstrip("/"))
        self.out_dir = cfg.out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.db = CrawlerDB(self.out_dir / "crawler.db")
        self.async_queue = asyncio.Queue()
        self.sync_queue = queue.Queue()
        self.producer_done = threading.Event()
        self.total_queued = 0
        self.total_processed = 0
        self._setup_logging()
        if self.cfg.insecure:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.sync_session = self._make_sync_session()

    def _setup_logging(self):
        log_file = self.out_dir / "crawler.log"
        logger = logging.getLogger("Crawler")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        self.logger = logger

    def _make_sync_session(self):
        s = requests.Session()
        retry = Retry(total=self.cfg.retries, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        return s

    def _sleep(self):
        time.sleep(random.uniform(self.cfg.sleep_min, self.cfg.sleep_max))

    def _extract_form_fields(self, soup: BeautifulSoup) -> Tuple[Dict[str, str], str]:
        form = soup.find("form", id="aspnetForm") or soup.find("form")
        fields = {}
        for inp in form.find_all("input"):
            name = inp.get("name")
            if not name:
                continue
            typ = (inp.get("type") or "text").lower()
            if typ in ("submit", "button", "file", "image"):
                continue
            if typ in ("checkbox", "radio"):
                if inp.has_attr("checked"):
                    fields[name] = inp.get("value", "on")
                continue
            fields[name] = inp.get("value", "")
        for sel in form.find_all("select"):
            name = sel.get("name")
            if not name:
                continue
            opt = sel.find("option", selected=True) or sel.find("option")
            fields[name] = opt.get("value", "") if opt else ""
        fields.setdefault("__EVENTTARGET", "")
        fields.setdefault("__EVENTARGUMENT", "")
        action = urljoin(self.search_url, form.get("action") or self.search_url)
        return fields, action

    def _get_court_options(self) -> List[Tuple[str, str]]:
        self.logger.info("Fetching court list for level T")
        r = self.sync_session.get(self.search_url, verify=not self.cfg.insecure)
        soup = BeautifulSoup(r.text, "lxml")
        fields, action = self._extract_form_fields(soup)
        fields["ctl00$Content_home_Public$ctl00$Drop_Levels_top"] = "T"
        fields["__EVENTTARGET"] = "ctl00$Content_home_Public$ctl00$Drop_Levels_top"
        r = self.sync_session.post(action, data=fields, verify=not self.cfg.insecure)
        soup = BeautifulSoup(r.text, "lxml")
        court_select = soup.find("select", id="ctl00_Content_home_Public_ctl00_Ra_Drop_Courts_top")
        if not court_select:
            self.logger.error("Could not find court dropdown")
            return []
        options = []
        for opt in court_select.find_all("option"):
            val = opt.get("value", "").strip()
            text = opt.get_text(strip=True)
            if val and text and val not in self.cfg.exclude_courts:
                options.append((val, text))
        self.logger.info(f"Found {len(options)} courts to process")
        return options

    def _extract_links_from_soup(self, soup: BeautifulSoup) -> List[str]:
        out = set()
        zone = soup.find(id="List_group_pub") or soup
        for a in zone.select("a[href]"):
            href = (a.get("href") or "").strip()
            if href.startswith("javascript:") or "tra-cu-ban-an" in href.lower():
                continue
            u = urljoin(self.cfg.base_url, href)
            out.add(u)
        return sorted(out)

    def _producer_sync(self):
        self.logger.info("Producer thread started")
        court_options = self._get_court_options()
        if not court_options:
            self.producer_done.set()
            return

        start_court_val = self.db.get_state("last_court_val", "")
        start_page = int(self.db.get_state("last_page", "1")) if self.cfg.resume else 1

        for court_val, court_name in court_options:
            if self.cfg.resume and start_court_val and court_val != start_court_val:
                continue
            resume_page = start_page if court_val == start_court_val else 1
            self.logger.info(f"Processing court: {court_name}")
            r = self.sync_session.get(self.search_url, verify=not self.cfg.insecure)
            soup = BeautifulSoup(r.text, "lxml")
            fields, action = self._extract_form_fields(soup)
            fields["ctl00$Content_home_Public$ctl00$Drop_Levels_top"] = "T"
            fields["ctl00$Content_home_Public$ctl00$Ra_Drop_Courts_top"] = court_val
            fields["ctl00$Content_home_Public$ctl00$Drop_STATUS_JUDGMENT_SEARCH_top"] = "0"
            fields["ctl00$Content_home_Public$ctl00$txtKeyword_top"] = self.cfg.keyword
            fields["ctl00$Content_home_Public$ctl00$cmd_search_banner"] = "Tìm kiếm"
            r = self.sync_session.post(action, data=fields, verify=not self.cfg.insecure)
            soup = BeautifulSoup(r.text, "lxml")
            page_no = 1
            while True:
                if self.cfg.max_pages > 0 and page_no > self.cfg.max_pages:
                    break
                if resume_page and page_no < resume_page:
                    page_no += 1
                    continue
                links = self._extract_links_from_soup(soup)
                new_links = 0
                for url in links:
                    if not self.db.is_url_processed(url):
                        self.db.add_pending(url, court_value=court_val, court_name=court_name)
                        self.sync_queue.put(url)
                        new_links += 1
                        self.total_queued += 1
                self.logger.info(f"  Page {page_no}: {new_links} new URLs queued (total queued {self.total_queued})")
                self.db.set_state("last_court_val", court_val)
                self.db.set_state("last_page", str(page_no))
                if self.cfg.max_items > 0 and self.total_queued >= self.cfg.max_items:
                    break
                next_btn = soup.find("input", {"name": "ctl00$Content_home_Public$ctl00$cmdnext"})
                if not next_btn:
                    break
                fields, action = self._extract_form_fields(soup)
                fields["ctl00$Content_home_Public$ctl00$cmdnext"] = ">>"
                r = self.sync_session.post(action, data=fields, verify=not self.cfg.insecure)
                soup = BeautifulSoup(r.text, "lxml")
                page_no += 1
                self._sleep()
            if self.cfg.max_items > 0 and self.total_queued >= self.cfg.max_items:
                break
        self.producer_done.set()
        self.logger.info(f"Producer finished. Total URLs queued: {self.total_queued}")

    async def _bridge(self):
        loop = asyncio.get_running_loop()
        while not self.producer_done.is_set() or not self.sync_queue.empty():
            try:
                url = await loop.run_in_executor(None, self.sync_queue.get, True, 0.5)
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue
            await self.async_queue.put(url)
            self.sync_queue.task_done()

    async def download_worker(self, worker_id: int, session: aiohttp.ClientSession, sem: asyncio.Semaphore):
        self.logger.debug(f"Worker {worker_id} started")
        while not self.producer_done.is_set() or not self.async_queue.empty():
            try:
                url = await asyncio.wait_for(self.async_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            async with sem:
                try:
                    parsed_url = urlparse(url)
                    url_parts = [x for x in parsed_url.path.split("/") if x]
                    case_id = url_parts[0] if url_parts else "unknown_id"
                    async with session.get(url) as r:
                        html_text = await r.text()
                    s = BeautifulSoup(html_text, "lxml")
                    pdf_url = None
                    for a in s.select("a[href]"):
                        href = (a.get("href") or "").strip()
                        if href.lower().endswith(".pdf") or "download" in a.attrs:
                            # pdf_url = urljoin(url, href)
                            parsed = urlparse(href)
                            encoded_path = quote(parsed.path, safe='/')
                            encoded_href = urlunparse((
                                parsed.scheme, parsed.netloc, encoded_path, 
                                parsed.params, parsed.query, parsed.fragment
                            ))
                            pdf_url = urljoin(url, encoded_href)
                            break
                    if not pdf_url:
                        self.db.update_status(url, "failed", text_length=0)
                        self.logger.debug(f"No PDF link for {url}")
                        # self.async_queue.task_done()
                        continue
                    async with session.get(pdf_url) as r2:
                        if r2.status != 200:
                            self.db.update_status(url, "failed", text_length=0, pdf_url=pdf_url)
                            self.logger.warning(f"Failed PDF download {pdf_url} (status {r2.status})")
                            # self.async_queue.task_done()
                            continue
                        pdf_bytes = await r2.read()
                    def extract_text_sync(data):
                        doc = fitz.open(stream=data, filetype="pdf")
                        return "".join(page.get_text() for page in doc)
                    extracted_text = await asyncio.to_thread(extract_text_sync, pdf_bytes)
                    text_len = len(extracted_text.strip())
                    if text_len < 200:
                        self.db.update_status(url, "skipped_scanned", text_length=text_len, pdf_url=pdf_url)
                        self.logger.debug(f"Skipped {case_id}: scanned PDF (text length {text_len})")
                        # self.async_queue.task_done()
                        continue
                    pdf_filename = self.cfg.pdf_dir / f"{case_id}.pdf"
                    def save_pdf_sync():
                        pdf_filename.write_bytes(pdf_bytes)
                    await asyncio.to_thread(save_pdf_sync)
                    self.db.update_status(url, "downloaded", text_length=text_len, pdf_url=pdf_url, pdf_filename=f"{case_id}.pdf", case_id=case_id)
                    self.logger.info(f"Downloaded text PDF: {case_id}.pdf")
                    self.total_processed += 1
                except Exception as e:
                    self.db.update_status(url, "failed", text_length=0)
                    self.logger.error(f"Worker {worker_id} error on {url}: {e}")
                finally:
                    self.async_queue.task_done()
        self.logger.debug(f"Worker {worker_id} finished")

    async def run(self):
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=1) as executor:
            loop = asyncio.get_running_loop()
            producer_future = loop.run_in_executor(executor, self._producer_sync)
            bridge_task = asyncio.create_task(self._bridge())
            sem = asyncio.Semaphore(self.cfg.concurrent_requests)
            connector = aiohttp.TCPConnector(ssl=False) if self.cfg.insecure else None
            async with aiohttp.ClientSession(connector=connector) as session:
                workers = [asyncio.create_task(self.download_worker(i, session, sem)) for i in range(self.cfg.concurrent_requests)]
                await producer_future
                await bridge_task
                await self.async_queue.join()
                for w in workers:
                    w.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
        elapsed = time.time() - start_time
        self.logger.info(f"Crawling finished. Successfully downloaded: {self.total_processed} text PDFs. Time: {elapsed:.2f}s")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, default="")
    parser.add_argument("--court-level", type=str, default="T")
    parser.add_argument("--max-pages", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--max-items", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--concurrent-requests", type=int, default=5)
    parser.add_argument("--sleep-min", type=float, default=0.5)
    parser.add_argument("--sleep-max", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--insecure", action="store_true", default=True)
    parser.add_argument("--out-dir", type=Path, default=Path("./output_banan_async"))
    parser.add_argument("--resume", action="store_true", help="Resume from last court/page")
    default_exclude = {"3140","2449","2450","2451","2452","2453","2455","2457","2459","2458", "741"}
    parser.add_argument("--exclude-courts", type=str, default=",".join(default_exclude),
                        help="Comma-separated court values to exclude")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out_dir = args.out_dir
    exclude_set = set(args.exclude_courts.split(",")) if args.exclude_courts else set()
    cfg = CrawlConfig(
        keyword=args.keyword,
        court_level=args.court_level,
        max_pages=args.max_pages,
        max_items=args.max_items,
        concurrent_requests=args.concurrent_requests,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
        retries=args.retries,
        insecure=args.insecure,
        out_dir=out_dir,
        pdf_dir=out_dir / "pdfs",
        exclude_courts=exclude_set,
        resume=args.resume,
    )
    crawler = AsyncCrawler(cfg)
    asyncio.run(crawler.run())

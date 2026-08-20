from __future__ import annotations

import argparse
import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


BASE = "https://pakistancode.gov.pk"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
}
SKIP_PAGES = {
    "index.php", "LGu0xAD.php", "LGu0xBD.php", "LGu0xVD.php", "Xk3gLG.php",
    "UHyrdu.php", "sHyuRiF.php", "Rki82H.php", "Xkl72G.php", "Xki72H.php",
    "Xki72HF.php",
}


def fetch(session: requests.Session, url: str, timeout: int = 45) -> requests.Response | None:
    for attempt in range(3):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            if attempt == 2:
                print(f"[error] {url}: {exc}")
            else:
                time.sleep(2 * (attempt + 1))
    return None


def law_links(html: str, page_url: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    links: dict[str, str] = {}
    for anchor in soup.find_all("a", href=True):
        full = urljoin(page_url, anchor["href"].strip())
        parsed = urlparse(full)
        name = parsed.path.rstrip("/").split("/")[-1]
        title = anchor.get_text(" ", strip=True)
        if parsed.netloc == "pakistancode.gov.pk" and parsed.path.startswith("/english/") and name not in SKIP_PAGES and title:
            links[full] = title
    return links


def pdf_url(html: str) -> str | None:
    match = re.search(r"https://pakistancode\.gov\.pk/pdffiles/[A-Za-z0-9_-]+\.pdf", html)
    return match.group(0) if match else None


def safe_name(title: str) -> str:
    value = re.sub(r'[\\/*?:"<>|]', "_", title)
    return re.sub(r"\s+", " ", value).strip()[:150] + ".pdf"


def download_one(item: tuple[str, str], out_dir: Path, delay: float) -> dict[str, str]:
    detail_url, title = item
    session = requests.Session()
    session.headers.update(HEADERS)
    time.sleep(delay)
    detail = fetch(session, detail_url)
    base = {"title": title, "detail_url": detail_url, "pdf_url": "", "file": ""}
    if detail is None:
        return {**base, "status": "detail_fetch_failed"}
    target_url = pdf_url(detail.text)
    if not target_url:
        return {**base, "status": "no_pdf_found"}
    target = out_dir / safe_name(title)
    if target.exists() and target.stat().st_size > 0:
        return {**base, "pdf_url": target_url, "file": target.name, "status": "already_downloaded"}
    time.sleep(delay)
    downloaded = fetch(session, target_url, timeout=90)
    if downloaded is None:
        return {**base, "pdf_url": target_url, "status": "pdf_fetch_failed"}
    target.write_bytes(downloaded.content)
    return {**base, "pdf_url": target_url, "file": target.name, "status": "downloaded"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listing-url", action="append", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/documents/pakistan_code_pdfs"))
    parser.add_argument("--limit", type=int, default=50, help="maximum number of law pages to process")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.75)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)
    all_links: dict[str, str] = {}
    for listing in args.listing_url:
        response = fetch(session, listing)
        if response:
            all_links.update(law_links(response.text, listing))
    selected = list(all_links.items())[: max(0, args.limit)]
    print(f"Found {len(all_links)} law pages; processing first {len(selected)}")

    results: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 4))) as pool:
        futures = [pool.submit(download_one, item, args.out_dir, args.delay) for item in selected]
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(f"[{number}/{len(selected)}] {result['status']:20} {result['title'][:70]}")

    manifest = args.out_dir / "index.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["title", "detail_url", "pdf_url", "file", "status"])
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda row: row["title"].casefold()))
    downloaded = sum(row["status"] == "downloaded" for row in results)
    existing = sum(row["status"] == "already_downloaded" for row in results)
    print(f"Done. Downloaded={downloaded}, already_present={existing}, manifest={manifest}")


if __name__ == "__main__":
    main()

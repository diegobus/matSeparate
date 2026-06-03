#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import mimetypes
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

USER_AGENT = "matSeparate MINC-S downloader/1.0"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CONTENT_TYPE_EXTENSION = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


class FullResolutionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.current_href: str | None = None
        self.current_text_parts: list[str] = []
        self.full_resolution_href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        self.current_href = None
        self.current_text_parts = []
        for key, value in attrs:
            if key.lower() == "href" and value is not None:
                self.current_href = value
                break

    def handle_data(self, data: str) -> None:
        if self.current_href is not None:
            self.current_text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self.current_href is None:
            return
        text = " ".join(part.strip() for part in self.current_text_parts).strip().lower()
        if text == "full resolution" and self.full_resolution_href is None:
            self.full_resolution_href = self.current_href
        self.current_href = None
        self.current_text_parts = []


@dataclass
class DownloadResult:
    photo_id: str
    status: str
    page_url: str
    image_url: str | None = None
    output_path: str | None = None
    detail: str | None = None


def read_needed_photo_ids(segments_txt: Path) -> list[str]:
    photo_ids: set[str] = set()
    with open(segments_txt) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                continue
            _, photo_id, _ = parts
            photo_ids.add(photo_id)
    return sorted(photo_ids)


def read_existing_photo_ids(photos_dir: Path) -> set[str]:
    return {path.stem for path in photos_dir.glob("*") if path.is_file()}


def fetch_url(url: str, timeout: float) -> tuple[bytes, str, object]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()
        final_url = response.geturl()
        headers = response.info()
    return payload, final_url, headers


def fetch_text(url: str, timeout: float) -> tuple[str, str]:
    payload, final_url, headers = fetch_url(url, timeout=timeout)
    charset = headers.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace"), final_url


def extract_full_resolution_url(html: str, page_url: str) -> str | None:
    parser = FullResolutionParser()
    parser.feed(html)
    if parser.full_resolution_href is None:
        return None
    return urljoin(page_url, parser.full_resolution_href)


def choose_extension(image_url: str, headers: object) -> str:
    ext = Path(urlparse(image_url).path).suffix.lower()
    if ext in SUPPORTED_EXTENSIONS:
        return ext
    content_type = headers.get_content_type() if hasattr(headers, "get_content_type") else None
    if content_type in CONTENT_TYPE_EXTENSION:
        return CONTENT_TYPE_EXTENSION[content_type]
    guessed_ext = mimetypes.guess_extension(content_type or "") or ".jpg"
    if guessed_ext == ".jpe":
        guessed_ext = ".jpg"
    return guessed_ext


def resolve_existing_path(photos_dir: Path, photo_id: str) -> Path | None:
    for extension in SUPPORTED_EXTENSIONS:
        candidate = photos_dir / f"{photo_id}{extension}"
        if candidate.exists():
            return candidate
    return None


def download_one(
    photo_id: str,
    photos_dir: Path,
    page_template: str,
    timeout: float,
    retries: int,
    sleep_seconds: float,
) -> DownloadResult:
    existing = resolve_existing_path(photos_dir, photo_id)
    if existing is not None:
        return DownloadResult(
            photo_id=photo_id,
            status="exists",
            page_url=page_template.format(photo_num=int(photo_id), photo_id=photo_id),
            output_path=str(existing),
        )

    page_url = page_template.format(photo_num=int(photo_id), photo_id=photo_id)
    attempt = 0
    while True:
        try:
            html, resolved_page_url = fetch_text(page_url, timeout=timeout)
            image_url = extract_full_resolution_url(html, resolved_page_url)
            if image_url is None:
                return DownloadResult(
                    photo_id=photo_id,
                    status="missing_full_resolution_link",
                    page_url=resolved_page_url,
                    detail="No 'Full resolution' link found on photo page.",
                )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            payload, resolved_image_url, headers = fetch_url(image_url, timeout=timeout)
            extension = choose_extension(resolved_image_url, headers)
            output_path = photos_dir / f"{photo_id}{extension}"
            temp_path = output_path.with_suffix(output_path.suffix + ".part")
            temp_path.write_bytes(payload)
            temp_path.replace(output_path)
            return DownloadResult(
                photo_id=photo_id,
                status="downloaded",
                page_url=resolved_page_url,
                image_url=resolved_image_url,
                output_path=str(output_path),
                detail=f"{len(payload)} bytes",
            )
        except HTTPError as exc:
            detail = f"HTTP {exc.code}: {exc.reason}"
        except URLError as exc:
            detail = str(exc.reason)
        except Exception as exc:
            detail = repr(exc)
        attempt += 1
        if attempt > retries:
            return DownloadResult(
                photo_id=photo_id,
                status="failed",
                page_url=page_url,
                detail=detail,
            )
        time.sleep(max(0.5, sleep_seconds))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments-txt", type=Path, default=Path("data/external/minc/minc-s/test-segments.txt"))
    parser.add_argument("--photos-dir", type=Path, default=Path("data/external/minc/minc-s/photos"))
    parser.add_argument(
        "--page-template",
        type=str,
        default="http://opensurfaces.cs.cornell.edu/photos/{photo_num}/",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("data/external/minc/minc-s/download_missing_photos_report.json"),
    )
    parser.add_argument(
        "--missing-ids-out",
        type=Path,
        default=Path("data/external/minc/minc-s/missing_photo_ids.txt"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.photos_dir.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.missing_ids_out.parent.mkdir(parents=True, exist_ok=True)

    needed_photo_ids = read_needed_photo_ids(args.segments_txt)
    existing_photo_ids = read_existing_photo_ids(args.photos_dir)
    missing_photo_ids = [photo_id for photo_id in needed_photo_ids if photo_id not in existing_photo_ids]
    if args.limit is not None:
        missing_photo_ids = missing_photo_ids[: args.limit]

    args.missing_ids_out.write_text("\n".join(missing_photo_ids) + ("\n" if missing_photo_ids else ""))

    print(f"needed_unique_photos={len(needed_photo_ids)}")
    print(f"existing_unique_photos={len(existing_photo_ids)}")
    print(f"missing_unique_photos={len(missing_photo_ids)}")

    if args.dry_run:
        return

    started_at = time.time()
    results: list[DownloadResult] = []
    total = len(missing_photo_ids)
    if total == 0:
        report = {
            "needed_unique_photos": len(needed_photo_ids),
            "existing_unique_photos": len(existing_photo_ids),
            "attempted_missing_photos": 0,
            "elapsed_sec": 0.0,
            "status_counts": {},
            "results": [],
        }
        args.report_json.write_text(json.dumps(report, indent=2))
        return

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_photo_id = {
            executor.submit(
                download_one,
                photo_id,
                args.photos_dir,
                args.page_template,
                args.timeout,
                args.retries,
                args.sleep_seconds,
            ): photo_id
            for photo_id in missing_photo_ids
        }
        completed = 0
        for future in as_completed(future_to_photo_id):
            result = future.result()
            results.append(result)
            completed += 1
            if completed == total or completed % 25 == 0:
                counts = Counter(item.status for item in results)
                counts_str = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
                print(f"completed={completed}/{total} {counts_str}")

    elapsed_sec = time.time() - started_at
    status_counts = Counter(result.status for result in results)
    report = {
        "needed_unique_photos": len(needed_photo_ids),
        "existing_unique_photos_before_run": len(existing_photo_ids),
        "attempted_missing_photos": total,
        "elapsed_sec": elapsed_sec,
        "status_counts": dict(status_counts),
        "results": [asdict(result) for result in sorted(results, key=lambda item: item.photo_id)],
    }
    args.report_json.write_text(json.dumps(report, indent=2))
    print(json.dumps({"elapsed_sec": elapsed_sec, "status_counts": dict(status_counts)}, indent=2))


if __name__ == "__main__":
    main()

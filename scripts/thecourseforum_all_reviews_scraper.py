# Major component: Data collection and preprocessing.

# This is the original collection script, kept to show how we obtained the reviews.
# Do not rerun it: our agreement with thecourseforum does not allow further scraping.

# Input: live course and instructor pages, plus any saved progress file.
# Output: the original review csv, a progress file, and a log of the collection.

"""
thecourseforum_all_reviews_scraper.py

Purpose:
Scrape every reachable review from theCourseForum by traversing:
Browse -> Departments -> Courses -> Course-Offering pages -> Paginated reviews.

This script is designed to be resilient to layout shifts by using multiple extraction
strategies (review cards, review links, and loose text parsing) and persistent checkpoints.

Usage examples:
    python thecourseforum_all_reviews_scraper.py
    python thecourseforum_all_reviews_scraper.py --output all_reviews.csv --checkpoint all_reviews_checkpoint.json
    python thecourseforum_all_reviews_scraper.py --headful --min-delay 0.8 --max-delay 1.8

Notes:
- Respect the website's Terms of Service and robots policies.
- Use conservative delays to avoid excessive load.
- This script is created per request but is not executed automatically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Dict, Iterable, List, Optional, Set, Tuple

from selenium import webdriver
from selenium.common.exceptions import InvalidSessionIdException, TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

BASE_URL = "https://thecourseforum.com"
BROWSE_URL = f"{BASE_URL}/browse/"

CSV_FIELDS = [
    "review_id",
    "department",
    "course_code",
    "course_title",
    "instructor_name",
    "semester_taken",
    "review_date",
    "overall_rating",
    "difficulty_rating",
    "instructor_rating",
    "enjoyability_rating",
    "recommend_rating",
    "hours_per_week",
    "upvotes",
    "review_text",
    "review_url",
    "source_url",
    "scraped_at",
]


@dataclass
class Config:
    output_csv: str
    checkpoint_json: str
    log_file: str
    headless: bool
    page_load_timeout: int
    wait_timeout: int
    min_delay: float
    max_delay: float
    max_pages_per_offering: int
    max_departments: int


# Save collection messages to the terminal and a log file so problems can be traced.
def setup_logging(log_file: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
    )


# Open the browser used for the original collection and apply its settings.
def build_driver(config: Config) -> webdriver.Chrome:
    options = Options()
    if config.headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_experimental_option(
        "prefs",
        {
            "profile.managed_default_content_settings.images": 2,
            "profile.default_content_setting_values.notifications": 2,
        },
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(config.page_load_timeout)
    return driver


# Leave a short pause between requests during the original collection.
def sleep_polite(config: Config) -> None:
    time.sleep(random.uniform(config.min_delay, config.max_delay))


# Replace repeated whitespace with single spaces to make page text easier to compare.
def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


# Use visible text when available, then try the element's text content as a fallback.
def safe_text(elem) -> str:
    text = (elem.text or "").strip()
    if text:
        return text
    return (elem.get_attribute("textContent") or "").strip()


# Write column names when creating the output file, without replacing an existing file.
def ensure_output_header(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


# Add newly collected reviews to the output file using the same column order.
def append_rows(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        for row in rows:
            writer.writerow(row)


# Read saved progress so an interrupted collection could resume.
def load_checkpoint(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {
            "completed_offerings": [],
            "seen_review_ids": [],
            "next_department_index": 0,
            "updated_at": None,
        }
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("completed_offerings", [])
    data.setdefault("seen_review_ids", [])
    data.setdefault("next_department_index", 0)
    data.setdefault("updated_at", None)
    return data


# Save completed pages and review identifiers so progress is not lost after an interruption.
def save_checkpoint(path: str, checkpoint: Dict[str, object]) -> None:
    checkpoint["updated_at"] = datetime.now(UTC).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


# Recognize a lost browser session so collection can stop with its progress saved.
def is_invalid_session_error(exc: BaseException) -> bool:
    if isinstance(exc, InvalidSessionIdException):
        return True
    return "invalid session id" in str(exc).lower()


# Create a repeatable review identifier to help avoid saving the same record twice.
def hash_review_key(*parts: str) -> str:
    payload = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.md5(payload).hexdigest()


# Read a number that appears after a rating label, returning no value if it cannot be read.
def extract_metric_after_label(text: str, label: str) -> Optional[float]:
    m = re.search(rf"{label}\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# Check the alternate layout where the number comes before its rating label.
def extract_metric_before_label(text: str, label: str) -> Optional[float]:
    m = re.search(rf"([0-9]+(?:\.[0-9]+)?)\s*{label}", text, re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# Look for a rating before or after its label because page layouts can differ.
def extract_metric_flexible(text: str, label: str) -> Optional[float]:
    value = extract_metric_after_label(text, label)
    if value is not None:
        return value
    return extract_metric_before_label(text, label)


# Find the term and year in the collected page text.
def parse_semester(text: str) -> str:
    m = re.search(r"\b(Fall|Spring|Summer|Winter|January)\s+20\d{2}\b", text)
    return m.group(0) if m else ""


# Find a written posting date using the month names supported by this pattern.
def parse_date(text: str) -> str:
    m = re.search(
        r"\b(Jan(?:uary|\.)?|Feb(?:ruary|\.)?|Mar(?:ch|\.)?|Apr(?:il|\.)?|May|"
        r"Jun(?:e|\.)?|Jul(?:y|\.)?|Aug(?:ust|\.)?|Sep(?:tember|\.)?|"
        r"Oct(?:ober|\.)?|Nov(?:ember|\.)?|Dec(?:ember|\.)?)\s+\d{1,2},\s+20\d{2}\b",
        text,
    )
    return m.group(0) if m else ""


# Separate the course code and title from the page heading when they are available.
def parse_course_meta(header_text: str) -> Tuple[str, str]:
    code = ""
    title = ""

    lines = [ln.strip() for ln in header_text.split("\n") if ln.strip()]
    if lines:
        code_match = re.search(r"\b([A-Z]{2,5}\s*\d{4})\b", lines[0])
        if code_match:
            code = normalize_space(code_match.group(1))
    if len(lines) >= 2:
        title = normalize_space(lines[1])

    if not code:
        inline_match = re.search(r"\b([A-Z]{2,5}\s*\d{4})\b", header_text)
        if inline_match:
            code = normalize_space(inline_match.group(1))

    return code, title


# Find department links, which were the starting points for collection.
def collect_departments(driver: webdriver.Chrome, wait: WebDriverWait, max_departments: int) -> List[Dict[str, str]]:
    driver.get(BROWSE_URL)
    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.department-card")))

    departments: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for elem in driver.find_elements(By.CSS_SELECTOR, "a.department-card"):
        href = elem.get_attribute("href") or ""
        name = normalize_space(elem.get_attribute("textContent") or "")
        if not href or href in seen:
            continue
        departments.append({"name": name, "url": href})
        seen.add(href)
        if max_departments > 0 and len(departments) >= max_departments:
            break

    logging.info("Departments discovered: %d", len(departments))
    return departments


# Find each course within a department and skip repeated links.
def collect_courses(driver: webdriver.Chrome, dept_url: str) -> List[Dict[str, str]]:
    driver.get(dept_url)

    courses: List[Dict[str, str]] = []
    seen: Set[str] = set()

    anchors = driver.find_elements(
        By.XPATH,
        "//a[contains(@href, '/course/') and not(contains(@href, '?mode='))]",
    )
    for a in anchors:
        href = a.get_attribute("href") or ""
        if not href:
            continue
        if not re.search(r"/course/[A-Z]+/\d+/?$", href):
            continue
        if href in seen:
            continue

        label_text = safe_text(a)
        code, title = parse_course_meta(label_text)
        courses.append({"url": href, "label": label_text, "course_code": code, "course_title": title})
        seen.add(href)

    return courses


# Find the instructor pages connected to each course.
def collect_offerings(driver: webdriver.Chrome, course_url: str) -> List[Dict[str, str]]:
    driver.get(course_url)

    offerings: List[Dict[str, str]] = []
    seen: Set[str] = set()

    page_title = ""
    h1 = driver.find_elements(By.TAG_NAME, "h1")
    if h1:
        page_title = safe_text(h1[0])

    for a in driver.find_elements(By.XPATH, "//a[contains(@href, '/course/')]"):
        href = a.get_attribute("href") or ""
        if not re.search(r"/course/\d+/\d+/?$", href):
            continue
        if href in seen:
            continue
        seen.add(href)

        instructor_text = safe_text(a)
        instructor_name = normalize_space(instructor_text.split("\n")[0]) if instructor_text else ""
        offerings.append(
            {
                "offering_url": href,
                "instructor_name": instructor_name,
                "course_meta": page_title,
            }
        )

    return offerings


# Look for page sections that contain reviews using several possible layouts.
def possible_review_blocks(driver: webdriver.Chrome) -> Iterable[Tuple[str, Optional[str]]]:
    selectors = [
        "[data-testid*='review']",
        ".review-card",
        ".review",
        "article",
        "li",
        ".card",
    ]

    seen_texts: Set[str] = set()

    for selector in selectors:
        for elem in driver.find_elements(By.CSS_SELECTOR, selector):
            text = safe_text(elem)
            if len(text) < 40:
                continue
            text_norm = normalize_space(text)
            if text_norm in seen_texts:
                continue

            # Skip page navigation and other sections that do not look like reviews.
            has_metric_token = any(k in text for k in ["Difficulty", "Instructor", "Enjoyability", "Recommend"]) 
            has_time_token = bool(parse_date(text) or parse_semester(text))
            if not (has_metric_token or has_time_token):
                continue

            link = None
            links = elem.find_elements(By.XPATH, ".//a[contains(@href, '/review') or contains(@href, '/reviews')]")
            if links:
                link = links[0].get_attribute("href") or None

            seen_texts.add(text_norm)
            yield text, link


# Pull the review text, ratings, and course details into one row.
def parse_review_blob(
    blob_text: str,
    review_url: Optional[str],
    source_url: str,
    department: str,
    course_code: str,
    course_title: str,
    instructor_name: str,
) -> Optional[Dict[str, object]]:
    text = blob_text.strip()
    if not text:
        return None

    semester_taken = parse_semester(text)
    review_date = parse_date(text)

    overall_rating = extract_metric_flexible(text, r"(?:☆\s*)?Rating")
    difficulty_rating = extract_metric_flexible(text, "Difficulty")
    instructor_rating = extract_metric_flexible(text, "Instructor")
    enjoyability_rating = extract_metric_flexible(text, "Enjoyability")
    recommend_rating = extract_metric_flexible(text, "Recommend")
    hours_per_week = extract_metric_flexible(text, "Hours/Week")
    if hours_per_week is None:
        hours_per_week = extract_metric_flexible(text, "Total Hours")

    upvotes = 0
    upvote_match = re.search(r"\b([\-\+]?\d{1,4})\b\s*(?:upvote|helpful)", text, re.IGNORECASE)
    if upvote_match:
        try:
            upvotes = int(upvote_match.group(1))
        except ValueError:
            upvotes = 0

    # Keep the written review while removing some rating information at the end.
    review_text = re.sub(
        r"(Instructor|Enjoyability|Recommend|Difficulty|Hours/Week|Total Hours)\s*[0-9\.\-\+—NAnan/ ]+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    review_text = normalize_space(review_text)

    # Use the available review details to create a repeatable identifier.
    review_id = hash_review_key(
        review_url or "",
        course_code or "",
        instructor_name or "",
        review_date or "",
        review_text[:180],
    )

    # Keep rows only when they contain enough information to look like a review.
    if not any(v is not None for v in [overall_rating, difficulty_rating, instructor_rating, enjoyability_rating]) and len(review_text) < 60:
        return None

    return {
        "review_id": review_id,
        "department": department,
        "course_code": course_code,
        "course_title": course_title,
        "instructor_name": instructor_name,
        "semester_taken": semester_taken,
        "review_date": review_date,
        "overall_rating": overall_rating,
        "difficulty_rating": difficulty_rating,
        "instructor_rating": instructor_rating,
        "enjoyability_rating": enjoyability_rating,
        "recommend_rating": recommend_rating,
        "hours_per_week": hours_per_week,
        "upvotes": upvotes,
        "review_text": review_text,
        "review_url": review_url or "",
        "source_url": source_url,
        "scraped_at": datetime.now(UTC).isoformat(),
    }


# Read the review pages for one offering and skip records already collected.
def scrape_offering_reviews(
    driver: webdriver.Chrome,
    offering: Dict[str, str],
    department: str,
    max_pages: int,
    seen_review_ids: Set[str],
) -> List[Dict[str, object]]:
    offering_url = offering["offering_url"].split("?")[0]
    instructor_name = offering.get("instructor_name", "")
    course_meta = offering.get("course_meta", "")
    course_code, course_title = parse_course_meta(course_meta)

    rows: List[Dict[str, object]] = []

    for page in range(1, max_pages + 1):
        source_url = f"{offering_url}?page={page}"
        try:
            driver.get(source_url)
        except TimeoutException:
            logging.warning("Timeout loading %s", source_url)
            break

        # Check the possible review sections found on this page.
        page_candidates = list(possible_review_blocks(driver))
        if not page_candidates:
            break

        page_new = 0
        for blob_text, review_url in page_candidates:
            parsed = parse_review_blob(
                blob_text=blob_text,
                review_url=review_url,
                source_url=source_url,
                department=department,
                course_code=course_code,
                course_title=course_title,
                instructor_name=instructor_name,
            )
            if not parsed:
                continue
            rid = str(parsed["review_id"])
            if rid in seen_review_ids:
                continue
            seen_review_ids.add(rid)
            rows.append(parsed)
            page_new += 1

        logging.info("Offering %s page %d: +%d reviews", offering_url, page, page_new)

        # Stop when the page has no button or link for more reviews.
        next_links = driver.find_elements(
            By.XPATH,
            "//a[contains(translate(., 'NEXT', 'next'),'next') and not(contains(@class,'disabled'))]",
        )
        if not next_links:
            break

    return rows


# Read the original collection settings, including file locations, delays, and page limits.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape all reachable theCourseForum reviews")
    parser.add_argument("--output", default="thecourseforum_all_reviews.csv", help="Output CSV path")
    parser.add_argument("--checkpoint", default="thecourseforum_all_reviews_checkpoint.json", help="Checkpoint JSON path")
    parser.add_argument("--log-file", default="all_reviews_scraper.log", help="Log file path")
    parser.add_argument("--headful", action="store_true", help="Run with visible browser")
    parser.add_argument("--page-load-timeout", type=int, default=25, help="Page load timeout seconds")
    parser.add_argument("--wait-timeout", type=int, default=15, help="Explicit wait timeout seconds")
    parser.add_argument("--min-delay", type=float, default=0.8, help="Minimum delay between requests")
    parser.add_argument("--max-delay", type=float, default=1.8, help="Maximum delay between requests")
    parser.add_argument("--max-pages-per-offering", type=int, default=40, help="Pagination cap per offering")
    parser.add_argument("--max-departments", type=int, default=0, help="Limit departments for test runs (0 = all)")
    return parser.parse_args()


# Follow departments, courses, and offerings, saving reviews and progress along the way.
def main() -> None:
    args = parse_args()
    config = Config(
        output_csv=args.output,
        checkpoint_json=args.checkpoint,
        log_file=args.log_file,
        headless=not args.headful,
        page_load_timeout=args.page_load_timeout,
        wait_timeout=args.wait_timeout,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        max_pages_per_offering=args.max_pages_per_offering,
        max_departments=args.max_departments,
    )

    setup_logging(config.log_file)
    ensure_output_header(config.output_csv)

    # Restore the last saved position and the reviews already collected.
    checkpoint = load_checkpoint(config.checkpoint_json)
    completed_offerings: Set[str] = set(checkpoint.get("completed_offerings", []))
    seen_review_ids: Set[str] = set(checkpoint.get("seen_review_ids", []))
    next_department_index = int(checkpoint.get("next_department_index", 0) or 0)

    logging.info("Starting all-reviews crawl")
    logging.info("Checkpoint loaded: %d completed offerings, %d seen reviews", len(completed_offerings), len(seen_review_ids))

    driver = None
    total_written = 0

    try:
        driver = build_driver(config)
        wait = WebDriverWait(driver, config.wait_timeout)

        departments = collect_departments(driver, wait, config.max_departments)

        if next_department_index < 0:
            next_department_index = 0
        if next_department_index >= len(departments):
            logging.info(
                "Checkpoint cursor (%d) is at or beyond discovered departments (%d). "
                "Starting from the beginning.",
                next_department_index,
                len(departments),
            )
            next_department_index = 0

        if next_department_index > 0:
            logging.info(
                "Resuming from department index %d of %d",
                next_department_index,
                len(departments),
            )

        for d_idx in range(next_department_index, len(departments)):
            dept = departments[d_idx]
            dept_name = dept["name"]
            dept_url = dept["url"]
            logging.info("[%d/%d] Department: %s", d_idx + 1, len(departments), dept_name)

            try:
                courses = collect_courses(driver, dept_url)
            except WebDriverException as exc:
                logging.error("Department failed (%s): %s", dept_name, exc)
                if is_invalid_session_error(exc):
                    logging.error(
                        "Browser session became invalid while processing department '%s'. "
                        "Checkpoint preserved at this department for resume.",
                        dept_name,
                    )
                    break
                continue

            logging.info("  Courses found: %d", len(courses))

            for c_idx, course in enumerate(courses, start=1):
                course_url = course["url"]
                logging.info("  -> Course [%d/%d]: %s", c_idx, len(courses), course_url)

                try:
                    offerings = collect_offerings(driver, course_url)
                except WebDriverException as exc:
                    logging.error("  Course failed (%s): %s", course_url, exc)
                    if is_invalid_session_error(exc):
                        logging.error(
                            "Browser session became invalid while processing course '%s'. "
                            "Checkpoint preserved at this department for resume.",
                            course_url,
                        )
                        break
                    continue

                if not offerings:
                    continue

                for off in offerings:
                    offering_url = off["offering_url"].split("?")[0]
                    if offering_url in completed_offerings:
                        continue

                    sleep_polite(config)
                    rows = scrape_offering_reviews(
                        driver=driver,
                        offering=off,
                        department=dept_name,
                        max_pages=config.max_pages_per_offering,
                        seen_review_ids=seen_review_ids,
                    )

                    # Append new reviews before marking this offering as finished.
                    if rows:
                        append_rows(config.output_csv, rows)
                        total_written += len(rows)
                        logging.info("    Wrote %d reviews (running total this run: %d)", len(rows), total_written)

                    completed_offerings.add(offering_url)
                    checkpoint["completed_offerings"] = sorted(completed_offerings)
                    checkpoint["seen_review_ids"] = sorted(seen_review_ids)
                    save_checkpoint(config.checkpoint_json, checkpoint)

                else:
                    continue
                break

            else:
                checkpoint["next_department_index"] = d_idx + 1
                save_checkpoint(config.checkpoint_json, checkpoint)
                continue

            break

        logging.info("Crawl complete. New rows written this run: %d", total_written)
        logging.info("Total known unique review IDs in checkpoint: %d", len(seen_review_ids))

    # Close the browser even if the original collection ended with an error.
    finally:
        if driver is not None:
            driver.quit()


if __name__ == "__main__":
    main()

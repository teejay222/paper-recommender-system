#!/usr/bin/env python3
"""Collect yearly arXiv papers and citation counts."""

import os
import time
import csv
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
import requests
from dotenv import load_dotenv

# Run settings
TARGET_YEAR = 2026           # batch year
TARGET_PAPERS = 8333         # target rows

# Read secrets from .env.
load_dotenv()
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

# CS categories
ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]

# API settings
ARXIV_BASE_URL   = "https://export.arxiv.org/api/query"
ARXIV_BATCH_SIZE = 100          # page size
ARXIV_DELAY      = 3.0          # API pause

SS_BATCH_URL     = "https://api.semanticscholar.org/graph/v1/paper/batch"
SS_FIELDS        = "citationCount,referenceCount"
SS_BATCH_SIZE    = 500
SS_BATCH_DELAY   = 1.0

# Paths
OUTPUT_DIR      = os.path.join("data", "raw")
OUTPUT_FILE     = os.path.join(OUTPUT_DIR, f"papers_{TARGET_YEAR}.csv")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, f"collection_state_{TARGET_YEAR}.json")

# Paper schema
CSV_HEADERS = [
    "arxiv_id",
    "title",
    "abstract",
    "authors",
    "category",
    "publication_year",
    "citation_count",
    "reference_count",
    "published_date",
    "arxiv_url"
]

ARXIV_NAMESPACE = {
    "atom":  "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom"
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Files and checkpoint
def init_storage():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()

def append_records_to_csv(records):
    if not records:
        return
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerows(records)

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return {"category_index": 0, "start_index": 0, "total_collected": 0}
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"category_index": 0, "start_index": 0, "total_collected": 0}

def save_checkpoint(category_index, start_index, total_collected):
    state = {
        "category_index":  category_index,
        "start_index":     start_index,
        "total_collected": total_collected
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)

# arXiv fetch
def build_arxiv_query(category, start_index):
    date_from = f"{TARGET_YEAR}01010000"
    date_to   = f"{TARGET_YEAR}12312359"
    return {
        "search_query": f"cat:{category} AND submittedDate:[{date_from} TO {date_to}]",
        "start":        start_index,
        "max_results":  ARXIV_BATCH_SIZE,
        "sortBy":       "submittedDate",
        "sortOrder":    "descending"
    }

def parse_arxiv_xml(xml_text):
    """Parse arxiv xml."""
    papers = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.error("XML Syntax Error encountered within response payload string.")
        return papers, 0

    all_entries = root.findall("atom:entry", ARXIV_NAMESPACE)
    raw_count   = len(all_entries)

    for entry in all_entries:
        # Paper ids
        raw_id   = entry.findtext("atom:id", default="", namespaces=ARXIV_NAMESPACE)
        arxiv_id = raw_id.split("/abs/")[-1].strip()
        if "v" in arxiv_id:
            arxiv_id = arxiv_id.rsplit("v", 1)[0]

        # Read every category on the paper.
        cat_elements = entry.findall("atom:category", ARXIV_NAMESPACE)
        all_categories = [c.get("term") for c in cat_elements if c.get("term") is not None]
        
        # Keep direct cross-list matches.
        matching_targets = [c for c in all_categories if c in ARXIV_CATEGORIES]

        if not matching_targets:
            continue  # Safe bypass: Drops pure math/physics items with no CS relationships

        # Use the first matching CS field.
        assigned_category = matching_targets[0]

        title    = entry.findtext("atom:title", default="", namespaces=ARXIV_NAMESPACE)
        abstract = entry.findtext("atom:summary", default="", namespaces=ARXIV_NAMESPACE)
        published = entry.findtext("atom:published", default="", namespaces=ARXIV_NAMESPACE)
        publication_year = published[:4] if published else str(TARGET_YEAR)

        author_nodes = entry.findall("atom:author", ARXIV_NAMESPACE)
        authors = "; ".join([a.findtext("atom:name", default="", namespaces=ARXIV_NAMESPACE) for a in author_nodes])

        papers.append({
            "arxiv_id":         arxiv_id,
            "title":            " ".join(title.split()),
            "abstract":         " ".join(abstract.split()),
            "authors":          authors,
            "category":         assigned_category,
            "publication_year": publication_year,
            "citation_count":   0,   # Filled in Step 2
            "reference_count":  0,   # Filled in Step 2
            "published_date":   published,
            "arxiv_url":        raw_id
        })

    return papers, raw_count

def fetch_arxiv_batch(category, start_index):
    params = build_arxiv_query(category, start_index)
    resp = requests.get(ARXIV_BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    return parse_arxiv_xml(resp.text)

# Semantic Scholar metrics
def enrich_batch_metrics(papers):
    if not papers:
        return papers
        
    headers = {"Content-Type": "application/json", "x-api-key": SEMANTIC_SCHOLAR_API_KEY}
    id_map  = {f"ArXiv:{p['arxiv_id']}": i for i, p in enumerate(papers)}
    ids     = list(id_map.keys())

    try:
        resp = requests.post(SS_BATCH_URL, headers=headers, params={"fields": SS_FIELDS}, json={"ids": ids}, timeout=45)
        resp.raise_for_status()
        results = resp.json()
        
        for ss_id, metric_set in zip(ids, results):
            if metric_set is None:
                continue
            idx = id_map.get(ss_id)
            if idx is not None:
                papers[idx]["citation_count"]  = metric_set.get("citationCount", 0)
                papers[idx]["reference_count"] = metric_set.get("referenceCount", 0)
    except Exception as err:
        logger.warning("Metrics enrichment pass bypassed for current block: %s", err)
        
    return papers

# Main run
def run_pipeline():
    init_storage()
    state = load_checkpoint()

    total_collected = state["total_collected"]
    category_index  = state["category_index"]
    start_index     = state["start_index"]

    logger.info("=" * 70)
    logger.info("Yearly Pipeline Active | Target Horizon: %d | Goal: %d Papers", TARGET_YEAR, TARGET_PAPERS)
    logger.info("Active Restore Checkpoint | Current Pool Size: %d", total_collected)
    logger.info("=" * 70)

    while total_collected < TARGET_PAPERS:
        category = ARXIV_CATEGORIES[category_index]
        logger.info("Query Phase | Target Loop Channel: %s | API Cursor Index: %d", category, start_index)

        try:
            batch, raw_count = fetch_arxiv_batch(category, start_index)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning("Rate limit (429) flagged by Firewall. Commencing 60s cooldown...")
                time.sleep(60)
            else:
                logger.error("HTTP Transport Error: %s. Re-attempting loop step in 30s.", e)
                time.sleep(30)
            continue
        except Exception as e:
            logger.error("Unexpected System Fault: %s. Re-attempting loop step in 30s.", e)
            time.sleep(30)
            continue

        # Move on only when this category is empty.
        if raw_count == 0:
            logger.info("Channel Exhausted | Category %s catalog complete. Switching loop focus.", category)
            category_index += 1
            start_index = 0
            if category_index >= len(ARXIV_CATEGORIES):
                logger.info("Data Exhaustion Alert. All targeted archive endpoints fully parsed.")
                break
            save_checkpoint(category_index, start_index, total_collected)
            continue

        # Advance past empty cross-list gaps.
        if not batch:
            logger.info("Padding Window | Skipping batch containing only unaligned cross-listings (+%d offsets)", raw_count)
            start_index += raw_count
            save_checkpoint(category_index, start_index, total_collected)
            time.sleep(ARXIV_DELAY)
            continue

        # Stop at the target count.
        remaining = TARGET_PAPERS - total_collected
        batch = batch[:remaining]

        # Fetch citation data in bulk.
        batch = enrich_batch_metrics(batch)

        # Write the CSV checkpoint.
        append_records_to_csv(batch)
        total_collected += len(batch)
        
        # Advance by the feed size.
        start_index += raw_count
        save_checkpoint(category_index, start_index, total_collected)

        logger.info("Extraction Metric Tracker | Progress Status: %d / %d records persisted", total_collected, TARGET_PAPERS)
        time.sleep(ARXIV_DELAY)

    # Remove the checkpoint after a full run.
    if total_collected >= TARGET_PAPERS and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    logger.info("=" * 70)
    logger.info("Pipeline Execution Finalized. Asset Available At: %s", OUTPUT_FILE)
    logger.info("=" * 70)

if __name__ == "__main__":
    run_pipeline()
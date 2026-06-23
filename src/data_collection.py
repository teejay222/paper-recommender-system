#!/usr/bin/env python3
"""
src/data_collection_yearly.py
-----------------------------
Collects a high-quality, targeted volume of papers for a single year.
Uses an inclusive yet safe multi-category scanner across all secondary listings,
and fixes pagination offsets by tracking raw API payload item counts.

Outputs:
  - data/raw/papers_YYYY.csv
"""

import os
import time
import csv
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
import requests
from dotenv import load_dotenv

# ===========================================================================
# CONFIGURATION - ADJUST PER RUN
# ===========================================================================
TARGET_YEAR = 2026           # Target calendar year for this isolated batch
TARGET_PAPERS = 8333         # Production batch volume constraint

# Loaded from .env — never hard-code secrets.
load_dotenv()
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

# Bounded Computer Science target fields
ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]

# API Endpoints & Transport Rules
ARXIV_BASE_URL   = "https://export.arxiv.org/api/query"
ARXIV_BATCH_SIZE = 100          # Stable pagination block size
ARXIV_DELAY      = 3.0          # 3-second delay to protect against IP bans

SS_BATCH_URL     = "https://api.semanticscholar.org/graph/v1/paper/batch"
SS_FIELDS        = "citationCount,referenceCount"
SS_BATCH_SIZE    = 500
SS_BATCH_DELAY   = 1.0

# File System Workspace Definitions
OUTPUT_DIR      = os.path.join("data", "raw")
OUTPUT_FILE     = os.path.join(OUTPUT_DIR, f"papers_{TARGET_YEAR}.csv")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, f"collection_state_{TARGET_YEAR}.json")

# Document schema matching core specifications
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

# ===========================================================================
# LOGGING SYSTEM
# ===========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ===========================================================================
# FILE I/O OPERATIONS & CHECKPOINT MANAGEMENT
# ===========================================================================
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

# ===========================================================================
# ARXIV CORE CONTEXT RETRIEVER
# ===========================================================================
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
    """
    Parses entry feeds using an intersection search across ALL attached categories.
    Returns parsed paper models alongside the true, unfiltered payload record count.
    """
    papers = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.error("XML Syntax Error encountered within response payload string.")
        return papers, 0

    all_entries = root.findall("atom:entry", ARXIV_NAMESPACE)
    raw_count   = len(all_entries)

    for entry in all_entries:
        # Extract metadata identifiers
        raw_id   = entry.findtext("atom:id", default="", namespaces=ARXIV_NAMESPACE)
        arxiv_id = raw_id.split("/abs/")[-1].strip()
        if "v" in arxiv_id:
            arxiv_id = arxiv_id.rsplit("v", 1)[0]

        # MULTI-CATEGORY SAFE SCANNER: Extract all secondary classification mappings
        cat_elements = entry.findall("atom:category", ARXIV_NAMESPACE)
        all_categories = [c.get("term") for c in cat_elements if c.get("term") is not None]
        
        # Isolate direct cross-listed intersection matches
        matching_targets = [c for c in all_categories if c in ARXIV_CATEGORIES]

        if not matching_targets:
            continue  # Safe bypass: Drops pure math/physics items with no CS relationships

        # Anchor document safely under the first matching CS target field
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

# ===========================================================================
# SEMANTIC SCHOLAR BULK METRICS METRIC ENRICHER
# ===========================================================================
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

# ===========================================================================
# ORCHESTRATION PIPELINE
# ===========================================================================
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

        # CRITICAL RECOVERY LOGIC: Move to next category ONLY if the API returns 0 total items
        if raw_count == 0:
            logger.info("Channel Exhausted | Category %s catalog complete. Switching loop focus.", category)
            category_index += 1
            start_index = 0
            if category_index >= len(ARXIV_CATEGORIES):
                logger.info("Data Exhaustion Alert. All targeted archive endpoints fully parsed.")
                break
            save_checkpoint(category_index, start_index, total_collected)
            continue

        # If the batch list is empty but raw_count > 0, advance the cursor past the cross-listed gap
        if not batch:
            logger.info("Padding Window | Skipping batch containing only unaligned cross-listings (+%d offsets)", raw_count)
            start_index += raw_count
            save_checkpoint(category_index, start_index, total_collected)
            time.sleep(ARXIV_DELAY)
            continue

        # Cap entries to prevent target overshoot
        remaining = TARGET_PAPERS - total_collected
        batch = batch[:remaining]

        # Enqueue live citation profiles via bulk POST
        batch = enrich_batch_metrics(batch)

        # Flush data down to CSV file system
        append_records_to_csv(batch)
        total_collected += len(batch)
        
        # Advance pagination using the absolute entry response length
        start_index += raw_count
        save_checkpoint(category_index, start_index, total_collected)

        logger.info("Extraction Metric Tracker | Progress Status: %d / %d records persisted", total_collected, TARGET_PAPERS)
        time.sleep(ARXIV_DELAY)

    # Clean up checkpoint on perfect job completion
    if total_collected >= TARGET_PAPERS and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    logger.info("=" * 70)
    logger.info("Pipeline Execution Finalized. Asset Available At: %s", OUTPUT_FILE)
    logger.info("=" * 70)

if __name__ == "__main__":
    run_pipeline()
"""
build_synonyms.py — Auto-Synonym Extraction Pipeline

Mines synonym clusters from the scheme corpus using:
  1. TF-IDF to extract important terms from high-signal fields
  2. SentenceTransformer to embed extracted terms
  3. Cosine similarity to discover synonym pairs
  4. Transitive closure to form synonym groups
  5. Saves synonym_map.json for query expansion at search time

Run once (or whenever dataset changes):
  python build_synonyms.py
"""

import os
import json
import time
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

_BASE_DIR = os.path.dirname(__file__)
INDEXED_CSV = os.path.join(_BASE_DIR, "schemes_indexed.csv")
RAW_DATA = os.path.join(_BASE_DIR, "updated_data.csv")
SYNONYM_OUTPUT = os.path.join(_BASE_DIR, "synonym_map.json")
MODEL_NAME = "all-MiniLM-L6-v2"

# Cosine similarity threshold for linking two terms as synonyms.
# 0.72 balances precision (avoiding false chains) and recall (catching Hindi-English pairs).
# Was 0.65 initially but caused mega-clusters via transitive closure.
SIM_THRESHOLD = 0.72

# Maximum group size — if a cluster grows larger than this, it's too loose.
# We split it into tighter sub-clusters using a higher threshold.
MAX_GROUP_SIZE = 25

# TF-IDF parameters
MIN_DF = 5       # Term must appear in at least 5 schemes
MAX_DF = 0.70    # Term must appear in fewer than 70% of schemes
MAX_FEATURES = 2000  # Cap vocabulary to top 2000 terms by TF-IDF score

# ── Domain Stopwords ──────────────────────────────────────────────────────────
# These pass TF-IDF filters but carry zero discriminative value for user-intent matching.

DOMAIN_STOPWORDS = {
    # Procedural terms (every scheme has these)
    "scheme", "schemes", "government", "govt", "applicant", "applicants",
    "application", "applications", "department", "ministry", "board",
    "documents", "required", "apply", "form", "portal", "online",
    "submission", "submit", "register", "registration", "login",
    "certificate", "certificates", "document", "proof",
    # Geographic noise
    "state", "states", "central", "india", "indian", "national",
    "pradesh", "district", "block", "village", "urban", "rural",
    # Generic descriptors
    "benefits", "benefit", "eligibility", "eligible", "objective",
    "objectives", "details", "detail", "features", "feature",
    "shall", "may", "must", "provide", "provided", "providing",
    "given", "available", "avail", "respective",
    # Temporal / numeric noise
    "year", "years", "month", "months", "date", "age",
    "rs", "inr", "per", "annum", "lakh", "crore", "rupees",
    "01", "02", "03", "04", "05", "10", "12", "15", "18", "20", "21",
    "25", "30", "35", "40", "45", "50", "55", "60", "65", "100",
    # Web noise
    "www", "http", "https", "com", "gov", "nic", "org", "html",
    # Filler words that bypass English stopwords
    "also", "however", "including", "following", "mentioned",
    "concerned", "already", "within", "along", "towards", "upto",
    "category", "categories", "tags", "tag",
}


# ── Step 1: Extract High-Signal Text ─────────────────────────────────────────

def extract_high_signal_texts(df: pd.DataFrame) -> list:
    """
    Build text corpus from the MOST discriminative fields only.
    Skip 'application' and 'documents' (procedural noise).
    Repeat scheme_name and tags for emphasis (short but high signal).
    """
    HIGH_SIGNAL_FIELDS = ["scheme_name", "tags", "eligibility", "benefits"]
    MEDIUM_SIGNAL_FIELDS = ["details"]

    texts = []
    for _, row in df.iterrows():
        parts = []

        # High-signal fields: include twice for emphasis
        for col in HIGH_SIGNAL_FIELDS:
            val = str(row.get(col, "")).strip()
            if val and val.lower() != "nan":
                parts.append(val)
                parts.append(val)  # Repeat for TF boost

        # Medium-signal: include once, truncated to reduce dilution
        for col in MEDIUM_SIGNAL_FIELDS:
            val = str(row.get(col, "")).strip()
            if val and val.lower() != "nan":
                # Keep first 300 words to avoid long descriptions drowning out terms
                parts.append(" ".join(val.split()[:300]))

        texts.append(" ".join(parts))

    return texts


# ── Step 2: TF-IDF Term Extraction ───────────────────────────────────────────

def extract_important_terms(texts: list) -> list:
    """
    Use TF-IDF to surface terms that are important but not ubiquitous.
    Returns sorted list of (term, avg_tfidf_score) tuples.
    """
    print("\n[Step 2] Running TF-IDF term extraction...")

    vectorizer = TfidfVectorizer(
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # Words only, 2+ chars
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    print(f"  TF-IDF vocabulary size (before filtering): {len(feature_names)}")

    # Compute average TF-IDF score across all documents for each term
    avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

    # Filter out domain stopwords
    filtered_terms = []
    removed_count = 0
    for term, score in zip(feature_names, avg_scores):
        # Check if ANY word in the term (for bigrams) is a domain stopword
        term_words = set(term.lower().split())
        if term_words & DOMAIN_STOPWORDS:
            removed_count += 1
            continue
        # Skip single-character or purely numeric terms
        if len(term) <= 2:
            removed_count += 1
            continue
        filtered_terms.append((term, float(score)))

    # Sort by score descending
    filtered_terms.sort(key=lambda x: x[1], reverse=True)

    print(f"  Removed {removed_count} domain stopwords")
    print(f"  Final vocabulary size: {len(filtered_terms)}")
    print(f"\n  Top 20 extracted terms:")
    for term, score in filtered_terms[:20]:
        print(f"    {score:.4f}  {term}")

    return filtered_terms


# ── Step 3: Embed Terms ──────────────────────────────────────────────────────

def embed_terms(terms: list, model_name: str):
    """
    Encode all extracted terms using the same SentenceTransformer model
    used for the scheme corpus. Returns (model, term_list, embeddings).
    """
    print(f"\n[Step 3] Embedding {len(terms)} terms with {model_name}...")

    model = SentenceTransformer(model_name)
    term_strings = [t[0] for t in terms]

    start = time.time()
    embeddings = model.encode(
        term_strings,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=128,
    )
    elapsed = time.time() - start

    print(f"  Embedded {len(term_strings)} terms in {elapsed:.1f}s")
    print(f"  Embeddings shape: {embeddings.shape}")

    return model, term_strings, embeddings


# ── Step 4: Discover Synonym Clusters ────────────────────────────────────────

def build_synonym_clusters(term_strings: list, embeddings: np.ndarray, threshold: float) -> dict:
    """
    Find all pairs of terms with cosine similarity >= threshold.
    Apply transitive closure to form complete synonym groups.
    If any group exceeds MAX_GROUP_SIZE, re-cluster it with a tighter threshold.
    Returns dict mapping each term to its full synonym group.
    """
    print(f"\n[Step 4] Discovering synonym clusters (threshold={threshold})...")

    # Compute full pairwise cosine similarity matrix
    sim_matrix = embeddings @ embeddings.T

    n = len(term_strings)
    pair_count = 0

    # Union-Find for transitive closure
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Link all pairs above threshold
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                union(i, j)
                pair_count += 1

    print(f"  Found {pair_count} synonym pairs above threshold")

    # Group by root
    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(n):
        root = find(i)
        groups[root].append(i)

    # Filter singletons and handle oversized groups
    multi_groups = {}
    oversized_count = 0

    for root, member_indices in groups.items():
        if len(member_indices) <= 1:
            continue

        if len(member_indices) > MAX_GROUP_SIZE:
            # This group is too big — transitive closure linked unrelated terms.
            # Instead of one mega-group, extract only tight sub-clusters.
            oversized_count += 1
            tighter_threshold = min(threshold + 0.10, 0.90)

            # Re-cluster just this group with a tighter threshold
            sub_parent = {idx: idx for idx in member_indices}

            def sub_find(x):
                while sub_parent[x] != x:
                    sub_parent[x] = sub_parent[sub_parent[x]]
                    x = sub_parent[x]
                return x

            def sub_union(x, y):
                px, py = sub_find(x), sub_find(y)
                if px != py:
                    sub_parent[px] = py

            for ii, idx_a in enumerate(member_indices):
                for idx_b in member_indices[ii + 1:]:
                    if sim_matrix[idx_a, idx_b] >= tighter_threshold:
                        sub_union(idx_a, idx_b)

            sub_groups = defaultdict(list)
            for idx in member_indices:
                sub_groups[sub_find(idx)].append(idx)

            for sub_root, sub_members in sub_groups.items():
                if 1 < len(sub_members) <= MAX_GROUP_SIZE:
                    multi_groups[sub_root] = [term_strings[i] for i in sub_members]
        else:
            multi_groups[root] = [term_strings[i] for i in member_indices]

    # Sort each group alphabetically
    multi_groups = {k: sorted(v) for k, v in multi_groups.items()}

    if oversized_count:
        print(f"  Re-clustered {oversized_count} oversized group(s) with tighter threshold")
    print(f"  Formed {len(multi_groups)} synonym groups (excluding singletons)")

    # Build flat lookup
    synonym_map = {}
    for group_members in multi_groups.values():
        for member in group_members:
            synonym_map[member] = group_members

    # Print discovered groups
    print(f"\n  Discovered synonym groups:")
    for i, (_, members) in enumerate(sorted(multi_groups.items(), key=lambda x: -len(x[1]))):
        if i >= 40:
            remaining = len(multi_groups) - 40
            print(f"    ... and {remaining} more groups")
            break
        print(f"    [{len(members)}] {' | '.join(members)}")

    return synonym_map


# ── Step 5: Save ─────────────────────────────────────────────────────────────

def save_synonym_map(synonym_map: dict, output_path: str):
    """Save synonym map as JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(synonym_map, f, indent=2, ensure_ascii=False)

    # Statistics
    unique_groups = set()
    for members in synonym_map.values():
        unique_groups.add(tuple(sorted(members)))

    print(f"\n[Step 5] Saved synonym map to: {output_path}")
    print(f"  Total entries: {len(synonym_map)}")
    print(f"  Unique synonym groups: {len(unique_groups)}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("SchemeFinder AI — Auto-Synonym Extraction Pipeline")
    print("=" * 70)

    # Load dataset
    data_path = INDEXED_CSV if os.path.exists(INDEXED_CSV) else RAW_DATA
    print(f"\n[Step 1] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} schemes")

    # Step 1: Extract high-signal text
    texts = extract_high_signal_texts(df)

    # Step 2: TF-IDF term extraction
    important_terms = extract_important_terms(texts)

    if len(important_terms) == 0:
        print("\n[ERROR] No important terms extracted! Check dataset columns.")
        exit(1)

    # Step 3: Embed all terms
    model, term_strings, embeddings = embed_terms(important_terms, MODEL_NAME)

    # Step 4: Cluster synonyms
    synonym_map = build_synonym_clusters(term_strings, embeddings, SIM_THRESHOLD)

    if len(synonym_map) == 0:
        print("\n[WARNING] No synonym pairs found! Consider lowering SIM_THRESHOLD.")
        # Save empty map so search_engine doesn't crash
        save_synonym_map({}, SYNONYM_OUTPUT)
    else:
        # Step 5: Save
        save_synonym_map(synonym_map, SYNONYM_OUTPUT)

    print("\n[OK] Synonym extraction complete!")
    print(f"  Run 'python search_engine.py' to test query expansion.")

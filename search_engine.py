"""
search_engine.py — Core search module (Phase 3 + Phase 5 + Phase 6: Query Expansion)

Provides:
  - semantic_search(): expand query → embed → dot product → top-K
  - keyword_search(): word-overlap baseline for comparison
  - expand_query(): inject corpus-mined synonyms before encoding
  - load_index(): load pre-built embeddings + metadata + synonym map at startup

This file is the ONLINE component — it never re-embeds the dataset.
It only loads the pre-built embeddings.npy, schemes_indexed.csv, and synonym_map.json.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re
import json
from collections import Counter
from rank_bm25 import BM25Okapi

# ── Index Loading ─────────────────────────────────────────────────────────────

_BASE_DIR = os.path.dirname(__file__)
_model = None
_embeddings = None
_df = None
_synonym_map = None
_bm25 = None


def load_index():
    """Load embeddings, metadata, synonym map, and BM25 index from disk. Call once at startup."""
    global _model, _embeddings, _df, _synonym_map, _bm25

    embeddings_path = os.path.join(_BASE_DIR, "embeddings.npy")
    csv_path = os.path.join(_BASE_DIR, "schemes_indexed.csv")
    synonym_path = os.path.join(_BASE_DIR, "synonym_map.json")

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"embeddings.npy not found at {embeddings_path}. "
            "Run build_index.py first."
        )

    print("Loading search index...")
    _embeddings = np.load(embeddings_path)
    _df = pd.read_csv(csv_path)
    _model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load synonym map for query expansion (optional — graceful fallback)
    if os.path.exists(synonym_path):
        with open(synonym_path, "r", encoding="utf-8") as f:
            _synonym_map = json.load(f)
        print(f"  Loaded synonym map: {len(_synonym_map)} entries")
    else:
        _synonym_map = {}
        print("  No synonym_map.json found — query expansion disabled")

    # Build BM25 index on 100% full unclipped scheme text (excluding documents and application)
    print("  Building BM25 index on full-row content (name, tags, eligibility, benefits, details)...")
    def _build_full_bm25_text(row):
        parts = [
            str(row.get("scheme_name", "")),
            str(row.get("scheme_name", "")),  # 2x boost for name anchor
            str(row.get("category", row.get("schemeCategory", ""))),
            str(row.get("tags", "")),
            str(row.get("eligibility", "")),   # 100% complete
            str(row.get("benefits", "")),      # 100% complete
            str(row.get("details", row.get("description", "")))  # 100% complete
        ]
        return " ".join(p for p in parts if p and str(p).lower() != "nan")

    tokenized_corpus = [_tokenize(_build_full_bm25_text(row)) for _, row in _df.iterrows()]
    _bm25 = BM25Okapi(tokenized_corpus)
    print("  BM25 full-text index initialized successfully.")

    print(f"  Loaded {len(_df)} schemes, embeddings shape: {_embeddings.shape}")
    return _df


def get_metadata() -> pd.DataFrame:
    """Return the loaded metadata DataFrame."""
    if _df is None:
        load_index()
    return _df


def get_unique_states() -> list:
    """Return sorted list of unique states for filter dropdowns."""
    df = get_metadata()
    # States can be semicolon-separated; expand them
    all_states = set()
    for val in df["state"].dropna().unique():
        for s in val.split(";"):
            s = s.strip()
            if s:
                all_states.add(s)
    return sorted(all_states)


def get_unique_categories() -> list:
    """Return sorted list of unique categories for filter dropdowns."""
    df = get_metadata()
    all_cats = set()
    for val in df["category"].dropna().unique():
        for c in val.split(";"):
            c = c.strip()
            if c:
                all_cats.add(c)
    return sorted(all_cats)


# ── Query Expansion (Phase 6) ────────────────────────────────────────────────

def expand_query(query: str) -> str:
    """
    Expand user query with corpus-mined synonyms before encoding.

    For each word in the query, if it appears in the synonym_map,
    append all its synonyms. This enriches short/colloquial queries
    to produce a denser, more accurate embedding.

    Example:
        Input:  "farmer loan"
        Output: "farmer loan kisan agriculturist cultivator financial assistance"
    """
    if not _synonym_map:
        return query

    query_lower = query.lower()
    tokens = query_lower.split()
    expanded = list(tokens)  # Start with original tokens

    # Check single-word matches
    for token in tokens:
        if token in _synonym_map:
            for syn in _synonym_map[token]:
                if syn.lower() not in [t.lower() for t in expanded]:
                    expanded.append(syn)

    # Check bigram matches (e.g., "financial assistance")
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if bigram in _synonym_map:
            for syn in _synonym_map[bigram]:
                if syn.lower() not in [t.lower() for t in expanded]:
                    expanded.append(syn)

    expanded_query = " ".join(expanded)

    if expanded_query != query_lower:
        print(f"  [Query Expansion] '{query}' → '{expanded_query}'")

    return expanded_query


# ── Query Intent & Entity Extraction (Phase 7: Zero-Click Filtering) ─────────

STATE_PATTERNS = {
    # Acronyms & Short forms (word-bounded)
    r"\b(up|u\.p\.)\b": "Uttar Pradesh",
    r"\b(mp|m\.p\.)\b": "Madhya Pradesh",
    r"\b(ap|a\.p\.)\b": "Andhra Pradesh",
    r"\b(tn|t\.n\.)\b": "Tamil Nadu",
    r"\b(wb|w\.b\.)\b": "West Bengal",
    r"\b(hp|h\.p\.)\b": "Himachal Pradesh",
    r"\b(j&k|jk|j and k)\b": "Jammu and Kashmir",
    r"\bdl\b": "Delhi",
    r"\buk\b": "Uttarakhand",
    # Aliases
    r"\bhimachal\b": "Himachal Pradesh",
    r"\borissa\b": "Odisha",
    r"\bpondicherry\b": "Puducherry",
    r"\btamilnadu\b": "Tamil Nadu",
    r"\bbengal\b": "West Bengal",
    # Full Names
    r"\bandhra pradesh\b": "Andhra Pradesh",
    r"\barunachal pradesh\b": "Arunachal Pradesh",
    r"\bassam\b": "Assam",
    r"\bbihar\b": "Bihar",
    r"\bchandigarh\b": "Chandigarh",
    r"\bchhattisgarh\b": "Chhattisgarh",
    r"\bdelhi\b": "Delhi",
    r"\bgoa\b": "Goa",
    r"\bgujarat\b": "Gujarat",
    r"\bharyana\b": "Haryana",
    r"\bjammu and kashmir\b": "Jammu and Kashmir",
    r"\bjharkhand\b": "Jharkhand",
    r"\bkarnataka\b": "Karnataka",
    r"\bkerala\b": "Kerala",
    r"\bladakh\b": "Ladakh",
    r"\blakshadweep\b": "Lakshadweep",
    r"\bmadhya pradesh\b": "Madhya Pradesh",
    r"\bmaharashtra\b": "Maharashtra",
    r"\bmanipur\b": "Manipur",
    r"\bmeghalaya\b": "Meghalaya",
    r"\bmizoram\b": "Mizoram",
    r"\bnagaland\b": "Nagaland",
    r"\bodisha\b": "Odisha",
    r"\bpuducherry\b": "Puducherry",
    r"\bpunjab\b": "Punjab",
    r"\brajasthan\b": "Rajasthan",
    r"\bsikkim\b": "Sikkim",
    r"\btamil nadu\b": "Tamil Nadu",
    r"\btelangana\b": "Telangana",
    r"\btripura\b": "Tripura",
    r"\buttar pradesh\b": "Uttar Pradesh",
    r"\buttarakhand\b": "Uttarakhand",
    r"\bwest bengal\b": "West Bengal",
}

CATEGORY_KEYWORDS = {
    "Agriculture, Rural & Environment": [
        "farmer", "farmers", "kisan", "crop", "crops", "tractor", "seed", "seeds",
        "fertilizer", "fertilizers", "agriculture", "agricultural", "farming",
        "cultivator", "cultivation", "harvest", "irrigation", "dairy", "livestock",
        "pashupalan", "soil", "krishi", "pisciculture", "fisheries"
    ],
    "Education & Learning": [
        "scholarship", "scholarships", "student", "students", "college", "school",
        "schools", "colleges", "tuition", "matric", "post-matric", "pre-matric",
        "degree", "diploma", "phd", "hostel", "fellowship", "fellowships",
        "education", "educational", "internship", "vidyarthi", "shiksha"
    ],
    "Health & Wellness": [
        "hospital", "hospitals", "medical", "treatment", "cancer", "disease",
        "diseases", "illness", "doctor", "surgery", "medicine", "medicines",
        "ayushman", "healthcare", "health insurance", "patient", "patients",
        "clinic", "hospitalization"
    ],
    "Housing & Shelter": [
        "house", "housing", "pucca house", "awas", "pmay", "shelter",
        "flat", "flats", "home loan subsidy", "grih nirman", "plot", "homeless"
    ],
    "Women and Child": [
        "widow", "widows", "vidhwa", "girl child", "daughter", "daughters",
        "maternity", "pregnant", "pregnancy", "infant", "women", "woman",
        "mahila", "kanya", "lactating"
    ],
    "Business & Entrepreneurship": [
        "startup", "startups", "msme", "msmes", "shop", "shopkeeper",
        "vendor", "street vendor", "business loan", "udyam", "entrepreneur",
        "entrepreneurs", "micro enterprise", "small business"
    ],
    "Skills & Employment": [
        "skill", "skills", "skill development", "iti", "vocational",
        "vocational training", "pmkvy", "training", "unemployed", "unemployment",
        "job training", "apprentice", "apprenticeship", "rojgar"
    ],
    "Social welfare & Empowerment": [
        "disabled", "disabilities", "disability", "divyang", "handicapped", "pwd",
        "pension", "pensions", "vridha", "senior citizen", "senior citizens",
        "old age", "sc", "st", "scheduled caste", "scheduled tribe", "dalit", "adivasi"
    ],
    "Banking, Financial Services and Insurance": [
        "bank loan", "credit card", "kcc", "insurance cover", "micro finance",
        "life insurance", "accidental insurance", "financial assistance"
    ],
    "Utility & Sanitation": [
        "toilet", "toilets", "sanitation", "swachh", "shauchalay", "solar panel",
        "solar rooftop", "electricity connection", "lpg", "gas connection",
        "ujjwala", "tap water", "jal jeevan", "drinking water"
    ]
}


def parse_query_intent(query: str) -> dict:
    """
    Extract state entities and primary category intent from raw natural language queries.
    Returns:
        {
            "detected_state": str or None,
            "detected_category": str or None,
            "detected_tokens": list of matched trigger terms
        }
    """
    if not query:
        return {"detected_state": None, "detected_category": None, "detected_tokens": []}

    q_lower = query.lower()
    detected_state = None
    detected_category = None
    detected_tokens = []

    # 1. State Extraction via regex patterns
    for pattern, state_name in STATE_PATTERNS.items():
        match = re.search(pattern, q_lower, re.IGNORECASE)
        if match:
            detected_state = state_name
            detected_tokens.append(match.group(0))
            break

    # 2. Category Extraction (Score by keyword density)
    best_cat = None
    max_matches = 0

    for cat_name, kw_list in CATEGORY_KEYWORDS.items():
        matches = 0
        for kw in kw_list:
            if re.search(r"\b" + re.escape(kw) + r"\b", q_lower, re.IGNORECASE):
                matches += 1
                detected_tokens.append(kw)
        if matches > max_matches:
            max_matches = matches
            best_cat = cat_name

    if max_matches > 0:
        detected_category = best_cat

    return {
        "detected_state": detected_state,
        "detected_category": detected_category,
        "detected_tokens": list(set(detected_tokens)),
    }


# ── Semantic Search (Phase 3 + Phase 6 + Phase 7) ───────────────────────────

def semantic_search(query: str, k: int = 5, state: str = None, category: str = None, auto_filter: bool = True) -> pd.DataFrame:
    """
    Embed the query, compute cosine similarity against all schemes,
    apply optional or auto-detected hard filters, return top-K results with scores.

    Args:
        query: plain-language description of the user's situation
        k: number of results to return
        state: if set, only return schemes available in this state
        category: if set, only return schemes in this category
        auto_filter: if True and state/category are None, infer them from query

    Returns:
        DataFrame with top-K rows + a 'score' column, sorted descending.
    """
    if _model is None or _embeddings is None:
        load_index()

    intent = parse_query_intent(query) if auto_filter else {}
    if auto_filter:
        if not state and intent.get("detected_state"):
            state = intent["detected_state"]
            print(f"  [Semantic Search Auto-Filter] Applied State: '{state}'")
        if not category and intent.get("detected_category"):
            category = intent["detected_category"]
            print(f"  [Semantic Search Auto-Filter] Applied Category: '{category}'")

    # Phase 6: Expand query with corpus-mined synonyms before encoding
    expanded = expand_query(query)

    # Encode expanded query with same model → 384-dim normalized vector
    query_vec = _model.encode([expanded], normalize_embeddings=True)[0]

    # Dot product = cosine similarity (vectors are unit-length)
    scores = _embeddings @ query_vec  # shape: (num_schemes,)

    # Apply hard filters by zeroing out excluded rows
    if state:
        # A scheme matches if the user's state appears in its semicolon-separated state list,
        # OR if the scheme is marked "All" (nationwide)
        state_mask = _df["state"].fillna("").apply(
            lambda s: state in [x.strip() for x in s.split(";")] or "All" in [x.strip() for x in s.split(";")]
        ).values
        scores = np.where(state_mask, scores, -1.0)

    if category:
        cat_mask = _df["category"].fillna("").apply(
            lambda c: category in [x.strip() for x in c.split(";")]
        ).values
        scores = np.where(cat_mask, scores, -1.0)

    # Filter out excluded rows (-1.0)
    valid_mask = scores > -0.99
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        empty = _df.head(0).copy()
        empty["score"] = pd.Series(dtype=float)
        return empty

    # Top-K by descending score among valid candidates
    sorted_order = np.argsort(-scores[valid_indices])[:k]
    top_indices = valid_indices[sorted_order]
    results = _df.iloc[top_indices].copy()
    results["score"] = scores[top_indices]

    return results


# ── Keyword Search (Phase 5 — baseline) ──────────────────────────────────────

def _tokenize(text: str) -> list:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r'[a-z0-9]+', text.lower())


def keyword_search(query: str, k: int = 5, state: str = None, category: str = None) -> pd.DataFrame:
    """
    Naive keyword-overlap search: count how many query words appear in
    each scheme's search_text. Normalized by query length.

    This is the baseline that semantic search should beat.
    """
    if _df is None:
        load_index()

    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return _df.head(0)

    def overlap_score(text):
        doc_tokens = set(_tokenize(str(text)))
        return len(query_tokens & doc_tokens) / len(query_tokens)

    scores = _df["search_text"].apply(overlap_score).values

    # Apply same hard filters
    if state:
        state_mask = _df["state"].fillna("").apply(
            lambda s: state in [x.strip() for x in s.split(";")] or "All" in [x.strip() for x in s.split(";")]
        ).values
        scores = np.where(state_mask, scores, -1.0)

    if category:
        cat_mask = _df["category"].fillna("").apply(
            lambda c: category in [x.strip() for x in c.split(";")]
        ).values
        scores = np.where(cat_mask, scores, -1.0)

    # Filter out excluded rows (-1.0) and non-matching documents (0.0)
    valid_mask = scores > 0.0
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        empty = _df.head(0).copy()
        empty["score"] = pd.Series(dtype=float)
        return empty

    sorted_order = np.argsort(-scores[valid_indices])[:k]
    top_indices = valid_indices[sorted_order]
    results = _df.iloc[top_indices].copy()
    results["score"] = scores[top_indices]

    return results


# ── BM25 Lexical Search (Okapi BM25) ─────────────────────────────────────────

def bm25_search(query: str, k: int = 5, state: str = None, category: str = None, auto_filter: bool = True) -> pd.DataFrame:
    """
    Standard Okapi BM25 search with term frequency saturation and document length normalization.
    """
    if _df is None or _bm25 is None:
        load_index()

    intent = parse_query_intent(query) if auto_filter else {}
    if auto_filter:
        if not state and intent.get("detected_state"):
            state = intent["detected_state"]
        if not category and intent.get("detected_category"):
            category = intent["detected_category"]

    query_tokens = _tokenize(query)
    if not query_tokens:
        empty = _df.head(0).copy()
        empty["score"] = pd.Series(dtype=float)
        return empty

    scores = np.array(_bm25.get_scores(query_tokens))

    # Apply hard filters
    if state:
        state_mask = _df["state"].fillna("").apply(
            lambda s: state in [x.strip() for x in s.split(";")] or "All" in [x.strip() for x in s.split(";")]
        ).values
        scores = np.where(state_mask, scores, -1.0)

    if category:
        cat_mask = _df["category"].fillna("").apply(
            lambda c: category in [x.strip() for x in c.split(";")]
        ).values
        scores = np.where(cat_mask, scores, -1.0)

    # Filter out excluded rows (-1.0) and non-matching documents (<= 0.0)
    valid_mask = scores > 0.0
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        empty = _df.head(0).copy()
        empty["score"] = pd.Series(dtype=float)
        return empty

    sorted_order = np.argsort(-scores[valid_indices])[:k]
    top_indices = valid_indices[sorted_order]
    results = _df.iloc[top_indices].copy()
    results["score"] = scores[top_indices]

    return results


# ── Hybrid Search (Semantic + BM25 Reciprocal Rank Fusion) ───────────────────

def hybrid_search(
    query: str,
    k: int = 5,
    state: str = None,
    category: str = None,
    sem_weight: float = 0.60,
    bm25_weight: float = 0.40,
    rrf_k: int = 60,
    auto_filter: bool = True,
) -> pd.DataFrame:
    """
    Weighted Reciprocal Rank Fusion (RRF) combining:
      1. Dense Semantic vector similarity (with query expansion)
      2. Lexical BM25 ranking (term saturation & length normalization)
      3. Automatic query intent & state entity extraction

    Formula:
      RRF(d) = sem_weight / (rrf_k + rank_sem(d)) + bm25_weight / (rrf_k + rank_bm25(d))

    Why RRF over linear score normalization:
      - Raw BM25 scores are unbounded and skewed by rare terms.
      - Cosine similarities have a tight bell distribution.
      - RRF fuses ranking orders directly, preventing outliers from dominating.
    """
    if _model is None or _embeddings is None or _bm25 is None:
        load_index()

    # Phase 7: Auto-infer state and category if not manually passed
    intent = parse_query_intent(query) if auto_filter else {}
    if auto_filter:
        if not state and intent.get("detected_state"):
            state = intent["detected_state"]
            print(f"  [Hybrid Search Auto-Filter] Applied State: '{state}'")
        if not category and intent.get("detected_category"):
            category = intent["detected_category"]
            print(f"  [Hybrid Search Auto-Filter] Applied Category: '{category}'")

    # 1. Semantic Scores (Cosine similarity)
    expanded = expand_query(query)
    query_vec = _model.encode([expanded], normalize_embeddings=True)[0]
    sem_scores = _embeddings @ query_vec  # shape: (num_schemes,)

    # 2. BM25 Scores
    query_tokens = _tokenize(query)
    if query_tokens:
        bm25_scores = np.array(_bm25.get_scores(query_tokens))
    else:
        bm25_scores = np.zeros(len(_df))

    # 3. Hard Filters (State & Category)
    if state:
        state_mask = _df["state"].fillna("").apply(
            lambda s: state in [x.strip() for x in s.split(";")] or "All" in [x.strip() for x in s.split(";")]
        ).values
        sem_scores = np.where(state_mask, sem_scores, -1.0)
        bm25_scores = np.where(state_mask, bm25_scores, -1.0)

    if category:
        cat_mask = _df["category"].fillna("").apply(
            lambda c: category in [x.strip() for x in c.split(";")]
        ).values
        sem_scores = np.where(cat_mask, sem_scores, -1.0)
        bm25_scores = np.where(cat_mask, bm25_scores, -1.0)

    # 4. Valid candidates mask
    valid_mask = (sem_scores > -0.99) & (bm25_scores > -0.99)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        empty = _df.head(0).copy()
        empty["score"] = pd.Series(dtype=float)
        return empty

    valid_sem = sem_scores[valid_indices]
    valid_bm25 = bm25_scores[valid_indices]

    # Calculate 0-indexed ranks (0 is highest score)
    sem_ranks = np.argsort(np.argsort(-valid_sem))
    bm25_ranks = np.argsort(np.argsort(-valid_bm25))

    # Reciprocal rank computation (1-indexed ranks: rank + 1)
    sem_rrf = sem_weight / (rrf_k + 1 + sem_ranks)

    # BM25 contribution is zero if the document has 0 BM25 score (no matching tokens)
    bm25_rrf = np.where(
        valid_bm25 > 0.0,
        bm25_weight / (rrf_k + 1 + bm25_ranks),
        0.0,
    )

    combined_rrf = sem_rrf + bm25_rrf

    # Sort candidates by combined RRF score descending
    sorted_order = np.argsort(-combined_rrf)[:k]
    top_indices = valid_indices[sorted_order]

    results = _df.iloc[top_indices].copy()
    results["score"] = combined_rrf[sorted_order]
    results["semantic_score"] = valid_sem[sorted_order]
    results["bm25_score"] = valid_bm25[sorted_order]

    return results


# ── Quick CLI test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_index()

    test_queries = [
        "farmer tractor loan in UP",
        "scholarship for disabled college students in Maharashtra",
        "free housing scheme for BPL in Bihar",
        "laptop assistance for students in Gujarat",
        "widow pension in Delhi",
        "kisan credit card",
    ]

    for q in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {q}")
        intent = parse_query_intent(q)
        print(f"  [Intent Detected] State: {intent['detected_state']} | Category: {intent['detected_category']} | Tokens: {intent['detected_tokens']}")

        print(f"\n--- Hybrid Search (Semantic + BM25 RRF + Zero-Click Filter) ---")
        h_results = hybrid_search(q, k=3)
        for _, row in h_results.iterrows():
            print(f"  [RRF: {row['score']:.4f} | State: {row.get('state', '')} | Sem: {row['semantic_score']:.3f} | BM25: {row['bm25_score']:.1f}] {row['scheme_name']}")

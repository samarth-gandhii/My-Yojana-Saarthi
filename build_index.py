"""
build_index.py — Offline indexing script (Phase 1 + Phase 2)

Run once to:
  1. Load updated_data.csv (or specified dataset), normalize schema
  2. Synthesize rich search_text (Name, Details, Benefits, Eligibility, Category, Tags)
  3. Embed all search_text with all-MiniLM-L6-v2 (384-dimensional normalized vectors)
  4. Save embeddings.npy + schemes_indexed.csv

Re-run only when the dataset changes.
"""

import os
import re
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

_BASE_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(_BASE_DIR, "updated_data.csv")
LEGACY_DATA_FILE = os.path.join(_BASE_DIR, "govt_schemes_dataset.csv")
EMBEDDINGS_FILE = os.path.join(_BASE_DIR, "embeddings.npy")
INDEXED_CSV = os.path.join(_BASE_DIR, "schemes_indexed.csv")
MODEL_NAME = "all-MiniLM-L6-v2"

# ── 15 Standard MyScheme Categories ──────────────────────────────────────────

KNOWN_CATS = [
    "Agriculture,Rural & Environment", "Agriculture, Rural & Environment",
    "Banking,Financial Services and Insurance", "Banking, Financial Services and Insurance",
    "Business & Entrepreneurship",
    "Education & Learning",
    "Health & Wellness",
    "Housing & Shelter",
    "Public Safety,Law & Justice", "Public Safety, Law & Justice",
    "Science, IT & Communications",
    "Skills & Employment",
    "Social welfare & Empowerment",
    "Sports & Culture",
    "Transport & Infrastructure",
    "Travel & Tourism",
    "Utility & Sanitation",
    "Women and Child",
]

CAT_STD_MAP = {
    c: c.replace("Agriculture,Rural & Environment", "Agriculture, Rural & Environment")
        .replace("Banking,Financial Services and Insurance", "Banking, Financial Services and Insurance")
        .replace("Public Safety,Law & Justice", "Public Safety, Law & Justice")
    for c in KNOWN_CATS
}


def parse_and_standardize_categories(val: str) -> str:
    """Standardize category strings into sorted, semicolon-separated official categories."""
    if not isinstance(val, str) or not val.strip():
        return "General Welfare"
    found = []
    # Match longest first to avoid partial conflicts
    for kc in sorted(KNOWN_CATS, key=len, reverse=True):
        if kc in val:
            norm = CAT_STD_MAP[kc]
            if norm not in found:
                found.append(norm)
            val = val.replace(kc, "")
    return ";".join(found) if found else "General Welfare"


# ── State and Ministry Resolution ─────────────────────────────────────────────

def resolve_states_and_ministries(df: pd.DataFrame, base_dir: str):
    """
    Resolve accurate state and nodal ministry for all schemes:
      1. Cross-reference legacy dataset by slug/scheme_id when available
      2. Central schemes default to state 'All'
      3. State schemes match known Indian states & UTs via regex
    """
    old_states = {}
    old_ministries = {}
    legacy_path = os.path.join(base_dir, "govt_schemes_dataset.csv")
    if os.path.exists(legacy_path):
        try:
            df_old = pd.read_csv(legacy_path)
            if "scheme_id" in df_old.columns:
                if "state" in df_old.columns:
                    old_states = df_old.set_index("scheme_id")["state"].to_dict()
                if "ministry" in df_old.columns:
                    old_ministries = df_old.set_index("scheme_id")["ministry"].to_dict()
        except Exception as e:
            print(f"  Note: Legacy cross-reference skipped ({e})")

    all_indian_states = [
        "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam",
        "Bihar", "Chandigarh", "Chhattisgarh", "Dadra & Nagar Haveli and Daman & Diu",
        "Delhi", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir",
        "Jharkhand", "Karnataka", "Kerala", "Ladakh", "Lakshadweep", "Madhya Pradesh",
        "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha",
        "Puducherry", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
        "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
    ]

    state_aliases = {
        "H.P.": "Himachal Pradesh",
        "Himachal": "Himachal Pradesh",
        "JK": "Jammu and Kashmir",
        "J&K": "Jammu and Kashmir",
        "YSR": "Andhra Pradesh",
        "Ganga Kalyana": "Karnataka",
        "Puducherry": "Puducherry",
        "Pondicherry": "Puducherry",
        "Tamil Nadu": "Tamil Nadu",
        "Tamilnadu": "Tamil Nadu",
        "Delhi": "Delhi",
        "Odisha": "Odisha",
        "Orissa": "Odisha",
        "Uttarakhand": "Uttarakhand",
        "Uttaranchal": "Uttarakhand",
        "Uttar Pradesh": "Uttar Pradesh",
        "UP": "Uttar Pradesh",
        "West Bengal": "West Bengal",
        "WB": "West Bengal",
        "Madhya Pradesh": "Madhya Pradesh",
        "MP": "Madhya Pradesh",
        "Andhra": "Andhra Pradesh",
        "AP": "Andhra Pradesh",
        "Telangana": "Telangana",
        "Maharashtra": "Maharashtra",
        "Gujarat": "Gujarat",
        "Rajasthan": "Rajasthan",
        "Kerala": "Kerala",
        "Karnataka": "Karnataka",
        "Goa": "Goa",
        "Bihar": "Bihar",
        "Assam": "Assam",
        "Punjab": "Punjab",
        "Haryana": "Haryana",
        "Jharkhand": "Jharkhand",
        "Chhattisgarh": "Chhattisgarh",
        "Sikkim": "Sikkim",
        "Meghalaya": "Meghalaya",
        "Mizoram": "Mizoram",
        "Manipur": "Manipur",
        "Nagaland": "Nagaland",
        "Tripura": "Tripura",
        "Arunachal": "Arunachal Pradesh",
        "Ladakh": "Ladakh",
        "Chandigarh": "Chandigarh",
    }

    resolved_states = []
    resolved_mins = []

    for _, row in df.iterrows():
        slug = str(row.get("slug", row.get("scheme_id", ""))).strip()
        is_central = str(row.get("level", "")).strip().lower() == "central"

        # 1. State resolution
        st_val = None
        if slug in old_states and pd.notna(old_states[slug]):
            st_val = str(old_states[slug]).strip()
        elif is_central:
            st_val = "All"
        else:
            text = f"{row.get('scheme_name', '')} {row.get('details', '')} {row.get('eligibility', '')}"
            for alias, target in state_aliases.items():
                if re.search(r"\b" + re.escape(alias) + r"\b", text, re.IGNORECASE):
                    st_val = target
                    break
            if not st_val:
                for target in all_indian_states:
                    if re.search(r"\b" + re.escape(target) + r"\b", text, re.IGNORECASE):
                        st_val = target
                        break

        resolved_states.append(st_val or "All")

        # 2. Ministry resolution
        min_val = None
        if slug in old_ministries and pd.notna(old_ministries[slug]):
            min_val = str(old_ministries[slug]).strip()
        elif is_central:
            min_val = "Ministry / Central Government of India"
        else:
            min_val = "Not specified (state scheme)"

        resolved_mins.append(min_val)

    return resolved_states, resolved_mins


# ── Enriched Search Text Generation (Field-Weighted & Token Budgeted) ─────────

def _clean_snippet(val: str, max_words: int) -> str:
    """Helper to clean and limit text to a word budget."""
    if not isinstance(val, str) or not val.strip() or val.strip().lower() == "nan":
        return ""
    words = val.strip().split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return " ".join(words)


def build_search_text(row: pd.Series) -> str:
    """
    Synthesize field-weighted, token-budgeted search text.
    Strictly stays within the 256-token sequence limit of SentenceTransformer
    (~150-160 words max) while ensuring all high-salience fields are represented.

    Field allocation (approx 150 words):
      1. Scheme Name (repeated 2x as vector anchor): ~15 words
      2. Category: ~5 words
      3. Tags (repeated 2x): ~20 words
      4. Eligibility Criteria (prioritized before details): ~45 words
      5. Key Benefits: ~35 words
      6. Details / Overview: ~35 words
    """
    parts = []

    # 1. Scheme Name (anchor: repeated 2x)
    scheme_name = str(row.get("scheme_name", "")).strip()
    if scheme_name and scheme_name.lower() != "nan":
        parts.append(f"{scheme_name}. {scheme_name}")

    # 2. Category
    category = str(row.get("category", row.get("schemeCategory", ""))).strip()
    if category and category.lower() != "nan":
        parts.append(f"Category: {category}")

    # 3. Tags (clean semicolons to commas, repeated 2x for keyword density)
    tags_raw = str(row.get("tags", "")).strip()
    if tags_raw and tags_raw.lower() != "nan":
        tags_clean = ", ".join(t.strip() for t in tags_raw.split(";") if t.strip())
        if tags_clean:
            parts.append(f"Tags: {tags_clean}. {tags_clean}")

    # 4. Eligibility (Demographic conditions: income, caste, age, gender - capped at 45 words)
    eligibility_val = str(row.get("eligibility", ""))
    elig_snippet = _clean_snippet(eligibility_val, max_words=45)
    if elig_snippet:
        parts.append(f"Eligibility: {elig_snippet}")

    # 5. Benefits (Financial outcome, subsidies, laptops, etc. - capped at 35 words)
    benefits_val = str(row.get("benefits", ""))
    benefits_snippet = _clean_snippet(benefits_val, max_words=35)
    if benefits_snippet:
        parts.append(f"Benefits: {benefits_snippet}")

    # 6. Details / Overview (capped at 35 words to prevent drowning out metadata)
    details_val = str(row.get("details", row.get("description", "")))
    details_snippet = _clean_snippet(details_val, max_words=35)
    if details_snippet:
        parts.append(f"Overview: {details_snippet}")

    text = ". ".join(p.rstrip(".") for p in parts if p) + "."
    return text


# ── Phase 1: Data Pipeline ────────────────────────────────────────────────────

def load_and_validate(path: str) -> pd.DataFrame:
    """Load dataset, normalize schema, synthesize search_text, and validate."""
    if not os.path.exists(path):
        # Fall back to legacy if updated_data.csv not found
        if os.path.exists(LEGACY_DATA_FILE):
            print(f"Warning: {path} not found. Falling back to {LEGACY_DATA_FILE}")
            path = LEGACY_DATA_FILE
        else:
            raise FileNotFoundError(f"Neither {path} nor {LEGACY_DATA_FILE} found.")

    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"  Raw rows loaded: {len(df)}")
    print(f"  Raw columns: {list(df.columns)}")

    # Drop empty unnamed columns if present (e.g. Unnamed: 9)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]

    # Map slug -> scheme_id
    if "slug" in df.columns and "scheme_id" not in df.columns:
        df["scheme_id"] = df["slug"]
    elif "scheme_id" in df.columns and "slug" not in df.columns:
        df["slug"] = df["scheme_id"]

    # Map details -> description
    if "details" in df.columns and "description" not in df.columns:
        df["description"] = df["details"]
    elif "description" in df.columns and "details" not in df.columns:
        df["details"] = df["description"]

    # Standardize category
    if "schemeCategory" in df.columns and "category" not in df.columns:
        print("  Standardizing categories from 'schemeCategory'...")
        df["category"] = df["schemeCategory"].apply(parse_and_standardize_categories)
    elif "category" in df.columns:
        df["category"] = df["category"].apply(parse_and_standardize_categories)
    else:
        df["category"] = "General Welfare"

    # Resolve state and ministry
    if "state" not in df.columns or "ministry" not in df.columns:
        print("  Resolving state and ministry assignments...")
        states, ministries = resolve_states_and_ministries(df, _BASE_DIR)
        if "state" not in df.columns:
            df["state"] = states
        if "ministry" not in df.columns:
            df["ministry"] = ministries

    # Ensure level exists
    if "level" not in df.columns:
        df["level"] = "Central"

    # Synthesize search_text using field-weighted token budgeting
    print("  Generating field-weighted, token-budgeted search_text (<=256 tokens)...")
    df["search_text"] = df.apply(build_search_text, axis=1)

    # Validate required columns
    required = ["scheme_name", "scheme_id", "search_text", "state", "category", "level"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    print(f"\n  Final dataset summary:")
    print(f"  - Total schemes: {len(df)}")
    print(f"  - Unique states/territories: {len(set(s for val in df['state'].dropna() for s in str(val).split(';')))}")
    print(f"  - Unique categories: {len(set(c for val in df['category'].dropna() for c in str(val).split(';')))}")
    print(f"\n  Sample search_text (row 0):")
    print(f"  {df['search_text'].iloc[0][:220]}...")

    return df


# ── Phase 2: Indexing ─────────────────────────────────────────────────────────

def build_embeddings(df: pd.DataFrame, model_name: str) -> np.ndarray:
    """Encode all search_text values and return normalized embeddings."""
    print(f"\nLoading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = df["search_text"].tolist()
    print(f"Encoding {len(texts)} scheme texts with batch size 64...")

    start = time.time()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,  # Unit-length vectors -> dot product = cosine similarity
        show_progress_bar=True,
        batch_size=64,
    )
    elapsed = time.time() - start

    print(f"  Done in {elapsed:.1f}s ({len(texts)/elapsed:.1f} schemes/sec)")
    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings


def save_index(df: pd.DataFrame, embeddings: np.ndarray):
    """Save embeddings and aligned metadata to disk."""
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"\nSaved embeddings to: {EMBEDDINGS_FILE}")

    df.to_csv(INDEXED_CSV, index=False)
    print(f"Saved indexed metadata to: {INDEXED_CSV}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("SchemeFinder AI — Index Builder (Data Pipeline & Neural Encoding)")
    print("=" * 70)

    # Phase 1: Load and normalize
    df = load_and_validate(DATA_FILE)

    # Phase 2: Encode
    embeddings = build_embeddings(df, MODEL_NAME)

    # Verify shape
    assert embeddings.shape == (len(df), 384), (
        f"Unexpected shape: {embeddings.shape}, expected ({len(df)}, 384)"
    )

    # Save
    save_index(df, embeddings)

    print("\n[OK] Index built successfully from updated_data.csv!")
    print(f"  {embeddings.shape[0]} schemes × {embeddings.shape[1]} dimensions")
    print(f"  embeddings.npy: {os.path.getsize(EMBEDDINGS_FILE) / 1e6:.1f} MB")
    print(f"  schemes_indexed.csv: {os.path.getsize(INDEXED_CSV) / 1e6:.1f} MB")


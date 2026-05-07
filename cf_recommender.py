"""
Collaborative-filtering recommender backed by the Food.com Kaggle dataset.

Dataset: shuyangli94/food-com-recipes-and-user-interactions
  RAW_recipes.csv      – recipe metadata (id, name, ingredients, …)
  RAW_interactions.csv – user ratings   (user_id, recipe_id, rating)

How it works
------------
1. Build a user × recipe sparse rating matrix from the interactions file.
2. Apply TruncatedSVD (50 latent factors) on the transposed matrix so each
   recipe gets a dense latent-factor vector.
3. At recommendation time, match the user's rated Spoonacular recipe titles
   against Food.com names via TF-IDF character-ngram similarity.
4. Average the latent vectors of the matched recipes (weighted by star rating)
   into a single "taste profile" vector and return the nearest neighbours.
"""

import ast
import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── paths ──────────────────────────────────────────────────────────────────
_DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
_MODEL_PATH = os.path.join(_DATA_DIR, "cf_model.pkl")

# ── filtering thresholds (trade-off: quality vs. build time) ──────────────
_MIN_RECIPE_RATINGS = 15   # drop rarely-rated recipes
_MIN_USER_RATINGS   = 8    # drop barely-active users
_N_SVD_COMPONENTS   = 50
_NAME_TFIDF_FEATS   = 30_000


# ── helpers ────────────────────────────────────────────────────────────────

def _parse_ingredients(raw: str) -> str:
    try:
        return ", ".join(ast.literal_eval(raw))
    except Exception:
        return str(raw)


def _find_dataset_path() -> str | None:
    """Return path to the Food.com dataset folder, or None if not found."""
    # 1. Explicit env override
    env = os.getenv("CF_DATASET_PATH", "").strip()
    if env and os.path.isdir(env):
        return env

    # 2. kagglehub cache (~/.cache/kagglehub/…)
    base = os.path.expanduser(
        "~/.cache/kagglehub/datasets/"
        "shuyangli94/food-com-recipes-and-user-interactions"
    )
    if os.path.isdir(base):
        # look inside versions/X/ subdirectories
        for sub in ("versions", ""):
            scan_dir = os.path.join(base, sub) if sub else base
            if not os.path.isdir(scan_dir):
                continue
            candidates = sorted(
                [d for d in os.listdir(scan_dir)
                 if os.path.isdir(os.path.join(scan_dir, d))],
                reverse=True,
            )
            for ver in candidates:
                candidate = os.path.join(scan_dir, ver)
                if os.path.exists(os.path.join(candidate, "RAW_interactions.csv")):
                    return candidate
            # files directly in base
            if os.path.exists(os.path.join(base, "RAW_interactions.csv")):
                return base

    return None


# ── model building ─────────────────────────────────────────────────────────

def _build_model(dataset_path: str) -> dict:
    """Read CSVs, build SVD latent factors, pickle result, return model dict."""
    inter = pd.read_csv(
        os.path.join(dataset_path, "RAW_interactions.csv"),
        usecols=["user_id", "recipe_id", "rating"],
    )
    recipes = pd.read_csv(
        os.path.join(dataset_path, "RAW_recipes.csv"),
        usecols=["id", "name", "ingredients"],
    )
    recipes["ingredients"] = recipes["ingredients"].apply(_parse_ingredients)

    # Drop sparse recipes / users so the SVD stays tractable
    recipe_counts = inter["recipe_id"].value_counts()
    user_counts   = inter["user_id"].value_counts()
    inter = inter[
        inter["recipe_id"].isin(recipe_counts[recipe_counts >= _MIN_RECIPE_RATINGS].index) &
        inter["user_id"].isin(user_counts[user_counts >= _MIN_USER_RATINGS].index)
    ]

    item_ids  = inter["recipe_id"].unique()
    user_ids  = inter["user_id"].unique()
    item_idx  = {rid: i for i, rid in enumerate(item_ids)}
    user_idx  = {uid: i for i, uid in enumerate(user_ids)}

    rows = inter["user_id"].map(user_idx).values
    cols = inter["recipe_id"].map(item_idx).values
    vals = inter["rating"].astype(np.float32).values

    # user × item sparse matrix
    matrix = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_ids)))

    # TruncatedSVD on item × user (items as rows) → (n_items, n_components)
    svd          = TruncatedSVD(n_components=_N_SVD_COMPONENTS, random_state=42)
    item_factors = svd.fit_transform(matrix.T).astype(np.float32)

    # Recipe metadata for the items we kept
    recipe_meta = (
        recipes[recipes["id"].isin(item_ids)]
        .drop_duplicates("id")
        .set_index("id")[["name", "ingredients"]]
    )

    # TF-IDF on recipe names (char 3-grams) for fuzzy title matching
    name_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 3), max_features=_NAME_TFIDF_FEATS
    )
    name_mat = name_vec.fit_transform(recipe_meta["name"].str.lower())

    model = {
        "item_idx":     item_idx,
        "idx_item":     {v: k for k, v in item_idx.items()},
        "item_factors": item_factors,
        "recipe_meta":  recipe_meta,
        "name_vec":     name_vec,
        "name_mat":     name_mat,
    }

    with open(_MODEL_PATH, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    return model


# ── public API ─────────────────────────────────────────────────────────────

def model_pickle_exists() -> bool:
    return os.path.exists(_MODEL_PATH)


def load_cf_model() -> dict | None:
    """
    Load the cached model from disk, or build it from the Kaggle dataset.
    Returns None if the dataset cannot be found.
    """
    if os.path.exists(_MODEL_PATH):
        with open(_MODEL_PATH, "rb") as f:
            return pickle.load(f)

    dataset_path = _find_dataset_path()
    if dataset_path is None:
        return None

    return _build_model(dataset_path)


def get_cf_recommendations(
    rated_recipes: dict,   # {spoonacular_title: star_rating (1-5)}
    model: dict,
    n: int = 6,
) -> list[dict]:
    """
    Return up to *n* Food.com recipe dicts with keys 'name' and 'ingredients'.

    Algorithm
    ---------
    For every rated title, find the best-matching Food.com recipe via TF-IDF
    character-ngram similarity.  Weight each recipe's SVD latent vector by
    the normalised star rating (1-5 → 0.1-1.0) and average into a taste-
    profile vector.  Return the nearest neighbours in latent space.
    """
    if not rated_recipes or model is None:
        return []

    name_vec  = model["name_vec"]
    name_mat  = model["name_mat"]
    item_fac  = model["item_factors"]
    idx_item  = model["idx_item"]
    meta      = model["recipe_meta"]

    seed_vecs  = []
    seen_idxs  = set()

    for title, rating in rated_recipes.items():
        q    = name_vec.transform([title.lower()])
        sims = cosine_similarity(q, name_mat)[0]
        best = int(sims.argmax())
        if sims[best] < 0.1:          # no convincing match → skip
            continue
        seen_idxs.add(best)
        weight = max((rating - 1) / 4.0, 0.1)   # 1-5 → 0.1-1.0
        seed_vecs.append(item_fac[best] * weight)

    if not seed_vecs:
        return []

    taste_profile = np.mean(seed_vecs, axis=0, keepdims=True)
    all_sims      = cosine_similarity(taste_profile, item_fac)[0]

    results = []
    for idx in np.argsort(all_sims)[::-1]:
        if idx in seen_idxs:
            continue
        food_id = idx_item[idx]
        if food_id not in meta.index:
            continue
        row = meta.loc[food_id]
        ingredients = [i.strip() for i in row["ingredients"].split(",") if i.strip()]
        results.append({
            "name":        row["name"].title(),
            "ingredients": ingredients[:6],
        })
        if len(results) >= n:
            break

    return results

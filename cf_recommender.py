# Recommendation system using the Food.com dataset.
#
# The goal of this file is to recommend recipes
# based on recipes the user liked before.

import ast
import os
import pickle

import numpy as np
import pandas as pd

# Used to create sparse matrices
from scipy.sparse import csr_matrix

# Machine learning model used to reduce dimensions
from sklearn.decomposition import TruncatedSVD

# Used to convert recipe names into vectors
from sklearn.feature_extraction.text import TfidfVectorizer

# Used to compare similarities between recipes
from sklearn.metrics.pairwise import cosine_similarity


# Folder where data is stored
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# File where the trained model will be saved
_MODEL_PATH = os.path.join(_DATA_DIR, "cf_model.pkl")


# Minimum number of ratings needed
# Recipes with too few ratings are ignored
_MIN_RECIPE_RATINGS = 15

# Ignore users with almost no activity
_MIN_USER_RATINGS = 8

# Number of dimensions used by SVD
# Bigger number = more detail but slower
_N_SVD_COMPONENTS = 50

# Maximum TF-IDF features
_NAME_TFIDF_FEATS = 30000


# Convert ingredient list into readable text
def _parse_ingredients(raw):

    try:
        # ast.literal_eval converts string list into real Python list
        # Example:
        # "['salt', 'pepper']"
        # becomes:
        # ['salt', 'pepper']
        return ", ".join(ast.literal_eval(raw))

    except Exception:

        # If something fails just return raw text
        return str(raw)


# Try to find where the dataset is located
def _find_dataset_path():

    # First check if user manually gave a path
    env = os.getenv("CF_DATASET_PATH", "").strip()

    if env and os.path.isdir(env):
        return env

    # Default kagglehub folder
    base = os.path.expanduser(
        "~/.cache/kagglehub/datasets/"
        "shuyangli94/food-com-recipes-and-user-interactions"
    )

    # Check if folder exists
    if os.path.isdir(base):

        # Sometimes files are inside version folders
        for sub in ("versions", ""):

            scan_dir = os.path.join(base, sub) if sub else base

            # Skip if folder does not exist
            if not os.path.isdir(scan_dir):
                continue

            # Get all folders and sort them
            candidates = sorted(
                [
                    d for d in os.listdir(scan_dir)
                    if os.path.isdir(os.path.join(scan_dir, d))
                ],
                reverse=True,
            )

            # Go through folders one by one
            for ver in candidates:

                candidate = os.path.join(scan_dir, ver)

                # Check if interactions file exists
                if os.path.exists(
                    os.path.join(candidate, "RAW_interactions.csv")
                ):
                    return candidate

            # Sometimes files are directly inside base folder
            if os.path.exists(
                os.path.join(base, "RAW_interactions.csv")
            ):
                return base

    # Nothing found
    return None


# Build the recommendation model
def _build_model(dataset_path):

    # Load ratings data
    inter = pd.read_csv(
        os.path.join(dataset_path, "RAW_interactions.csv"),
        usecols=["user_id", "recipe_id", "rating"],
    )

    # Load recipe information
    recipes = pd.read_csv(
        os.path.join(dataset_path, "RAW_recipes.csv"),
        usecols=["id", "name", "ingredients"],
    )

    # Clean ingredient format
    recipes["ingredients"] = recipes["ingredients"].apply(
        _parse_ingredients
    )

    # Count how many ratings each recipe has
    recipe_counts = inter["recipe_id"].value_counts()

    # Count how many ratings each user has
    user_counts = inter["user_id"].value_counts()

    # Keep only recipes and users with enough ratings
    # Otherwise the matrix becomes too noisy
    inter = inter[
        inter["recipe_id"].isin(
            recipe_counts[
                recipe_counts >= _MIN_RECIPE_RATINGS
            ].index
        )
        &
        inter["user_id"].isin(
            user_counts[
                user_counts >= _MIN_USER_RATINGS
            ].index
        )
    ]

    # Get unique recipe IDs
    item_ids = inter["recipe_id"].unique()

    # Get unique user IDs
    user_ids = inter["user_id"].unique()

    # Create dictionaries for indexes
    # Example:
    # recipe id -> matrix column
    item_idx = {
        rid: i for i, rid in enumerate(item_ids)
    }

    # user id -> matrix row
    user_idx = {
        uid: i for i, uid in enumerate(user_ids)
    }

    # Convert IDs into matrix positions
    rows = inter["user_id"].map(user_idx).values
    cols = inter["recipe_id"].map(item_idx).values

    # Ratings values
    vals = inter["rating"].astype(np.float32).values

    # Create sparse matrix
    # Rows = users
    # Columns = recipes
    matrix = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(user_ids), len(item_ids))
    )

    # SVD reduces matrix dimensions
    # This helps find hidden patterns in user tastes
    svd = TruncatedSVD(
        n_components=_N_SVD_COMPONENTS,
        random_state=42
    )

    # Create latent vectors for recipes
    item_factors = svd.fit_transform(matrix.T).astype(
        np.float32
    )

    # Keep metadata only for recipes used
    recipe_meta = (
        recipes[recipes["id"].isin(item_ids)]
        .drop_duplicates("id")
        .set_index("id")[["name", "ingredients"]]
    )

    # TF-IDF vectorizer for recipe names
    # Helps compare similar titles
    name_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 3),
        max_features=_NAME_TFIDF_FEATS
    )

    # Convert recipe names into vectors
    name_mat = name_vec.fit_transform(
        recipe_meta["name"].str.lower()
    )

    # Store everything inside dictionary
    model = {

        # recipe id -> matrix index
        "item_idx": item_idx,

        # matrix index -> recipe id
        "idx_item": {
            v: k for k, v in item_idx.items()
        },

        # latent vectors from SVD
        "item_factors": item_factors,

        # recipe names + ingredients
        "recipe_meta": recipe_meta,

        # TF-IDF model
        "name_vec": name_vec,

        # TF-IDF vectors
        "name_mat": name_mat,
    }

    # Save trained model to file
    with open(_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model


# Check if model file already exists
def model_pickle_exists():

    return os.path.exists(_MODEL_PATH)


# Load saved model
# If model does not exist, build it
def load_cf_model():

    # Load existing model
    if os.path.exists(_MODEL_PATH):

        with open(_MODEL_PATH, "rb") as f:
            return pickle.load(f)

    # Find dataset
    dataset_path = _find_dataset_path()

    # Stop if dataset missing
    if dataset_path is None:
        return None

    # Build new model
    return _build_model(dataset_path)


# Recommend recipes based on user ratings
def get_cf_recommendations(
    rated_recipes,
    model,
    n=6
):

    # Stop if nothing to use
    if not rated_recipes or model is None:
        return []

    # Load objects from model dictionary
    name_vec = model["name_vec"]
    name_mat = model["name_mat"]
    item_fac = model["item_factors"]
    idx_item = model["idx_item"]
    meta = model["recipe_meta"]

    # Store recipe vectors
    seed_vecs = []

    # Avoid duplicates
    seen_idxs = set()

    # Go through recipes rated by user
    for title, rating in rated_recipes.items():

        # Convert title into vector
        q = name_vec.transform([title.lower()])

        # Compare with Food.com titles
        sims = cosine_similarity(q, name_mat)[0]

        # Best match
        best = int(sims.argmax())

        # Ignore weak matches
        if sims[best] < 0.1:
            continue

        # Save recipe index
        seen_idxs.add(best)

        # Convert rating into weight
        # 5 stars = stronger importance
        weight = max((rating - 1) / 4.0, 0.1)

        # Save weighted vector
        seed_vecs.append(
            item_fac[best] * weight
        )

    # Stop if nothing matched
    if not seed_vecs:
        return []

    # Average vectors together
    # This becomes the user's taste profile
    taste_profile = np.mean(
        seed_vecs,
        axis=0,
        keepdims=True
    )

    # Compare taste profile with all recipes
    all_sims = cosine_similarity(
        taste_profile,
        item_fac
    )[0]

    results = []

    # Sort recipes from best to worst
    for idx in np.argsort(all_sims)[::-1]:

        # Skip recipes already used
        if idx in seen_idxs:
            continue

        # Get recipe ID
        food_id = idx_item[idx]

        # Skip if recipe missing
        if food_id not in meta.index:
            continue

        # Get recipe row
        row = meta.loc[food_id]

        # Clean ingredients
        ingredients = [
            i.strip()
            for i in row["ingredients"].split(",")
            if i.strip()
        ]

        # Save recommendation
        results.append({

            # Recipe title
            "name": row["name"].title(),

            # First 6 ingredients only
            "ingredients": ingredients[:6],
        })

        # Stop when enough recipes found
        if len(results) >= n:
            break

    return results

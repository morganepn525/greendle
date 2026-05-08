# recommender.py — Content-based ML recommender using TF-IDF and cosine similarity.
#
# How it works:
#   1. Each recipe's ingredient list is converted into a TF-IDF vector.
#   2. The user's ingredient list is converted into the same vector space.
#   3. Cosine similarity measures how closely each recipe matches the user's ingredients.
#   4. Scores are boosted by the user's past star ratings, so highly-rated recipes
#      (and recipes similar to them) rank higher over time — this is the learning component.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def rank_recipes(recipes: list[dict], user_ingredients: list[str], ratings: dict) -> list[dict]:
    """
    Score and rank recipes by ingredient match and past user ratings.

    Parameters:
        recipes          — list of recipe dicts from the Spoonacular API
        user_ingredients — ingredients the user entered on the search page
        ratings          — dict of {recipe_id: star_rating (1-5)} from previous sessions

    Returns the same list of recipes sorted from best to worst match.

    Scoring formula (per recipe):
        final_score = cosine_similarity * 0.7 + rating_boost * 0.3
    This means ingredient match is the main driver (70%), but the user's
    personal preferences (30%) influence the result over time.
    """
    if not recipes:
        return []

    # Convert each recipe's ingredient list into a single text string for vectorisation
    recipe_texts = [" ".join(r.get("ingredients", [])) for r in recipes]
    user_text    = " ".join(user_ingredients)

    # TF-IDF: rare ingredients are weighted higher than very common ones (e.g. "salt")
    # We fit the vectoriser on all texts at once so recipe and user vectors share the same space
    vectorizer = TfidfVectorizer()
    all_texts  = recipe_texts + [user_text]
    matrix     = vectorizer.fit_transform(all_texts)

    recipe_matrix = matrix[:-1]   # rows 0..N-1 are the recipes
    user_vector   = matrix[-1]    # last row is the user's ingredient query

    # Cosine similarity: 1.0 = identical ingredient sets, 0.0 = nothing in common
    scores = cosine_similarity(user_vector, recipe_matrix)[0]

    # Boost scores using the user's past ratings to personalise results over time
    for i, recipe in enumerate(recipes):
        meal_id = str(recipe.get("id", recipe.get("idMeal", "")))
        rating  = ratings.get(meal_id, 3)   # default to neutral (3 stars) if not yet rated
        rating_boost = (rating - 1) / 4     # normalise 1-5 stars → 0.0-1.0
        scores[i] = scores[i] * 0.7 + rating_boost * 0.3

    # Sort descending so the best match appears first
    ranked = sorted(
        zip(scores, recipes),
        key=lambda x: x[0],
        reverse=True
    )
    return [recipe for _, recipe in ranked]

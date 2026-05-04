from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rank_recipes(recipes: list[dict], user_ingredients: list[str], ratings: dict) -> list[dict]:
    """
    Score each recipe by how well its ingredients match what the user has,
    then boost the score using the user's past star ratings.

    recipes          — list of meal dicts from the API
    user_ingredients — ingredients the user entered
    ratings          — dict of {meal_id: star_rating (1-5)}
    """
    if not recipes:
        return []

    # Build one text string per recipe (all its ingredients joined)
    recipe_texts = [" ".join(r.get("ingredients", [])) for r in recipes]
    user_text    = " ".join(user_ingredients)

    # TF-IDF vectorizer turns ingredient lists into numeric vectors
    vectorizer = TfidfVectorizer()
    all_texts  = recipe_texts + [user_text]
    matrix     = vectorizer.fit_transform(all_texts)

    recipe_matrix = matrix[:-1]   # all recipes
    user_vector   = matrix[-1]    # the user's ingredients

    # Cosine similarity: 1.0 = perfect match, 0.0 = nothing in common
    scores = cosine_similarity(user_vector, recipe_matrix)[0]

    # Boost by past ratings (rating out of 5, normalised to 0-1)
    for i, recipe in enumerate(recipes):
        meal_id = str(recipe.get("id", recipe.get("idMeal", "")))
        rating  = ratings.get(meal_id, 3)   # default neutral score of 3
        rating_boost = (rating - 1) / 4     # maps 1-5 → 0.0-1.0
        scores[i] = scores[i] * 0.7 + rating_boost * 0.3

    # Sort recipes from best to worst match
    ranked = sorted(
        zip(scores, recipes),
        key=lambda x: x[0],
        reverse=True
    )
    return [recipe for _, recipe in ranked]
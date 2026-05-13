# recommender.py
#
# This file contains a simple recommendation system for recipes.
#
# The goal:
# Recommend recipes that match the user's ingredients.
#
# The system uses:
#   - TF-IDF vectorisation
#   - Cosine similarity
#   - User ratings for personalisation
#
# General process:
#
#   1. Convert recipe ingredients into text
#   2. Convert the user's ingredients into text
#   3. Transform all text into vectors (numbers)
#   4. Compare vectors using cosine similarity
#   5. Improve scores using previous user ratings
#   6. Sort recipes from best to worst


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def rank_recipes(recipes, user_ingredients, ratings):
    """
    Rank recipes using ingredient similarity and user ratings.

    Parameters
    ----------
    recipes:
        List of recipe dictionaries from the API

    user_ingredients:
        Ingredients entered by the user

    ratings:
        Dictionary containing previous ratings

        Example:
        {
            "1234": 5,
            "5678": 2
        }

    Returns
    -------
    A list of recipes sorted from best match to worst match.
    """

    # If the recipe list is empty,
    # return an empty list immediately
    if not recipes:
        return []

    # ---------------------------------------------------------
    # STEP 1 — Convert ingredients into text
    # ---------------------------------------------------------

    # Each recipe contains a list of ingredients.
    #
    # Example:
    # ["chicken", "rice", "onion"]
    #
    # becomes:
    # "chicken rice onion"

    recipe_texts = [
        " ".join(r.get("ingredients", []))
        for r in recipes
    ]

    # Convert the user's ingredients into one text string
    #
    # Example:
    # ["tomato", "garlic", "cheese"]
    #
    # becomes:
    # "tomato garlic cheese"

    user_text = " ".join(user_ingredients)

    # ---------------------------------------------------------
    # STEP 2 — TF-IDF vectorisation
    # ---------------------------------------------------------

    # TF-IDF converts words into numbers.
    #
    # Important ingredients receive higher values.
    #
    # Common ingredients like:
    #   "salt"
    #   "water"
    #
    # receive lower importance.

    vectorizer = TfidfVectorizer()

    # Combine all recipe texts + the user's text
    all_texts = recipe_texts + [user_text]

    # Convert all texts into vectors
    #
    # The result is a matrix of numbers
    matrix = vectorizer.fit_transform(all_texts)

    # All rows except the last one = recipe vectors
    recipe_matrix = matrix[:-1]

    # Last row = user vector
    user_vector = matrix[-1]

    # ---------------------------------------------------------
    # STEP 3 — Calculate similarity
    # ---------------------------------------------------------

    # Cosine similarity compares vectors.
    #
    # Results:
    #
    # 1.0  = perfect match
    # 0.0  = nothing in common

    scores = cosine_similarity(user_vector, recipe_matrix)[0]

    # ---------------------------------------------------------
    # STEP 4 — Add personalisation using ratings
    # ---------------------------------------------------------

    # Recipes the user rated highly should rank higher.
    #
    # Example:
    #
    # If two recipes are equally similar,
    # but the user liked one more in the past,
    # that recipe should appear first.

    for i, recipe in enumerate(recipes):

        # Get recipe ID
        #
        # Some APIs use:
        #   "id"
        #
        # Others use:
        #   "idMeal"

        meal_id = str(
            recipe.get("id", recipe.get("idMeal", ""))
        )

        # Get previous rating
        #
        # Default = 3 stars (neutral)
        rating = ratings.get(meal_id, 3)

        # Convert rating scale:
        #
        # 1 star -> 0.0
        # 5 stars -> 1.0

        rating_boost = (rating - 1) / 4

        # Final score formula:
        #
        # 70% ingredient similarity
        # 30% user preference

        scores[i] = scores[i] * 0.7 + rating_boost * 0.3

    # ---------------------------------------------------------
    # STEP 5 — Sort recipes
    # ---------------------------------------------------------

    # zip(scores, recipes)
    #
    # combines:
    #
    # [
    #   (0.91, recipe1),
    #   (0.72, recipe2),
    #   ...
    # ]

    ranked = sorted(
        zip(scores, recipes),

        # Sort using the score
        key=lambda x: x[0],

        # Highest score first
        reverse=True
    )

    # Return only the recipes
    # without the scores

    return [recipe for _, recipe in ranked]

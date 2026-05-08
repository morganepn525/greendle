# api.py — Handles all communication with the Spoonacular Recipe API.
# The API key is loaded from the .env file to keep credentials out of source code.

import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file (contains the API key)
load_dotenv()
API_KEY = os.getenv("SPOONACULAR_KEY")
BASE_URL = "https://api.spoonacular.com/recipes"


# Custom exceptions so the app can show helpful error messages to the user
class APIKeyMissingError(Exception):
    """Raised when the Spoonacular API key is missing or invalid (HTTP 401)."""
    pass

class APIError(Exception):
    """Raised for any other API error (e.g. quota exceeded, server error)."""
    pass


def _check_response(response: requests.Response) -> dict:
    """Validate an API response and return the JSON body, or raise a descriptive error."""
    if response.status_code == 401:
        raise APIKeyMissingError(
            "Spoonacular API key is missing or invalid. "
            "Add SPOONACULAR_KEY=your_key to your .env file."
        )
    if response.status_code == 402:
        # 402 means the free daily quota (150 points) has been used up
        raise APIError("Daily API quota exceeded. Try again tomorrow.")
    if not response.ok:
        raise APIError(f"API error {response.status_code}: {response.text[:200]}")
    return response.json()


def search_by_ingredients(ingredients: list[str], number: int = 8,
                          diet: str = None, intolerances: list[str] = None) -> list[dict]:
    """
    Search for recipes that use the given ingredients.

    Parameters:
        ingredients  — list of ingredient names entered by the user
        number       — max number of recipes to return (default 8 to limit API usage)
        diet         — optional dietary filter, e.g. "vegan" or "vegetarian"
        intolerances — optional list of intolerances, e.g. ["gluten", "dairy"]

    Returns a list of recipe dicts from the Spoonacular API, each enriched with
    an 'ingredients' key containing a plain list of ingredient names.
    """
    url = f"{BASE_URL}/complexSearch"
    params = {
        "includeIngredients": ",".join(ingredients),
        "number": number,
        "apiKey": API_KEY,
        "addRecipeInformation": True,   # include full recipe details in response
        "fillIngredients": True,         # ensure extendedIngredients is populated
        "sort": "max-used-ingredients",  # prioritise recipes that use the most of the user's ingredients
    }
    if diet:
        params["diet"] = diet
    if intolerances:
        params["intolerances"] = ",".join(intolerances)

    data = _check_response(requests.get(url, params=params))
    results = data.get("results", [])

    # Add a flat 'ingredients' list to each recipe for easier use in the recommender
    for r in results:
        r["ingredients"] = [i["name"] for i in r.get("extendedIngredients", [])]
    return results


def get_recipe_by_id(recipe_id: int) -> dict | None:
    """
    Fetch the full details of a single recipe by its Spoonacular ID.

    Used to enrich search results with complete ingredient lists and
    instructions, which are not always fully included in the search response.
    Returns None if the recipe is not found.
    """
    url = f"{BASE_URL}/{recipe_id}/information"
    data = _check_response(requests.get(url, params={"apiKey": API_KEY}))
    if "id" not in data:
        return None
    # Add flat ingredient name list for use in the ML recommender
    data["ingredients"] = [i["name"] for i in data.get("extendedIngredients", [])]
    return data

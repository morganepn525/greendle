import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("SPOONACULAR_KEY")
BASE_URL = "https://api.spoonacular.com/recipes"


def search_by_ingredients(ingredients: list[str], number: int = 8,
                          diet: str = None, intolerances: list[str] = None) -> list[dict]:
    """Search recipes that use the given ingredients, optionally filtered by diet."""
    url = f"{BASE_URL}/complexSearch"
    params = {
        "includeIngredients": ",".join(ingredients),
        "number": number,
        "apiKey": API_KEY,
        "addRecipeInformation": True,
        "fillIngredients": True,
        "sort": "max-used-ingredients",
    }
    if diet:
        params["diet"] = diet
    if intolerances:
        params["intolerances"] = ",".join(intolerances)

    response = requests.get(url, params=params)
    results = response.json().get("results", [])

    for r in results:
        r["ingredients"] = [i["name"] for i in r.get("extendedIngredients", [])]
    return results


def get_recipe_by_id(recipe_id: int) -> dict | None:
    """Return full recipe details for a given Spoonacular recipe ID."""
    url = f"{BASE_URL}/{recipe_id}/information"
    params = {"apiKey": API_KEY}
    response = requests.get(url, params=params)
    data = response.json()
    if "id" not in data:
        return None
    data["ingredients"] = [i["name"] for i in data.get("extendedIngredients", [])]
    return data

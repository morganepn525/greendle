import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("SPOONACULAR_KEY")
BASE_URL = "https://api.spoonacular.com/recipes"


class APIKeyMissingError(Exception):
    pass

class APIError(Exception):
    pass


def _check_response(response: requests.Response) -> dict:
    if response.status_code == 401:
        raise APIKeyMissingError(
            "Spoonacular API key is missing or invalid. "
            "Add SPOONACULAR_KEY=your_key to your .env file."
        )
    if response.status_code == 402:
        raise APIError("Daily API quota exceeded. Try again tomorrow.")
    if not response.ok:
        raise APIError(f"API error {response.status_code}: {response.text[:200]}")
    return response.json()


def search_by_ingredients(ingredients: list[str], number: int = 8,
                          diet: str = None, intolerances: list[str] = None) -> list[dict]:
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

    data = _check_response(requests.get(url, params=params))
    results = data.get("results", [])
    for r in results:
        r["ingredients"] = [i["name"] for i in r.get("extendedIngredients", [])]
    return results


def get_recipe_by_id(recipe_id: int) -> dict | None:
    url = f"{BASE_URL}/{recipe_id}/information"
    data = _check_response(requests.get(url, params={"apiKey": API_KEY}))
    if "id" not in data:
        return None
    data["ingredients"] = [i["name"] for i in data.get("extendedIngredients", [])]
    return data

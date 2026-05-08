# api.py
# This file is responsible for talking to the Spoonacular Recipe API.
# Think of it like a messenger: we send a request, the API sends back recipes.

import requests
import os
from dotenv import load_dotenv

# This loads our secret API key from a file called .env
# We do this so the key is not visible in our code
load_dotenv()
my_api_key = os.getenv("SPOONACULAR_KEY")

# This is the main address of the Spoonacular API
# All our requests will start with this address
website_address = "https://api.spoonacular.com/recipes"


# --- Custom Error Messages ---
# These are special error types we create ourselves.
# When something goes wrong, we can tell the user exactly what happened.

class APIKeyMissingError(Exception):
    # This error happens when the API key is wrong or missing
    pass

class APIError(Exception):
    # This error happens when something else goes wrong with the API
    pass


# --- Helper Function ---
# This function checks if the API response is okay.
# We use it in every request so we don't have to repeat the same checks.

def check_response(response):
    # If the status code is 401, the API key is wrong or missing
    if response.status_code == 401:
        raise APIKeyMissingError(
            "Spoonacular API key is missing or invalid. "
            "Add SPOONACULAR_KEY=your_key to your .env file."
        )

    # If the status code is 402, we have used up all our free requests for today
    if response.status_code == 402:
        raise APIError("Daily API quota exceeded. Try again tomorrow.")

    # If something else went wrong, we show a general error message
    if response.ok == False:
        raise APIError("API error " + str(response.status_code) + ": " + response.text[:200])

    # If everything is fine, we return the data as a dictionary
    return response.json()


# --- Search Function ---
# This function searches for recipes based on ingredients the user has at home.
# For example: ingredients_list = ["tomato", "cheese", "egg"]

def search_recipes_with_ingredients(ingredients_list, number_of_recipes=8, diet=None, intolerances=None):
    # This is the exact address for the search
    search_address = website_address + "/complexSearch"

    # These are all the options we send to the API
    options = {
        "includeIngredients": ",".join(ingredients_list),  # turns ["a", "b"] into "a,b"
        "number": number_of_recipes,                        # how many recipes we want back
        "apiKey": my_api_key,                               # our secret key
        "addRecipeInformation": True,                       # we want extra recipe details
        "fillIngredients": True,                            # we want the full ingredient list
        "sort": "max-used-ingredients",                     # show recipes that use the most of our ingredients first
    }

    # If the user wants to filter by diet (like "vegan"), we add that option
    if diet != None:
        options["diet"] = diet

    # If the user has food intolerances (like "gluten"), we add those too
    if intolerances != None:
        options["intolerances"] = ",".join(intolerances)

    # We send the request to the API and check if the response is okay
    response = requests.get(search_address, params=options)
    data = check_response(response)

    # We get the list of recipes from the response
    all_recipes = data.get("results", [])

    # For each recipe, we create a simple list with only the ingredient names
    # This makes it easier to work with later
    for one_recipe in all_recipes:
        ingredient_names = []
        for ingredient in one_recipe.get("extendedIngredients", []):
            ingredient_names.append(ingredient["name"])
        one_recipe["ingredients"] = ingredient_names

    # We return the full list of recipes
    return all_recipes


# --- Get Single Recipe Function ---
# This function gets all the details for one specific recipe.
# We need the recipe ID number to find it (every recipe has a unique ID).

def get_recipe_with_id(recipe_id):
    # This is the address for one specific recipe
    recipe_address = website_address + "/" + str(recipe_id) + "/information"

    # We send the request and check if the response is okay
    response = requests.get(recipe_address, params={"apiKey": my_api_key})
    recipe_data = check_response(response)

    # If the response does not contain an "id", the recipe was not found
    if "id" not in recipe_data:
        return None

    # We create a simple list with only the ingredient names
    ingredient_names = []
    for ingredient in recipe_data.get("extendedIngredients", []):
        ingredient_names.append(ingredient["name"])

    recipe_data["ingredients"] = ingredient_names

    # We return the recipe with all its details
    return recipe_data

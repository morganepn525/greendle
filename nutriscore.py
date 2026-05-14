# Greendle Health Score (0–100)
#   = Nutri component (0–60 pts, 60% weight)
#   + NOVA component  (10–40 pts, based on ingredient processing level)
#
# Nutri component
#   Scores 7 nutrients per serving against FDA Daily Values:
#   calories, total fat, saturated fat, sugar, sodium, protein, carbohydrates.
#   Negative nutrients lower the score; protein raises it.
#
# NOVA component
#   Classifies each ingredient with the NOVA processing scale (1–4) and
#   computes a gram-weighted average penalty:
#     NOVA 1 (unprocessed)       →  0 pts penalty
#     NOVA 2 (culinary ingred.)  →  5 pts penalty
#     NOVA 3 (processed)         → 15 pts penalty
#     NOVA 4 (ultra-processed)   → 30 pts penalty
#   NOVA component = 40 – weighted_average_penalty  (range 10–40)
#
# Primary NOVA classification uses curated keyword sets.
# The lightningxyz NOVA dataset is used as a secondary exact-match lookup.

import os
import re
import pandas as pd
import streamlit as st

try:
    import kagglehub
    _KAGGLE_AVAILABLE = True
except ImportError:
    _KAGGLE_AVAILABLE = False

# ── FDA Daily Values (per day) ───────────────────────────────────────────────
_DV_FAT      = 78.0    # g
_DV_SAT_FAT  = 20.0    # g
_DV_SUGAR    = 50.0    # g
_DV_SODIUM   = 2.3     # g  (= 2 300 mg)
_DV_PROTEIN  = 50.0    # g
_DV_CARBS    = 275.0   # g
_DV_CALORIES = 2000.0  # kcal

# ── NOVA keyword sets ────────────────────────────────────────────────────────
# Phrases are checked before individual tokens (longest-match wins).
_NOVA_PHRASES = {
    # NOVA 1
    "lemon juice": 1, "lime juice": 1, "orange juice": 1,
    "cherry tomato": 1, "grape tomato": 1,
    "pasta water": 1,  # just salted water, not the pasta itself
    # NOVA 2
    "olive oil": 2, "vegetable oil": 2, "canola oil": 2, "coconut oil": 2,
    "sunflower oil": 2, "sesame oil": 2, "avocado oil": 2,
    "baking powder": 2, "baking soda": 2, "cream of tartar": 2,
    "maple syrup": 2, "brown sugar": 2, "caster sugar": 2,
    "powdered sugar": 2, "icing sugar": 2, "confectioners sugar": 2,
    "heavy cream": 2, "double cream": 2, "whipping cream": 2,
    "sour cream": 2, "creme fraiche": 2,
    "all purpose flour": 2, "plain flour": 2, "bread flour": 2,
    "cake flour": 2, "whole wheat flour": 2, "corn flour": 2,
    "corn starch": 2, "cornstarch": 2,
    "rice vinegar": 2, "apple cider vinegar": 2, "balsamic vinegar": 2,
    "fish sauce": 2, "miso": 2, "tahini": 2,
    # NOVA 3
    "cream cheese": 3, "cottage cheese": 3,
    "canned tomato": 3, "canned bean": 3, "canned chickpea": 3,
    "tomato sauce": 3, "tomato paste": 3, "tomato puree": 3,
    "soy sauce": 3, "worcestershire sauce": 3, "oyster sauce": 3,
    "hoisin sauce": 3, "teriyaki sauce": 3,
    "peanut butter": 3, "almond butter": 3,
    "smoked salmon": 3,
    "sun dried tomato": 3, "sun-dried tomato": 3,
    # NOVA 4
    "hot sauce": 4, "ranch dressing": 4, "french dressing": 4,
    "caesar dressing": 4, "thousand island": 4,
    "bbq sauce": 4, "buffalo sauce": 4,
    "instant noodle": 4, "frozen meal": 4,
}

_NOVA1_WORDS = frozenset([
    # poultry & meat (raw/fresh)
    "chicken", "beef", "pork", "lamb", "turkey", "veal", "duck", "rabbit",
    "venison", "bison",
    # fish & seafood (raw/fresh)
    "salmon", "tuna", "cod", "shrimp", "prawn", "lobster", "crab",
    "mussel", "clam", "scallop", "halibut", "tilapia", "trout",
    "sardine", "anchovy", "mackerel", "sole", "snapper",
    # eggs
    "egg", "eggs",
    # vegetables
    "tomato", "onion", "garlic", "carrot", "celery", "zucchini", "squash",
    "cucumber", "lettuce", "spinach", "kale", "arugula", "broccoli",
    "cauliflower", "mushroom", "asparagus", "eggplant", "artichoke",
    "leek", "potato", "yam", "sweet potato", "corn", "pea", "edamame",
    "green bean", "beet", "fennel", "cabbage", "bok choy", "turnip",
    "radish", "okra", "scallion", "chive", "shallot", "jalapeno",
    "poblano", "serrano", "habanero", "tomatillo", "celeriac",
    "parsnip", "rutabaga", "kohlrabi", "watercress", "endive", "radicchio",
    # bell pepper / capsicum
    "pepper", "capsicum",
    # fruits
    "apple", "lemon", "lime", "orange", "banana", "avocado", "strawberry",
    "blueberry", "mango", "peach", "pear", "cherry", "grape", "watermelon",
    "pineapple", "kiwi", "fig", "date", "apricot", "plum", "nectarine",
    "raspberry", "blackberry", "cranberry", "grapefruit", "papaya",
    "pomegranate", "guava", "passion fruit", "dragonfruit", "jackfruit",
    # legumes & plant protein
    "bean", "lentil", "chickpea", "tofu", "tempeh", "seitan",
    # nuts & seeds (raw)
    "almond", "walnut", "cashew", "pecan", "pistachio", "hazelnut",
    "pine nut", "macadamia", "sesame", "chia", "flaxseed", "hemp",
    "sunflower seed", "pumpkin seed",
    # whole grains & starches (unprocessed)
    "rice", "oat", "quinoa", "barley", "millet", "buckwheat", "amaranth",
    # herbs (fresh)
    "basil", "parsley", "cilantro", "mint", "dill", "tarragon",
    "rosemary", "thyme", "sage", "oregano", "marjoram", "lavender",
    # plain dairy
    "milk", "cream", "yogurt", "buttermilk", "kefir", "whey",
    # other
    "water", "ice", "gelatin",
])

_NOVA2_WORDS = frozenset([
    # oils & fats
    "oil", "butter", "ghee", "lard", "shortening", "suet",
    # flours & starches
    "flour", "starch", "arrowroot",
    # sugars & sweeteners
    "sugar", "honey", "molasses", "agave", "stevia", "xylitol",
    # salt & leaveners
    "salt", "yeast",
    # dried spices
    "paprika", "cumin", "cinnamon", "turmeric", "nutmeg", "cardamom",
    "coriander", "allspice", "cayenne", "ginger", "clove", "anise",
    "fennel seed", "mustard seed", "fenugreek", "sumac", "za'atar",
    # vinegars
    "vinegar",
])

_NOVA3_WORDS = frozenset([
    # cured / fermented meats (minimally)
    "ham", "prosciutto", "pancetta", "pastrami",
    # cheeses
    "cheese", "mozzarella", "cheddar", "parmesan", "feta", "brie",
    "camembert", "gruyere", "gouda", "ricotta", "emmental",
    "parmigiano", "pecorino", "manchego", "halloumi", "burrata",
    # bread & baked staples
    "bread", "tortilla", "pita", "naan", "bagel", "croissant",
    # pasta & noodles
    "pasta", "noodle", "spaghetti", "fettuccine", "penne", "linguine",
    "lasagna", "ravioli", "gnocchi", "orzo", "rigatoni", "farfalle",
    "fettuccini", "tagliatelle",
    # canned / preserved (simple)
    "canned", "tinned", "smoked", "pickled", "brined",
    # simple condiments / fermented
    "mustard", "tabasco", "marmite",
    # other processed basics
    "breadcrumb", "panko",
])

_NOVA4_WORDS = frozenset([
    # processed meats with additives
    "bacon", "sausage", "salami", "pepperoni", "chorizo", "hot dog",
    "spam", "bologna", "frankfurter", "kielbasa",
    # condiments & dressings
    "ketchup", "mayonnaise", "ranch", "relish", "aioli",
    # sweet spreads
    "nutella", "jam", "jelly", "syrup",
    # snacks
    "chips", "crackers", "pretzels", "popcorn",
    # fats
    "margarine",
    # ready / instant
    "frozen", "instant", "microwave",
    # drinks
    "soda", "cola",
])

_NOVA_PENALTY = {1: 0, 2: 5, 3: 15, 4: 30}

# ── Nutri-Score weight map ────────────────────────────────────────────────────
# Lower bad nutrients → better subscore; higher protein → better subscore.
# Weights sum to 1.0.
_NUTRI_WEIGHTS = {
    "calories": 0.15,
    "fat":      0.10,
    "sat_fat":  0.20,
    "sugar":    0.15,
    "sodium":   0.15,
    "protein":  0.20,   # positive factor
    "carbs":    0.05,
}


# ── Cooking-method penalties ─────────────────────────────────────────────────
COOKING_PENALTIES = {
    "ultra_processed": {
        "keywords": [
            "nuggets", "instant", "ready meal", "frozen meal",
            "microwave", "processed", "artificial", "powder mix",
        ],
        "nova": 4, "penalty": 30,
    },
    "deep_fried": {
        "keywords": [
            "fried", "deep fried", "deep-fried", "battered",
            "breaded", "crispy", "buttermilk fried", "pan fried",
            "pan-fried", "fritter", "tempura", "schnitzel",
            "katsu", "churros", "doughnut", "donut", "fries",
            "chips", "croquette", "spring roll", "egg roll",
        ],
        "nova": 4, "penalty": 25,
    },
    "heavy_processed": {
        "keywords": [
            "smoked", "cured", "salted", "pickled", "marinated",
            "preserved", "jerky", "bacon", "ham", "sausage",
            "hot dog", "pepperoni", "salami", "chorizo",
            "bologna", "spam", "corned beef", "paté",
        ],
        "nova": 3, "penalty": 15,
    },
    "high_fat_cooking": {
        "keywords": [
            "buttered", "creamy", "cream sauce", "cheesy",
            "loaded", "smothered", "au gratin", "gratinée",
            "alfredo", "carbonara", "hollandaise", "béarnaise",
        ],
        "nova": 3, "penalty": 12,
    },
    "moderate_processed": {
        "keywords": [
            "roasted", "baked", "grilled", "bbq", "barbecue",
            "glazed", "caramelised", "caramelized", "candied",
            "stuffed", "wrapped", "crusted", "sautéed", "sauteed",
        ],
        "nova": 2, "penalty": 5,
    },
    "healthy_cooking": {
        "keywords": [
            "steamed", "raw", "fresh", "poached", "blanched",
            "boiled", "light", "salad", "smoothie", "juice",
            "whole grain", "wholegrain", "organic", "natural",
        ],
        "nova": 1, "penalty": 0,
    },
}

# ── Ingredient-level penalties / bonuses ─────────────────────────────────────
INGREDIENT_PENALTIES = {
    "very_unhealthy": {
        "keywords": [
            "lard", "shortening", "margarine", "hydrogenated",
            "trans fat", "high fructose", "corn syrup",
            "artificial sweetener", "msg", "monosodium",
        ],
        "penalty": 20,
    },
    "unhealthy": {
        "keywords": [
            "white sugar", "refined sugar", "heavy cream",
            "double cream", "condensed milk", "mayonnaise",
            "ranch", "thousand island", "cheese sauce",
        ],
        "penalty": 10,
    },
    "moderate": {
        "keywords": [
            "cheese", "butter", "cream cheese", "sour cream",
            "white rice", "white bread", "white flour",
            "pasta", "noodles", "tortilla",
        ],
        "penalty": 5,
    },
    "healthy": {
        "keywords": [
            "spinach", "kale", "broccoli", "quinoa", "lentils",
            "chickpeas", "avocado", "salmon", "sardine",
            "olive oil", "nuts", "seeds", "berries",
            "sweet potato", "oats", "legumes", "tofu",
        ],
        "penalty": -5,
    },
    "super_healthy": {
        "keywords": [
            "turmeric", "ginger", "garlic", "blueberries",
            "chia seeds", "flaxseed", "wheatgrass", "spirulina",
            "kimchi", "kefir", "kombucha", "miso", "tempeh",
        ],
        "penalty": -10,
    },
}


# Scan recipe name and ingredients against cooking-method and ingredient keyword sets.
# Returns NOVA level, total penalty (capped at 40), and matched keywords.
# Penalty scale → nova_0_100 = max(0, 100 − penalty × 2.5)
#   penalty =  0  → 100  (perfect)
#   penalty = 25  →  37.5 (deep fried)
#   penalty = 40  →   0  (worst)
# Negative penalties (bonuses) are capped so nova_0_100 ≤ 100.
def apply_keyword_penalties(recipe_name, ingredients):
    text = (recipe_name + " " + " ".join(ingredients)).lower()

    total_penalty    = 0
    nova_level       = 1
    matched_keywords = []

    for data in COOKING_PENALTIES.values():
        for kw in data["keywords"]:
            if kw in text:
                total_penalty += data["penalty"]
                nova_level     = max(nova_level, data["nova"])
                matched_keywords.append(kw)
                break  # one hit per category

    for data in INGREDIENT_PENALTIES.values():
        for kw in data["keywords"]:
            if kw in text:
                total_penalty += data["penalty"]
                matched_keywords.append(kw)
                break  # one hit per category

    total_penalty = min(total_penalty, 40)

    return {
        "nova_level":        nova_level,
        "keyword_penalty":   total_penalty,
        "matched_keywords":  matched_keywords,
    }


# ── Unit → grams conversion ──────────────────────────────────────────────────
_UNIT_GRAMS = {
    "g": 1.0, "gram": 1.0, "grams": 1.0,
    "kg": 1000.0,
    "ml": 1.0, "milliliter": 1.0, "milliliters": 1.0,
    "l": 1000.0, "liter": 1000.0, "liters": 1000.0,
    "tsp": 5.0, "tsps": 5.0, "teaspoon": 5.0, "teaspoons": 5.0,
    "tbsp": 15.0, "tbsps": 15.0, "tablespoon": 15.0, "tablespoons": 15.0,
    "oz": 28.35, "ounce": 28.35, "ounces": 28.35,
    "lb": 453.59, "lbs": 453.59, "pound": 453.59, "pounds": 453.59,
    "cup": 240.0, "cups": 240.0,
    "clove": 5.0, "cloves": 5.0,
    "slice": 30.0, "slices": 30.0, "strip": 30.0, "strips": 30.0,
    "bunch": 100.0, "stick": 113.0,
    "can": 400.0, "cans": 400.0, "small can": 400.0, "jar": 400.0,
    "sprig": 5.0, "sprigs": 5.0,
    "leaf": 2.0, "leaves": 2.0,
    "sheet": 10.0, "sheets": 10.0,
    "loaf": 500.0,
    "large": 150.0, "medium": 100.0, "small": 50.0,
    "large ball": 150.0,
    "inch": 30.0,
}
_DEFAULT_GRAMS = 50.0
_SKIP_UNITS = {"serving", "servings"}


# ── Loaders ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_nutrition_db():
    if not _KAGGLE_AVAILABLE:
        return pd.DataFrame()
    try:
        path = kagglehub.dataset_download("utsavdey1410/food-nutrition-dataset")
        folder = os.path.join(path, "FINAL FOOD DATASET")
        dfs = []
        for fname in [
            "FOOD-DATA-GROUP1.csv", "FOOD-DATA-GROUP2.csv", "FOOD-DATA-GROUP3.csv",
            "FOOD-DATA-GROUP4.csv", "FOOD-DATA-GROUP5.csv",
        ]:
            fp = os.path.join(folder, fname)
            if os.path.exists(fp):
                dfs.append(pd.read_csv(fp))
        if not dfs:
            return pd.DataFrame()
        df = pd.concat(dfs, ignore_index=True)
        df["food_lower"] = df["food"].str.lower().str.strip()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_nova_db():
    if not _KAGGLE_AVAILABLE:
        return pd.DataFrame()
    try:
        path = kagglehub.dataset_download("lightningxyz/international-food-nutrition-index-dataset")
        fp = os.path.join(path, "nutri_scan_final.csv")
        if not os.path.exists(fp):
            return pd.DataFrame()
        df = pd.read_csv(fp, usecols=["Product_Name", "NOVA_Group"])
        df = df.dropna(subset=["NOVA_Group"])
        df["product_lower"] = df["Product_Name"].str.lower().str.strip()
        df["nova_int"] = df["NOVA_Group"].astype(int)
        return df
    except Exception:
        return pd.DataFrame()


# ── NOVA classification ──────────────────────────────────────────────────────

# Return NOVA group (1–4) for an ingredient name.
# Priority: phrase match → token match → dataset lookup → default NOVA 2.
def _classify_nova(name, nova_df=None):
    n = name.lower().strip()
    n_clean = re.sub(r"[^a-z0-9 ]", " ", n)

    # 1. Phrase match (longest phrase first via sorted dict)
    for phrase in sorted(_NOVA_PHRASES, key=len, reverse=True):
        if phrase in n_clean:
            return _NOVA_PHRASES[phrase]

    # 2. Token match — take worst (most processed) NOVA found
    tokens = set(n_clean.split())
    worst = 0
    for tok in tokens:
        if tok in _NOVA4_WORDS:
            return 4
        if tok in _NOVA3_WORDS:
            worst = max(worst, 3)
        elif tok in _NOVA2_WORDS:
            worst = max(worst, 2)
        elif tok in _NOVA1_WORDS:
            worst = max(worst, 1)

    if worst > 0:
        return worst

    # 3. Exact product-name match in the NOVA dataset
    if nova_df is not None and not nova_df.empty:
        m = nova_df[nova_df["product_lower"] == n]
        if not m.empty:
            return int(m.iloc[0]["nova_int"])

    # 4. Default — assume a basic cooking ingredient
    return 2


# ── Ingredient helpers ───────────────────────────────────────────────────────

# Convert an ingredient's measured amount to grams. Skips 'servings' unit.
def _ingredient_grams(ing):
    metric   = ing.get("measures", {}).get("metric", {})
    m_unit   = (metric.get("unitShort") or metric.get("unitLong") or "").lower().strip()
    m_amount = float(metric.get("amount") or 0)

    if m_amount > 0 and m_unit not in _SKIP_UNITS:
        g = _UNIT_GRAMS.get(m_unit)
        if g:
            return m_amount * g

    us       = ing.get("measures", {}).get("us", {})
    u_unit   = (us.get("unitShort") or us.get("unitLong") or "").lower().strip()
    u_amount = float(us.get("amount") or 0)

    if u_amount > 0 and u_unit not in _SKIP_UNITS:
        g = _UNIT_GRAMS.get(u_unit)
        if g:
            return u_amount * g

    return _DEFAULT_GRAMS


# Return the best-matching nutrition-DB row for an ingredient name.
def _match_ingredient(name, db):
    if db.empty or not name:
        return None
    n = name.lower().strip()

    m = db[db["food_lower"] == n]
    if not m.empty:
        return m.iloc[0]

    m = db[db["food_lower"].str.startswith(n + " ")]
    if not m.empty:
        return m.loc[m["food_lower"].str.len().idxmin()]

    pattern = r"\b" + re.escape(n) + r"\b"
    m = db[db["food_lower"].str.contains(pattern, na=False, regex=True)]
    if not m.empty:
        return m.loc[m["food_lower"].str.len().idxmin()]

    m = db[db["food_lower"].str.contains(re.escape(n), na=False, regex=False)]
    if not m.empty:
        return m.loc[m["food_lower"].str.len().idxmin()]

    toks = set(n.split())
    best_row, best_score = None, 0.0
    for _, row in db.iterrows():
        rt    = set(row["food_lower"].split())
        union = toks | rt
        j     = len(toks & rt) / len(union) if union else 0.0
        if j > best_score:
            best_score, best_row = j, row
    return best_row if best_score > 0 else None


# ── Scoring ──────────────────────────────────────────────────────────────────

# Compute a 0–100 nutritional score from whole-recipe gram totals.
# Each nutrient is a fraction of its FDA Daily Value per serving; protein is the only positive factor.
def _nutri_score_0_100(calories, fat_g, sat_fat_g, sugar_g, sodium_g,
                        protein_g, carbs_g, servings):
    n = max(1, servings)

    def frac(val, dv):
        return min(1.0, (val / n) / dv)

    cal_f  = min(1.0, (calories / n) / _DV_CALORIES)
    fat_f  = frac(fat_g,     _DV_FAT)
    sat_f  = frac(sat_fat_g, _DV_SAT_FAT)
    sug_f  = frac(sugar_g,   _DV_SUGAR)
    sod_f  = frac(sodium_g,  _DV_SODIUM)
    pro_f  = frac(protein_g, _DV_PROTEIN)
    carb_f = frac(carbs_g,   _DV_CARBS)

    score = (
        (1.0 - cal_f)  * _NUTRI_WEIGHTS["calories"] +
        (1.0 - fat_f)  * _NUTRI_WEIGHTS["fat"]      +
        (1.0 - sat_f)  * _NUTRI_WEIGHTS["sat_fat"]  +
        (1.0 - sug_f)  * _NUTRI_WEIGHTS["sugar"]    +
        (1.0 - sod_f)  * _NUTRI_WEIGHTS["sodium"]   +
        pro_f          * _NUTRI_WEIGHTS["protein"]   +
        (1.0 - carb_f) * _NUTRI_WEIGHTS["carbs"]
    ) * 100.0

    return max(0.0, min(100.0, score))


# Compute the Greendle Health Score for a recipe.
# Returns a score from 1.0 to 10.0.
# Formula: greendle_raw = nutri_0_100 × 0.60 + nova_0_100 × 0.40
#   score_1_10 = 1 + (greendle_raw / 100) × 9  → rounded to 1 dp
#   nova_0_100 = 100 × (1 – avg_penalty / 30)  where avg_penalty is gram-weighted across all ingredients (0–30).
def recipe_nutriscore(recipe, nutrition_db, nova_db=None):
    ingredients = recipe.get("extendedIngredients", [])
    servings    = max(1, recipe.get("servings", 1))

    # Nutrient accumulators
    total_g = 0.0
    acc = {k: 0.0 for k in ("energy", "fat", "sat_fat", "sugars", "sodium", "protein", "carbs")}
    ingredient_names = []

    for ing in ingredients:
        name = ing.get("name") or ing.get("nameClean") or ""
        if not name:
            continue

        grams  = _ingredient_grams(ing)
        factor = grams / 100.0
        ingredient_names.append(name)

        row = _match_ingredient(name, nutrition_db)
        if row is not None:
            total_g        += grams
            acc["energy"]  += row["Caloric Value"]  * factor
            acc["fat"]     += row["Fat"]            * factor
            acc["sat_fat"] += row["Saturated Fats"] * factor
            acc["sugars"]  += row["Sugars"]         * factor
            acc["sodium"]  += row["Sodium"]         * factor
            acc["protein"] += row["Protein"]        * factor
            acc["carbs"]   += row["Carbohydrates"]  * factor

    # ── Nutri component (0–100) ──────────────────────────────────────────────
    nutri_raw = _nutri_score_0_100(
        acc["energy"], acc["fat"], acc["sat_fat"], acc["sugars"],
        acc["sodium"], acc["protein"], acc["carbs"], servings,
    ) if total_g > 0 else 50.0

    # ── NOVA component via keyword system (0–100) ────────────────────────────
    kw       = apply_keyword_penalties(recipe.get("title", ""), ingredient_names)
    nova_raw = max(0.0, min(100.0, 100.0 - kw["keyword_penalty"] * 2.5))

    # ── Greendle score: 60% nutri + 40% NOVA → scaled to 1–10 ───────────────
    greendle_raw = nutri_raw * 0.60 + nova_raw * 0.40
    score        = round(1.0 + (greendle_raw / 100.0) * 9.0, 1)
    score        = max(1.0, min(10.0, score))

    return score

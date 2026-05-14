# app.py — Main Streamlit application for Greendle.
#
# Greendle is a smart meal planner that helps users reduce food waste by
# suggesting recipes based on ingredients they already have at home.
# It learns from user ratings to improve recommendations over time.
#
# Pages:
#   Home         — problem statement and app overview
#   My Profile   — dietary restrictions, cuisine preferences and allergies
#   Find Recipes — ingredient-based recipe search with ML ranking
#   My Dashboard — health score visualisation for cooked meals
#   For You      — personalised recommendations driven by past ratings

#Imports:
import json # let us save Python data into text files and read it back later.
import os # build file paths in a way that works on Windows, Mac and Linux.
import re # lets us search and replace patterns in text, used to convert °F to °C in instructions.
import streamlit as st # "as st" gives streamlit the nickname st, so we can write st.button(...) instead of streamlit.button(...)
import plotly.graph_objects as go # same: go = plotly.graph_objects, used to draw the bar chart on the dashboard.

# Internal project modules:
from api import search_by_ingredients, get_recipe_by_id, APIKeyMissingError, APIError # We borrow the two search functions and two error types so we can show a nice message instead of a crash when something goes wrong.
from recommender import rank_recipes
from nutriscore import load_nutrition_db, load_nova_db, recipe_nutriscore, LETTER_COLOR, LETTER_LABEL # We use it on the dashboard to show a score from A to E.

# ── File paths for persistent storage───────────────────────────────────────────────
# Data is stored as JSON files so it survives browser refreshes and app restarts.
# os.path.dirname(__file__) returns the folder this app.py lives in, so paths work no matter where the app is launched from.
RATINGS_FILE = os.path.join(os.path.dirname(__file__), "data", "ratings.json")
CACHE_FILE   = os.path.join(os.path.dirname(__file__), "data", "recipe_cache.json")
PROFILE_FILE = os.path.join(os.path.dirname(__file__), "data", "profile.json")

# Two small reusable functions for reading and writing JSON files.
# Used everywhere we save/load ratings, profile, and the recipe cache.
def load_json(path):
    """Load a JSON file and return its contents, or an empty dict if missing/corrupt."""
    try:
        with open(path) as f:
            return json.load(f)  # json.load reads the text and converts it back into a Python dict.
    except (FileNotFoundError, json.JSONDecodeError):
        return {} # Return an empty dict so the app can keep running as if no data was saved.


def save_json(path, data):
    """Serialise data to a JSON file, overwriting any existing content."""
    with open(path, "w") as f: # "w" = write mode. This also erases the file's previous content, which is fine because we always overwrite with the latest version.
        json.dump(data, f)  # json.dump does the opposite of json.load: dict → JSON text in the file


# ── Session state initialisation ───────────────────────────────────────────────
# Streamlit reruns the script on every interaction, so we load persistent data 
# from disk into session_state once per browser session to avoid repeated file reads.
if "ratings" not in st.session_state: 
    st.session_state["ratings"] = load_json(RATINGS_FILE) # load user's past star ratings from disk
if "recipe_cache" not in st.session_state:  
    st.session_state["recipe_cache"] = load_json(CACHE_FILE) # load recipes already fetched from Spoonacular (avoids re-fetching)
if "profile" not in st.session_state:  
    st.session_state["profile"] = load_json(PROFILE_FILE) # load the user's dietary preferences

LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png") # If the file doesn't exist on disk, a text fallback is shown instead 

# ── Page configuration ─────────────────────────────────────────────────────────
# Sets the browser tab title, the small leaf icon next to it, and tells Streamlit to use the full width of the window
st.set_page_config(
    page_title="Greendle",
    page_icon="🌿",
    layout="wide"
)

# ── Global CSS styling ─────────────────────────────────────────────────────────
# Custom fonts and colours are injected via a <style> block to match the
# Greendle brand identity (cream background, forest green accents, serif headings).
# st.markdown normally shows text, but with unsafe_allow_html=True we can also inject raw HTML 
# The whole CSS is wrapped in triple quotes (""") so we can write it across many lines.
st.markdown(""" 
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400;1,600;1,700&family=Dancing+Script:wght@600;700&family=Lato:wght@300;400;700&display=swap');

html, body, [data-testid="stApp"], .main, [data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"] {
    background-color: #EDEAE3 !important;
    font-family: 'Lato', sans-serif !important;
}

#MainMenu {visibility: hidden;} footer {visibility: hidden;}
[data-testid="stDecoration"] {display: none;}
[data-testid="collapsedControl"] {visibility: visible !important;}

.block-container { padding-top: 1.5rem; max-width: 900px; }

section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
}

.stButton > button {
    background-color: #3A6B3A !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Lato', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-size: 0.82rem !important;
    padding: 0.6rem 1.8rem !important;
    transition: background-color 0.2s !important;
}
.stButton > button:hover {
    background-color: #2D5429 !important;
    color: white !important;
}

div[data-testid="stExpander"] {
    background-color: white !important;
    border-radius: 12px !important;
    border: 1px solid #E0DAD0 !important;
}

div[data-testid="stMetric"] {
    background-color: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #E0DAD0;
}

div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
    border-radius: 8px !important;
    background-color: white !important;
    border: 1px solid #CFC9BE !important;
    font-family: 'Lato', sans-serif !important;
}

.stSelectSlider, .stSlider { font-family: 'Lato', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ─────────────────────────────────────────────────────────
# Show logo image if it exists, otherwise fall back to a styled text header.
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, width=160)
else:
    st.sidebar.markdown("""
    <div style="padding: 0.5rem 0 1.5rem 0;">
        <span style="font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:700; color:#1A2D3D;">Greendle</span><span style="color:#3A6B3A; font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:700;">.</span>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True) # Empty div used purely as vertical spacing between the logo and the menu below.

# _nav_target allows buttons on any page to programmatically switch the active page
_pages = ["Home", "My Profile", "Find Recipes", "My Dashboard", "For You"]
_nav_index = _pages.index(st.session_state.pop("_nav_target", "Home"))
page = st.sidebar.radio(
    "Navigate",
    _pages,
    index=_nav_index,   # which page should be selected when the sidebar loads.
    label_visibility="collapsed" # hide the "Navigate" label since the options are self-explanatory
)


# ── Reusable UI helpers ────────────────────────────────────────────────────────
# These are shortcut functions for HTML/CSS we'd otherwise have to copy-paste
# everywhere. Defining them once keeps the page code shorter and easiert to read.

def section_heading(title):
    """Render a styled section heading with a green underline accent."""
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-family:'Playfair Display',serif; font-style:italic; font-weight:600;
                   color:#3A6B3A; font-size:1.65rem; margin-bottom:0.4rem;">{title}</h2>
        <div style="width:36px; height:2px; background:#3A6B3A; border-radius:2px;"></div>
    </div>
    """, unsafe_allow_html=True)


def card(content_html):
    """Wrap arbitrary HTML in a white rounded card with a subtle shadow."""
    st.markdown(f"""
    <div style="background:white; border-radius:14px; padding:1.6rem;
                box-shadow:0 2px 12px rgba(58,107,58,0.08); margin-bottom:1rem;
                border:1px solid #E8E4DC;">
        {content_html}
    </div>
    """, unsafe_allow_html=True)

# ── Halal filtering ────────────────────────────────────────────────────────────────────
# When the user checks "Halal" in their profile, we hide recipes containing non halal-products

_NON_HALAL_KEYWORDS = { 
    "pork", "bacon", "ham", "lard", "prosciutto", "pancetta", "chorizo",
    "salami", "pepperoni", "wine", "beer", "ale", "rum", "vodka",
    "whiskey", "whisky", "brandy", "liqueur", "sake", "champagne",
    "gin", "tequila", "bourbon", "alcohol", "spirits",
} # This is a simple keyword matching approach

def non_halal_label(recipe):
    """ Return the HTML badge in red for any halal ingredient present in the recipe; return an empty 
    string if there are no such ingredients."""
    names = [i.get("name", "").lower() for i in recipe.get("extendedIngredients", [])]
    if any(kw in name for name in names for kw in _NON_HALAL_KEYWORDS):  # Build a list of all ingredient names in lowercase, example: Pork =porc.
        return "<span style='color:#DC2626; font-size:0.85rem;'>🚫 Non-halal</span>"
    return ""


# Convert all Fahrenheit temperatures in a text string to Celsius.
def convert_f_to_c(text):
    pattern = r'(\d+(?:\.\d+)?)\s*(?:°\s*[Ff]|[Ff]ahrenheit|degrees?\s+[Ff](?:ahrenheit)?)\b'
    def replacer(match):
        c = round((float(match.group(1)) - 32) * 5 / 9)
        return f"{c}°C"
    return re.sub(pattern, replacer, text)


_KEEP_UNITS = {"tsp", "tsps", "teaspoon", "teaspoons", "tbsp", "tbsps", "tablespoon", "tablespoons"}

# Format an ingredient using metric units, keeping tsp and tbsp as-is.
def format_ingredient_metric(ing):
    original = ing.get("original", "")
    us = ing.get("measures", {}).get("us", {})
    us_unit = (us.get("unitShort") or us.get("unitLong") or "").lower().strip()

    if us_unit in _KEEP_UNITS:
        return original

    metric = ing.get("measures", {}).get("metric", {})
    m_amount = metric.get("amount")
    m_unit = (metric.get("unitShort") or metric.get("unitLong") or "").strip()
    name = ing.get("name", "")

    if m_amount and m_unit and name:
        if float(m_amount) == int(float(m_amount)):
            amount_str = str(int(float(m_amount)))
        else:
            amount_str = f"{float(m_amount):.1f}"
        return f"{amount_str} {m_unit} {name}"

    return original


# ── Page: Home ─────────────────────────────────────────────────────────────────
if page == "Home":
    # Hero title
    st.markdown("""
    <div style="text-align:center; padding:2.5rem 0 1.5rem 0;">
        <span style="font-family:'Playfair Display',serif; font-size:4rem; font-weight:800;
                     color:#1A2D3D; line-height:1; letter-spacing:-0.01em;">Greendle</span><span
             style="font-family:'Playfair Display',serif; font-size:4rem; font-weight:800;
                    color:#4A6741; line-height:1;">.</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Problem and solution summary cards
    col1, col2 = st.columns(2) # creates two side-by-side regions of equal width.
    with col1:
        st.markdown("""
        <div style="background:white; border-radius:14px; padding:1.6rem;
                    box-shadow:0 2px 12px rgba(58,107,58,0.08); border:1px solid #E8E4DC; height:100%;">
            <div style="font-family:'Playfair Display',serif; font-style:italic; font-weight:600;
                        color:#3A6B3A; font-size:1.25rem; margin-bottom:0.9rem;">The problem</div>
            <div style="font-family:'Lato',sans-serif; color:#4B5563; line-height:1.75; font-size:0.95rem;">
                The average household throws away 30% of their food. That's not a statistic.
                That's your Tuesday leftovers, your forgotten herbs, and that one lonely
                courgette at the back of the fridge.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:white; border-radius:14px; padding:1.6rem;
                    box-shadow:0 2px 12px rgba(58,107,58,0.08); border:1px solid #E8E4DC; height:100%;">
            <div style="font-family:'Playfair Display',serif; font-style:italic; font-weight:600;
                        color:#3A6B3A; font-size:1.25rem; margin-bottom:0.9rem;">Our Solution</div>
            <div style="font-family:'Lato',sans-serif; color:#4B5563; line-height:1.75; font-size:0.95rem;">
                Greendle takes what you already have, turns it into meal ideas you'll actually
                want to eat, and tells you how healthy they are.
                <strong style="color:#1A2D3D;">Less waste, better meals, zero guilt.</strong>
                Cook smarter. Eat better. Waste nothing.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_heading("How it works")

    # Three feature cards explaining the main app flows
    c1, c2, c3 = st.columns(3)
    features = [
        (c1,
         "Add your ingredients",
         "Tell us what's hiding in your fridge, freezer or cupboard. Yes, that half onion "
         "counts. Whatever you have, Greendle finds something delicious to cook with it."),
        (c2,
         "Cook a recipe",
         "Your personal dashboard updates automatically — tracking how healthy each of your "
         "meals are, so you can eat better without even thinking about it."),
        (c3,
         "Let Greendle learn your taste",
         "Browse recipes picked just for you. The more you use it, the more it gets you. "
         "Like a friend who really, really knows your fridge."),
    ]
    for col, title, desc in features:  # Each iteration col, title, desc are unpacked from one tuple in featurs.
        with col:
            st.markdown(f"""
            <div style="background:white; border-radius:14px; padding:1.5rem;
                        box-shadow:0 2px 12px rgba(58,107,58,0.08); border:1px solid #E8E4DC;
                        height:100%;">
                <div style="font-family:'Lato',sans-serif; font-weight:700; color:#1A2D3D;
                            margin-bottom:0.5rem; font-size:0.97rem;">{title}</div>
                <div style="font-family:'Lato',sans-serif; color:#6B7280; font-size:0.88rem;
                            line-height:1.65;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding:2.5rem 0 1rem 0;">
        <span style="font-family:'Dancing Script',cursive; font-size:1.6rem; font-weight:700;
                     color:#1A2D3D;">Everything starts in the sidebar, dinner won't make itself!</span>
    </div>
    """, unsafe_allow_html=True)


# ── Page: My Profile ───────────────────────────────────────────────────────────
# Allows saving of user's diatary preferences. Streamlit has a strict rule: 
# once you set a value for your widget it cannot be changed on the same run of the file. 
# Below are several ways to get around this issue.
elif page == "My Profile":
    section_heading("My Profile")
    st.markdown("<p style='color:#6B7280; margin-top:-1rem; margin-bottom:1.5rem;'>Tell us about your preferences so we can personalise your recipes.</p>", unsafe_allow_html=True)

    # Load the currently saved profile so fields are pre-filled on revisit.
    p = st.session_state.get("profile", {})

    # It’s impossible to set the values of those checkboxes after pressing “None of the above”  since Streamlit will refuse to do so after the elements are created. 
    # The solution: when pressing this button, we put a flag which make us run the whole script again. 
    # On the following script launch, before drawing the checkboxes, we check this flag, reset all the six checkboxes, then delete the flag with pop().
    restriction_keys = ["pref_vegan", "pref_vegetarian", "pref_gluten_free",
                        "pref_dairy_free", "pref_nut_free", "pref_halal"]
    if st.session_state.pop("do_clear", False):
        for k in restriction_keys:
            st.session_state[k] = False

    # Initialise widget session state from saved profile on first load only.
    # Once the user interacts with a widget its key is already in session_state,
    # so the default is only applied on a fresh page load / after page refresh.
    defaults = {
        "pref_vegan":       p.get("vegan", False),
        "pref_vegetarian":  p.get("vegetarian", False),
        "pref_gluten_free": p.get("gluten_free", False),
        "pref_dairy_free":  p.get("dairy_free", False),
        "pref_nut_free":    p.get("nut_free", False),
        "pref_halal":       p.get("halal", False),
        "pref_spice":       p.get("spice_level", 2),
        "pref_allergies":   ", ".join(p.get("allergies", [])), # Allergies are saved as a listso we glue them back into one string because the widget is a text box.
    }
    for k, v in defaults.items(): # this block runs on every rerun, but we only want to set the default the very first time. Without the guard, every click would overwrite the user's unsaved changes.
        if k not in st.session_state:
            st.session_state[k] = v

    # Dietary restriction checkboxes. Each tied to a session state key
    st.markdown("**Dietary Restrictions**")
    col1, col2 = st.columns(2)
    with col1:
        st.checkbox("Vegan",       key="pref_vegan")
        st.checkbox("Vegetarian",  key="pref_vegetarian")
        st.checkbox("Gluten-free", key="pref_gluten_free")
    with col2:
        st.checkbox("Dairy-free",  key="pref_dairy_free")
        st.checkbox("Nut-free",    key="pref_nut_free")
        st.checkbox("Halal",       key="pref_halal")

    # Sets a flag so all checkboxes are reset at the top of the next rerun.
    if st.button("None of the above — clear all restrictions"):
        st.session_state["do_clear"] = True
        st.rerun()

    st.markdown("<br>**Taste Preferences**", unsafe_allow_html=True)
    cuisine = st.multiselect(
        "Favourite cuisines",
        ["Italian", "Mexican", "Asian", "Mediterranean", "American", "French", "Indian"],
        default=p.get("cuisine", [])
    )
    st.slider("Spice tolerance", 0, 5, key="pref_spice", help="0 = no spice, 5 = very spicy")

    st.markdown("<br>**Allergies**", unsafe_allow_html=True)
    st.text_input("List any allergies (comma-separated)", key="pref_allergies", placeholder="e.g. peanuts, shellfish")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Save Profile"):
        # Collect all widget values and persist to both session state and disk
        st.session_state["profile"] = {
            "vegan":       st.session_state["pref_vegan"],
            "vegetarian":  st.session_state["pref_vegetarian"],
            "gluten_free": st.session_state["pref_gluten_free"],
            "dairy_free":  st.session_state["pref_dairy_free"],
            "nut_free":    st.session_state["pref_nut_free"],
            "halal":       st.session_state["pref_halal"],
            "cuisine":     cuisine,
            "spice_level": st.session_state["pref_spice"],
            "allergies":   [a.strip() for a in st.session_state["pref_allergies"].split(",") if a.strip()],
        }
        save_json(PROFILE_FILE, st.session_state["profile"])
        st.success("Profile saved!")


# ── Page: Find Recipes ─────────────────────────────────────────────────────────
elif page == "Find Recipes":
    section_heading("Find Recipes")
    st.markdown("<p style='color:#6B7280; margin-top:-1rem; margin-bottom:1.5rem;'>Enter the ingredients you have and we'll find matching recipes.</p>", unsafe_allow_html=True)

    ingredients_input = st.text_input( # Returns whatever the user types
        "Ingredients (comma-separated)",
        placeholder="e.g. chicken, garlic, lemon"
    )
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Search Recipes"):  # Everything inside this if-block only runs on the click
        if not ingredients_input:
            st.warning("Please enter at least one ingredient.")
        else:
            # Parse the comma-separated input into a clean list
            ingredients = [i.strip() for i in ingredients_input.split(",") if i.strip()]

            # Apply dietary filters from the saved user profile
            profile = st.session_state.get("profile", {})
            diet = None
            if profile.get("vegan"):        diet = "vegan"
            elif profile.get("vegetarian"): diet = "vegetarian"

            intolerances = []
            if profile.get("gluten_free"): intolerances.append("gluten")
            if profile.get("dairy_free"):  intolerances.append("dairy")
            if profile.get("nut_free"):    intolerances.append("tree nut")

            try:
                with st.spinner("Searching recipes..."): # shows a loading indicator while the API call is running.
                    recipes = search_by_ingredients(
                        ingredients, diet=diet, intolerances=intolerances or None # The API function expects either None or a non-empty list.

                    )
            except APIKeyMissingError:
                # Triggered when .env is missing or the key is invalid.
                # Show step-by-step setup instructions and halt the script.
                st.error(
                    "**Spoonacular API key not configured.**\n\n"
                    "1. Get a free key at [spoonacular.com/food-api](https://spoonacular.com/food-api)\n"
                    "2. Create a `.env` file in the project folder:\n"
                    "   ```\n   SPOONACULAR_KEY=your_key_here\n   ```\n"
                    "3. Restart the app."
                )
                st.stop()
            except APIError as e:
                # Any other API failure like quota exceeded, server down, etc
                st.error(str(e))
                st.stop()

            if not recipes: # If the list is empty, warn the user.
                st.warning("No recipes found for those ingredients. Try something more common like chicken, garlic, or pasta.")
            else:
                ratings = st.session_state.get("ratings", {})

                # Rank recipes using TF-IDF cosine similarity + rating boost (ML step)
                ranked = rank_recipes(recipes, ingredients, ratings)

                # Fetch full recipe details for each result and cache them to disk.
                # This avoids repeated API calls on subsequent page visits.
                with st.spinner("Loading recipe details..."):
                    full_recipes = []
                    recipe_cache = st.session_state["recipe_cache"]
                    for r in ranked:
                        rid = str(r["id"]) # use string IDs because JSON keys must be strings.
                        if rid not in recipe_cache:
                            full = get_recipe_by_id(r["id"]) # If the detail call returned nothing, keep the partial recipe so the user still sees something.
                            recipe_cache[rid] = full if full else r
                        full_recipes.append(recipe_cache[rid])
                    st.session_state["recipe_cache"] = recipe_cache
                    save_json(CACHE_FILE, recipe_cache)  # persist to disk

                # Halal filter: if the user checked Halal in their profil,
                # drop any recipe whose ingredients contain a forbidden keyword.
                if st.session_state.get("profile", {}).get("halal"): 
                    full_recipes = [r for r in full_recipes if not non_halal_label(r)]

                st.session_state["search_results"] = full_recipes
                st.session_state["search_ingredients"] = ingredients

    # This block sits outside the Search button if-statement on purpose: it
    # runs on every rerun, so results stay visible after clicks like Save Rating.
    if "search_results" in st.session_state:
        results = st.session_state["search_results"]

        def has_instructions(r):
            if (r.get("instructions") or "").strip():
                return True
            analyzed = r.get("analyzedInstructions", [])
            return any(s for section in analyzed for s in section.get("steps", []))
         
        # Filter out recipes with no instructions as they're not cookable.
        results = [r for r in results if has_instructions(r)]
        st.markdown(f"<p style='color:#3A6B3A; font-weight:600; margin-bottom:1rem;'>Found {len(results)} recipes</p>", unsafe_allow_html=True)

        for recipe in results:
            # Show recipe thumbnail and summary side by side
            col1, col2 = st.columns([1, 3])
            with col1:
                if recipe.get("image"):
                    st.image(recipe["image"], width=150)
            with col2:
                st.markdown(f"<div style='font-size:1.1rem; font-weight:700; color:#1A2D3D; margin-bottom:0.25rem;'>{recipe['title']}</div>", unsafe_allow_html=True)
                ready_in = recipe.get("readyInMinutes", "?")  # Cooking time + servings
                servings = recipe.get("servings", "?")
                st.markdown(f"<span style='color:#6B7280; font-size:0.88rem;'>⏱ {ready_in} min &nbsp;·&nbsp; 🍽 {servings} servings</span>", unsafe_allow_html=True)
                diets = recipe.get("diets", [])
                if diets:
                    st.markdown(f"<span style='color:#3A6B3A; font-size:0.85rem;'>🥗 {', '.join(d.capitalize() for d in diets[:3])}</span>", unsafe_allow_html=True) # diets[:3] = take the first 3 only (Spoonacular sometimes returns 10+)
                label = non_halal_label(recipe)
                if label:
                    st.markdown(label, unsafe_allow_html=True)

            # Full recipe details in a collapsible expander
            with st.expander("View full recipe"):
                # Ingredients as a bulleted list.
                st.markdown("**Ingredients**")
                for ing in recipe.get("extendedIngredients", []):
                    st.markdown(f"- {format_ingredient_metric(ing)}")

                st.markdown("**Instructions**")
                instructions = (recipe.get("instructions") or "").strip()
                if not instructions:
                    analyzed = recipe.get("analyzedInstructions", [])
                    steps = [s for section in analyzed for s in section.get("steps", [])]
                    instructions = "\n\n".join(f"**{s['number']}.** {s['step']}" for s in steps)
                st.markdown(convert_f_to_c(instructions), unsafe_allow_html=True)

                st.markdown("---")
                # Each recipe needs its own session_state slot for its slider.
                # The f-string make each key unique by injecting the recipe ID.
                # Without unique keys, all sliders would collide and Streamlit would crash.
                rating_key = f"rating_{recipe['id']}"
                rating = st.slider("Rate this recipe ⭐", 1, 5, 3, key=rating_key)
                if st.button("Save Rating", key=f"save_{recipe['id']}"):
                    st.session_state["ratings"][str(recipe["id"])] = rating
                    save_json(RATINGS_FILE, st.session_state["ratings"]) # This rating will boost similar recipes on the user's next search.
                    st.success("Rating saved!")

            st.markdown("<hr style='border:none; border-top:1px solid #E5E7EB; margin:0.5rem 0;'>", unsafe_allow_html=True)


# ── Page: My Dashboard ─────────────────────────────────────────────────────────
elif page == "My Dashboard":
    section_heading("My Health Dashboard")
    st.markdown("<p style='color:#6B7280; margin-top:-1rem; margin-bottom:1.5rem;'>The Greendle Score cuts through diet culture : it's a rating out of 10 that tells you what your body actually thrives on, not what shrinks your waistline. By combining the nutritional quality of your ingredients with how processed they are, it gives you a honest picture of how well a meal truly fuels, repairs, and energizes your body from the inside out.</p>", unsafe_allow_html=True)

    ratings      = st.session_state.get("ratings", {})
    results      = st.session_state.get("search_results", [])
    recipe_cache = st.session_state.get("recipe_cache", {})

    if not ratings:
        st.markdown('<div style="background:white;border-radius:16px;padding:1.5rem;box-shadow:0 2px 10px rgba(0,0,0,0.06);text-align:center;color:#6B7280;">Your stats will appear here once you rate some recipes.</div>', unsafe_allow_html=True)
    else:
        # Load nutrition databases (cached by Streamlit after first load)
        with st.spinner("Loading nutrition database…"):
            nutrition_db = load_nutrition_db()
            nova_db      = load_nova_db()

        # Merge search results and persistent cache so rated recipes are always found
        recipe_lookup = {str(r["id"]): r for r in results}
        recipe_lookup.update(recipe_cache)

        # Compute Greendle Health Score for each rated recipe
        labels, letters, grades = [], [], []
        for recipe_id in ratings:
            recipe = recipe_lookup.get(recipe_id)
            if not recipe:
                continue
            # recipe_nutriscore returns a letter (A-E) and a numeric score (1-10)
            letter, grade = recipe_nutriscore(recipe, nutrition_db, nova_db)
            short_title = recipe["title"][:25] + ("…" if len(recipe["title"]) > 25 else "")
            labels.append(short_title)
            letters.append(letter)
            grades.append(grade)

        if not labels:
            st.info("Recipe details not found in cache — try searching for a recipe first.")
        else:
            total_meals   = len(labels)
            overall_score = round(sum(grades) / len(grades), 1)

            # Map overall numeric score to letter grade for the summary card
            if overall_score >= 8.0:   overall_letter = "A"
            elif overall_score >= 6.0: overall_letter = "B"
            elif overall_score >= 4.0: overall_letter = "C"
            elif overall_score >= 2.0: overall_letter = "D"
            else:                      overall_letter = "E"

            score_color = LETTER_COLOR[overall_letter]

            # Summary metrics row
            c1, c2 = st.columns(2)
            c1.metric("Meals cooked", total_meals)
            with c2:
                st.markdown(f'<div style="background:white;border-radius:14px;padding:1.2rem 1.5rem;border:1px solid #E5E7EB;"><div style="font-size:0.85rem;color:#6B7280;margin-bottom:0.4rem;">Greendle Health Score</div><div style="display:flex;align-items:baseline;gap:0.3rem;"><span style="font-size:2rem;font-weight:800;color:{score_color};">{overall_score}</span><span style="font-size:1rem;color:#9CA3AF;">&thinsp;/ 10</span></div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Bar chart: one bar per rated recipe, coloured by health letter grade
            fig = go.Figure(go.Bar(
                x=labels, y=grades,
                marker_color=[LETTER_COLOR[l] for l in letters],
                marker_line_width=0,
                text=[str(g) for g in grades], textposition="outside",
                textfont=dict(color="#1A2D3D", size=13),
            ))
            fig.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter, sans-serif", color="#1A2D3D"),
                yaxis=dict(title="Greendle Score (/ 10)", range=[0, 12], gridcolor="#F0F0F0", zeroline=False),
                xaxis=dict(title="Recipe", tickangle=-20),
                margin=dict(t=20, b=60, l=40, r=20), height=380,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<p style='color:#9CA3AF; font-size:0.8rem;'>Greendle Score = 60% nutritional quality (calories, fat, saturated fat, sugar, sodium, protein, carbs) + 40% food processing level (NOVA 1 = unprocessed → NOVA 4 = ultra-processed).</p>", unsafe_allow_html=True)


# ── Page: For You ──────────────────────────────────────────────────────────────
elif page == "For You":
    section_heading("For You")
    st.markdown("<p style='color:#6B7280; margin-top:-1rem; margin-bottom:1.5rem;'>Personalised suggestions based on your taste profile, built from your star ratings.</p>", unsafe_allow_html=True)

    ratings      = st.session_state.get("ratings", {})
    recipe_cache = st.session_state.get("recipe_cache", {})

    if not ratings:
        # Prompt the user to rate recipes before recommendations can be generated
        st.markdown('<div style="background:white;border-radius:16px;padding:2rem;text-align:center;box-shadow:0 2px 10px rgba(0,0,0,0.06);"><div style="font-size:2.5rem;margin-bottom:0.6rem;">⭐</div><div style="font-weight:700;color:#1A2D3D;font-size:1.05rem;margin-bottom:0.4rem;">No ratings yet</div><div style="color:#6B7280;font-size:0.9rem;line-height:1.6;">Head to <strong>Find Recipes</strong>, cook something, and rate it — we\'ll use those stars to find what you\'ll love next.</div></div>', unsafe_allow_html=True)
    else:
        # Build a taste profile by extracting ingredients from the user's top-3 rated recipes
        top_rated = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:3]
        seed_ingredients = []
        for rid, _ in top_rated:
            recipe = recipe_cache.get(rid, {})
            for ing in recipe.get("extendedIngredients", []):
                name = ing.get("name", "").strip().lower()
                # Avoid duplicates in the seed list
                if name and name not in seed_ingredients:
                    seed_ingredients.append(name)

        # Limit to 10 seed ingredients to keep the API query focused
        seed_ingredients = seed_ingredients[:10]

        if not seed_ingredients:
            st.info("We need more recipe details to make recommendations. Try searching for a few recipes first.")
        else:
            # Use a hash of the current ratings as a cache key.
            # Recommendations are only recomputed when ratings change.
            ratings_key = hash(frozenset(ratings.items()))
            if st.session_state.get("_recs_key") != ratings_key:
                with st.spinner("Finding recipes you'll love..."):
                    # Search the API using the taste-profile ingredients
                    candidates = search_by_ingredients(seed_ingredients, number=8)
                if candidates:
                    # Re-rank candidates using the same ML model (TF-IDF + rating boost)
                    ranked = rank_recipes(candidates, seed_ingredients, ratings)
                    with st.spinner("Loading details..."):
                        full = []
                        for r in ranked:
                            ckey = str(r["id"])
                            if ckey not in recipe_cache:
                                fetched = get_recipe_by_id(r["id"])
                                recipe_cache[ckey] = fetched if fetched else r
                                save_json(CACHE_FILE, recipe_cache)
                            full.append(recipe_cache[ckey])
                    if st.session_state.get("profile", {}).get("halal"):
                        full = [r for r in full if not non_halal_label(r)]
                    st.session_state["_recs"]     = full
                    st.session_state["_recs_key"] = ratings_key

            recs = st.session_state.get("_recs", [])

            # Refresh button clears the cache key, forcing a new recommendation run
            col_refresh, _ = st.columns([2, 4])
            with col_refresh:
                if st.button("↻ Refresh", use_container_width=True):
                    st.session_state.pop("_recs_key", None)
                    st.rerun()

            n_rated = len(ratings)
            st.markdown(f"<p style='color:#3A6B3A; font-weight:600; margin-bottom:1.2rem;'>Based on {n_rated} recipe{'s' if n_rated != 1 else ''} you rated</p>", unsafe_allow_html=True)

            # Display each recommended recipe
            for recipe in recs:
                col1, col2 = st.columns([1, 3])
                with col1:
                    if recipe.get("image"):
                        st.image(recipe["image"], width=150)
                with col2:
                    st.markdown(f"<div style='font-size:1.05rem;font-weight:700;color:#1A2D3D;margin-bottom:0.25rem;'>{recipe['title']}</div>", unsafe_allow_html=True)
                    ready_in = recipe.get("readyInMinutes", "?")
                    servings = recipe.get("servings", "?")
                    st.markdown(f"<span style='color:#6B7280;font-size:0.88rem;'>⏱ {ready_in} min &nbsp;·&nbsp; 🍽 {servings} servings</span>", unsafe_allow_html=True)
                    diets = recipe.get("diets", [])
                    if diets:
                        st.markdown(f"<span style='color:#89cd8f;font-size:0.85rem;'>🥗 {', '.join(d.capitalize() for d in diets[:3])}</span>", unsafe_allow_html=True)
                    label = non_halal_label(recipe)
                    if label:
                        st.markdown(label, unsafe_allow_html=True)
                st.markdown("<hr style='border:none;border-top:1px solid #E5E7EB;margin:0.5rem 0;'>", unsafe_allow_html=True)

            # Algorithm explainer shown at the bottom for transparency
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div style="background:#F9FAFB;border-radius:12px;padding:1rem 1.3rem;border:1px solid #E5E7EB;font-size:0.8rem;color:#9CA3AF;line-height:1.6;"><strong style="color:#6B7280;">How this works</strong><br>Your top-rated recipes\' ingredients are extracted and used as a taste profile. A TF-IDF vectoriser converts ingredient lists into numerical vectors, and cosine similarity ranks new recipes by how closely they match your profile — weighted by your star ratings.</div>', unsafe_allow_html=True)

import json
import os
import streamlit as st
import plotly.graph_objects as go
from api import search_by_ingredients, get_recipe_by_id, APIKeyMissingError, APIError
from recommender import rank_recipes
from cf_recommender import load_cf_model, get_cf_recommendations, model_pickle_exists
from nutriscore import load_nutrition_db, load_nova_db, recipe_nutriscore, LETTER_COLOR, LETTER_LABEL

RATINGS_FILE = os.path.join(os.path.dirname(__file__), "data", "ratings.json")
CACHE_FILE   = os.path.join(os.path.dirname(__file__), "data", "recipe_cache.json")
PROFILE_FILE = os.path.join(os.path.dirname(__file__), "data", "profile.json")

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


@st.cache_resource(show_spinner=False)
def _cf_model():
    """Loaded once per server process; shared across all sessions."""
    return load_cf_model()


# Load persisted data into session state once per session
if "ratings" not in st.session_state:
    st.session_state["ratings"] = load_json(RATINGS_FILE)
if "recipe_cache" not in st.session_state:
    st.session_state["recipe_cache"] = load_json(CACHE_FILE)
if "profile" not in st.session_state:
    st.session_state["profile"] = load_json(PROFILE_FILE)

LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png")

st.set_page_config(
    page_title="Greendle",
    page_icon="🌿",
    layout="wide"
)

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

# ── Sidebar ───────────────────────────────────────────────────
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, width=160)
else:
    st.sidebar.markdown("""
    <div style="padding: 0.5rem 0 1.5rem 0;">
        <span style="font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:700; color:#1A2D3D;">Greendle</span><span style="color:#3A6B3A; font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:700;">.</span>
    </div>
    """, unsafe_allow_html=True)
st.sidebar.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

_pages = ["Home", "My Profile", "Find Recipes", "My Dashboard", "For You"]
_nav_index = _pages.index(st.session_state.pop("_nav_target", "Home"))
page = st.sidebar.radio(
    "Navigate",
    _pages,
    index=_nav_index,
    label_visibility="collapsed"
)


def section_heading(title):
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-family:'Playfair Display',serif; font-style:italic; font-weight:600;
                   color:#3A6B3A; font-size:1.65rem; margin-bottom:0.4rem;">{title}</h2>
        <div style="width:36px; height:2px; background:#3A6B3A; border-radius:2px;"></div>
    </div>
    """, unsafe_allow_html=True)


def card(content_html):
    st.markdown(f"""
    <div style="background:white; border-radius:14px; padding:1.6rem;
                box-shadow:0 2px 12px rgba(58,107,58,0.08); margin-bottom:1rem;
                border:1px solid #E8E4DC;">
        {content_html}
    </div>
    """, unsafe_allow_html=True)


# ── Pages ─────────────────────────────────────────────────────

if page == "Home":
    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:2.5rem 0 1.5rem 0;">
        <span style="font-family:'Playfair Display',serif; font-size:4rem; font-weight:800;
                     color:#1A2D3D; line-height:1; letter-spacing:-0.01em;">Greendle</span><span
             style="font-family:'Playfair Display',serif; font-size:4rem; font-weight:800;
                    color:#4A6741; line-height:1;">.</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Start Here CTA ───────────────────────────────────────────────────
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        if st.button("Start here →", use_container_width=True):
            st.session_state["_nav_target"] = "Find Recipes"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Problem / Solution ────────────────────────────────────────────────
    col1, col2 = st.columns(2)
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
    for col, title, desc in features:
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


elif page == "My Profile":
    section_heading("My Profile")
    st.markdown("<p style='color:#6B7280; margin-top:-1rem; margin-bottom:1.5rem;'>Tell us about your preferences so we can personalise your recipes.</p>", unsafe_allow_html=True)

    p = st.session_state.get("profile", {})

    # If clear was requested, reset all restriction keys before widgets are created
    restriction_keys = ["pref_vegan", "pref_vegetarian", "pref_gluten_free",
                        "pref_dairy_free", "pref_nut_free", "pref_halal"]
    if st.session_state.pop("do_clear", False):
        for k in restriction_keys:
            st.session_state[k] = False

    # Initialise widget keys from saved profile only on first load
    defaults = {
        "pref_vegan": p.get("vegan", False),
        "pref_vegetarian": p.get("vegetarian", False),
        "pref_gluten_free": p.get("gluten_free", False),
        "pref_dairy_free": p.get("dairy_free", False),
        "pref_nut_free": p.get("nut_free", False),
        "pref_halal": p.get("halal", False),
        "pref_spice": p.get("spice_level", 2),
        "pref_allergies": ", ".join(p.get("allergies", [])),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

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
        st.session_state["profile"] = {
            "vegan": st.session_state["pref_vegan"],
            "vegetarian": st.session_state["pref_vegetarian"],
            "gluten_free": st.session_state["pref_gluten_free"],
            "dairy_free": st.session_state["pref_dairy_free"],
            "nut_free": st.session_state["pref_nut_free"],
            "halal": st.session_state["pref_halal"],
            "cuisine": cuisine,
            "spice_level": st.session_state["pref_spice"],
            "allergies": [a.strip() for a in st.session_state["pref_allergies"].split(",") if a.strip()],
        }
        save_json(PROFILE_FILE, st.session_state["profile"])
        st.success("Profile saved!")


elif page == "Find Recipes":
    section_heading("Find Recipes")
    st.markdown("<p style='color:#6B7280; margin-top:-1rem; margin-bottom:1.5rem;'>Enter the ingredients you have and we'll find matching recipes.</p>", unsafe_allow_html=True)

    ingredients_input = st.text_input(
        "Ingredients (comma-separated)",
        placeholder="e.g. chicken, garlic, lemon"
    )
    time_limit = st.select_slider(
        "Available cooking time",
        options=[15, 30, 45, 60],
        value=30,
        format_func=lambda x: f"{x} min"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Search Recipes"):
        if not ingredients_input:
            st.warning("Please enter at least one ingredient.")
        else:
            ingredients = [i.strip() for i in ingredients_input.split(",") if i.strip()]

            profile = st.session_state.get("profile", {})
            diet = None
            if profile.get("vegan"):       diet = "vegan"
            elif profile.get("vegetarian"): diet = "vegetarian"

            intolerances = []
            if profile.get("gluten_free"): intolerances.append("gluten")
            if profile.get("dairy_free"):  intolerances.append("dairy")
            if profile.get("nut_free"):    intolerances.append("tree nut")

            try:
                with st.spinner("Searching recipes..."):
                    recipes = search_by_ingredients(
                        ingredients, diet=diet, intolerances=intolerances or None
                    )
            except APIKeyMissingError:
                st.error(
                    "**Spoonacular API key not configured.**\n\n"
                    "1. Get a free key at [spoonacular.com/food-api](https://spoonacular.com/food-api)\n"
                    "2. Create a `.env` file in the project folder:\n"
                    "   ```\n   SPOONACULAR_KEY=your_key_here\n   ```\n"
                    "3. Restart the app."
                )
                st.stop()
            except APIError as e:
                st.error(str(e))
                st.stop()

            if not recipes:
                st.warning("No recipes found for those ingredients. Try something more common like chicken, garlic, or pasta.")
            else:
                ratings = st.session_state.get("ratings", {})
                ranked  = rank_recipes(recipes, ingredients, ratings)

                with st.spinner("Loading recipe details..."):
                    full_recipes = []
                    recipe_cache = st.session_state["recipe_cache"]
                    for r in ranked:
                        rid = str(r["id"])
                        if rid not in recipe_cache:
                            full = get_recipe_by_id(r["id"])
                            recipe_cache[rid] = full if full else r
                        full_recipes.append(recipe_cache[rid])
                    st.session_state["recipe_cache"] = recipe_cache
                    save_json(CACHE_FILE, recipe_cache)

                st.session_state["search_results"] = full_recipes
                st.session_state["search_ingredients"] = ingredients

    if "search_results" in st.session_state:
        results = st.session_state["search_results"]
        st.markdown(f"<p style='color:#3A6B3A; font-weight:600; margin-bottom:1rem;'>Found {len(results)} recipes</p>", unsafe_allow_html=True)

        for recipe in results:
            col1, col2 = st.columns([1, 3])
            with col1:
                if recipe.get("image"):
                    st.image(recipe["image"], width=150)
            with col2:
                st.markdown(f"<div style='font-size:1.1rem; font-weight:700; color:#1A2D3D; margin-bottom:0.25rem;'>{recipe['title']}</div>", unsafe_allow_html=True)
                ready_in = recipe.get("readyInMinutes", "?")
                servings = recipe.get("servings", "?")
                st.markdown(f"<span style='color:#6B7280; font-size:0.88rem;'>⏱ {ready_in} min &nbsp;·&nbsp; 🍽 {servings} servings</span>", unsafe_allow_html=True)
                diets = recipe.get("diets", [])
                if diets:
                    st.markdown(f"<span style='color:#3A6B3A; font-size:0.85rem;'>🥗 {', '.join(d.capitalize() for d in diets[:3])}</span>", unsafe_allow_html=True)

            with st.expander("View full recipe"):
                st.markdown("**Ingredients**")
                for ing in recipe.get("extendedIngredients", []):
                    st.markdown(f"- {ing.get('original', '')}")

                st.markdown("**Instructions**")
                instructions = recipe.get("instructions", "").strip()
                if not instructions:
                    analyzed = recipe.get("analyzedInstructions", [])
                    steps = [s for section in analyzed for s in section.get("steps", [])]
                    if steps:
                        instructions = "\n\n".join(f"**{s['number']}.** {s['step']}" for s in steps)
                    else:
                        instructions = "No instructions available for this recipe."
                st.markdown(instructions, unsafe_allow_html=True)

                st.markdown("---")
                rating_key = f"rating_{recipe['id']}"
                rating = st.slider("Rate this recipe ⭐", 1, 5, 3, key=rating_key)
                if st.button("Save Rating", key=f"save_{recipe['id']}"):
                    st.session_state["ratings"][str(recipe["id"])] = rating
                    save_json(RATINGS_FILE, st.session_state["ratings"])
                    st.success("Rating saved!")

            st.markdown("<hr style='border:none; border-top:1px solid #E5E7EB; margin:0.5rem 0;'>", unsafe_allow_html=True)


elif page == "My Dashboard":
    section_heading("My Health Dashboard")
    st.markdown(
        "<p style='color:#6B7280; margin-top:-1rem; margin-bottom:1.5rem;'>"
        "See how nutritious your cooked meals are, scored with the official FSA Nutri-Score algorithm."
        "</p>",
        unsafe_allow_html=True,
    )

    ratings = st.session_state.get("ratings", {})
    results = st.session_state.get("search_results", [])

    if not ratings:
        st.markdown(
            '<div style="background:white;border-radius:16px;padding:1.5rem;'
            'box-shadow:0 2px 10px rgba(0,0,0,0.06);text-align:center;color:#6B7280;">'
            "Your stats will appear here once you rate some recipes.</div>",
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Loading nutrition database…"):
            nutrition_db = load_nutrition_db()
            nova_db      = load_nova_db()

        recipe_cache  = st.session_state.get("recipe_cache", {})
        recipe_lookup = {str(r["id"]): r for r in results}
        recipe_lookup.update(recipe_cache)

        labels, letters, grades = [], [], []
        for recipe_id in ratings:
            recipe = recipe_lookup.get(recipe_id)
            if not recipe:
                continue
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

            if overall_score >= 8.0:   overall_letter = "A"
            elif overall_score >= 6.0: overall_letter = "B"
            elif overall_score >= 4.0: overall_letter = "C"
            elif overall_score >= 2.0: overall_letter = "D"
            else:                      overall_letter = "E"

            score_color = LETTER_COLOR[overall_letter]

            # ── Summary metrics ──────────────────────────────────────────
            c1, c2 = st.columns(2)

            c1.metric("Meals cooked", total_meals)

            with c2:
                st.markdown(
                    f"""
                    <div style="background:white;border-radius:14px;padding:1.2rem 1.5rem;
                                border:1px solid #E5E7EB;">
                      <div style="font-size:0.85rem;color:#6B7280;font-weight:400;
                                  margin-bottom:0.4rem;">Greendle Health Score</div>
                      <div style="display:flex;align-items:baseline;gap:0.3rem;">
                        <span style="font-size:2rem;font-weight:800;color:{score_color};">
                          {overall_score}
                        </span>
                        <span style="font-size:1rem;color:#9CA3AF;font-weight:500;">&thinsp;/ 10</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Per-recipe bar chart ──────────────────────────────────────
            bar_colors = [LETTER_COLOR[l] for l in letters]

            fig = go.Figure(go.Bar(
                x=labels,
                y=grades,
                marker_color=bar_colors,
                marker_line_width=0,
                text=[str(g) for g in grades],
                textposition="outside",
                textfont=dict(color="#1A2D3D", size=13, family="Inter, sans-serif"),
            ))
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Inter, sans-serif", color="#1A2D3D"),
                yaxis=dict(
                    title="Greendle Score (/ 10)",
                    range=[0, 12],
                    gridcolor="#F0F0F0",
                    zeroline=False,
                ),
                xaxis=dict(title="Recipe", tickangle=-20),
                margin=dict(t=20, b=60, l=40, r=20),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                "<p style='color:#9CA3AF; font-size:0.8rem;'>Greendle Score = 60% nutritional quality "
                "(calories, fat, saturated fat, sugar, sodium, protein, carbs) + 40% food processing level "
                "(NOVA 1 = unprocessed → NOVA 4 = ultra-processed). Scored per serving, weighted by ingredient grams.</p>",
                unsafe_allow_html=True,
            )


# ── For You ────────────────────────────────────────────────────────────────
elif page == "For You":
    section_heading("For You")
    st.markdown(
        "<p style='color:#6B7280; margin-top:-1rem; margin-bottom:1.5rem;'>"
        "Personalised suggestions powered by collaborative filtering on 230 000+ Food.com recipes."
        "</p>",
        unsafe_allow_html=True,
    )

    ratings      = st.session_state.get("ratings", {})
    recipe_cache = st.session_state.get("recipe_cache", {})

    # ── no ratings yet ────────────────────────────────────────────────────
    if not ratings:
        st.markdown("""
        <div style="background:white; border-radius:16px; padding:2rem; text-align:center;
                    box-shadow:0 2px 10px rgba(0,0,0,0.06);">
            <div style="font-size:2.5rem; margin-bottom:0.6rem;">⭐</div>
            <div style="font-weight:700; color:#1A2D3D; font-size:1.05rem; margin-bottom:0.4rem;">
                No ratings yet
            </div>
            <div style="color:#6B7280; font-size:0.9rem; line-height:1.6;">
                Head to <strong>Find Recipes</strong>, cook something, and rate it —
                we'll use those stars to find what you'll love next.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── has ratings — try to load / build model ───────────────────────────
    else:
        pickle_ready = model_pickle_exists()
        spinner_msg  = (
            "Loading recommendation model…"
            if pickle_ready else
            "Building recommendation model for the first time — this can take a few minutes…"
        )

        with st.spinner(spinner_msg):
            model = _cf_model()

        # ── dataset not found ─────────────────────────────────────────────
        if model is None:
            st.markdown("""
            <div style="background:white; border-radius:16px; padding:1.8rem;
                        box-shadow:0 2px 10px rgba(0,0,0,0.06); border-left:4px solid #89cd8f;">
                <div style="font-weight:700; color:#1A2D3D; margin-bottom:0.6rem;">
                    Dataset not found
                </div>
                <div style="color:#6B7280; font-size:0.9rem; line-height:1.7;">
                    Download the Food.com dataset from Kaggle, then point Greendle to it
                    via your <code>.env</code> file or let <code>kagglehub</code> handle it
                    automatically.<br><br>
                    <strong>Option A — kagglehub (auto):</strong><br>
                    <code>pip install kagglehub</code><br>
                    Set up your Kaggle API key at <em>kaggle.com → Settings → API</em>,
                    then run once in a terminal:<br>
                    <code>python -c "import kagglehub; kagglehub.dataset_download('shuyangli94/food-com-recipes-and-user-interactions')"</code><br><br>
                    <strong>Option B — manual path:</strong><br>
                    Add to your <code>.env</code> file:<br>
                    <code>CF_DATASET_PATH=/path/to/food-com-recipes-and-user-interactions</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── model loaded — compute recommendations ────────────────────────
        else:
            # Build {title: rating} from rated Spoonacular recipes
            rated_dict = {
                recipe_cache[rid]["title"]: rating
                for rid, rating in ratings.items()
                if rid in recipe_cache and recipe_cache[rid].get("title")
            }

            # Cache recs in session state; recompute only when ratings change
            ratings_key = hash(frozenset(ratings.items()))
            if st.session_state.get("_cf_recs_key") != ratings_key:
                with st.spinner("Finding recommendations for you…"):
                    st.session_state["_cf_recs"]     = get_cf_recommendations(rated_dict, model, n=6)
                    st.session_state["_cf_recs_key"] = ratings_key

            recs = st.session_state.get("_cf_recs", [])

            col_refresh, _ = st.columns([1, 5])
            with col_refresh:
                if st.button("↻ Refresh"):
                    st.session_state.pop("_cf_recs_key", None)
                    st.cache_resource.clear()
                    st.rerun()

            if not recs:
                st.info(
                    "We couldn't match your rated recipes to our dataset. "
                    "Try rating a few more recipes in Find Recipes."
                )
            else:
                n_rated = len(ratings)
                st.markdown(
                    f"<p style='color:#3A6B3A; font-weight:600; margin-bottom:1.2rem;'>"
                    f"Based on {n_rated} recipe{'s' if n_rated != 1 else ''} you rated</p>",
                    unsafe_allow_html=True,
                )

                cols = st.columns(2)
                for i, rec in enumerate(recs):
                    ings_html = "".join(
                        f"<li style='margin:0.15rem 0;'>{ing}</li>"
                        for ing in rec["ingredients"]
                    )
                    with cols[i % 2]:
                        st.markdown(f"""
                        <div style="background:white; border-radius:14px; padding:1.2rem 1.4rem;
                                    margin-bottom:1rem; box-shadow:0 2px 8px rgba(0,0,0,0.06);
                                    border:1px solid #E5E7EB; min-height:140px;">
                            <div style="font-weight:700; color:#1A2D3D; margin-bottom:0.45rem;
                                        font-size:0.97rem; line-height:1.3;">{rec['name']}</div>
                            <div style="color:#9CA3AF; font-size:0.75rem; margin-bottom:0.3rem;
                                        text-transform:uppercase; letter-spacing:0.04em;">
                                Key ingredients
                            </div>
                            <ul style="color:#4B5563; font-size:0.82rem; margin:0;
                                       padding-left:1.1rem; line-height:1.6;">
                                {ings_html}
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

            # Algorithm explainer
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background:#F9FAFB; border-radius:12px; padding:1rem 1.3rem;
                        border:1px solid #E5E7EB; font-size:0.8rem; color:#9CA3AF; line-height:1.6;">
                <strong style="color:#6B7280;">How this works</strong><br>
                Your rated recipes are matched to Food.com titles using TF-IDF character-ngram
                similarity.  Each match contributes a latent-factor vector (from a 50-component
                SVD trained on 1 M+ user interactions), weighted by your star rating.
                The averaged vector is your taste profile — recommendations are its nearest
                neighbours in that latent space.
            </div>
            """, unsafe_allow_html=True)

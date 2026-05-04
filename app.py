import json
import os
import streamlit as st
import plotly.graph_objects as go
from api import search_by_ingredients, get_recipe_by_id
from recommender import rank_recipes

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

# Load persisted data into session state once per session
if "ratings" not in st.session_state:
    st.session_state["ratings"] = load_json(RATINGS_FILE)
if "recipe_cache" not in st.session_state:
    st.session_state["recipe_cache"] = load_json(CACHE_FILE)
if "profile" not in st.session_state:
    st.session_state["profile"] = load_json(PROFILE_FILE)

st.set_page_config(
    page_title="Greendle",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

#MainMenu {visibility: hidden;} footer {visibility: hidden;}

.block-container { padding-top: 2rem; max-width: 960px; }

h1 { font-size: 2rem !important; font-weight: 800 !important; color: #1A2D3D !important; }
h2 { font-size: 1.4rem !important; font-weight: 700 !important; color: #1A2D3D !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #1A2D3D !important; }

section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB;
}

.stButton > button {
    background-color: #89cd8f !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 1.6rem !important;
    transition: background-color 0.2s !important;
}
.stButton > button:hover {
    background-color: #2DA668 !important;
    color: white !important;
}

div[data-testid="stExpander"] {
    background-color: white !important;
    border-radius: 14px !important;
    border: 1px solid #E5E7EB !important;
}

div[data-testid="stMetric"] {
    background-color: white;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #E5E7EB;
}

div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
    border-radius: 10px !important;
    background-color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding: 0.5rem 0 1.5rem 0;">
    <span style="font-size:1.5rem; font-weight:800; color:#1A2D3D;">Greendle</span><span style="color:#89cd8f; font-size:1.5rem; font-weight:800;">.</span>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["Home", "My Profile", "Find Recipes", "My Dashboard"],
    label_visibility="collapsed"
)


def section_heading(title):
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="margin-bottom: 0.3rem;">{title}</h2>
        <div style="width:36px; height:3px; background:#89cd8f; border-radius:2px;"></div>
    </div>
    """, unsafe_allow_html=True)


def card(content_html):
    st.markdown(f"""
    <div style="background:white; border-radius:16px; padding:1.5rem;
                box-shadow:0 2px 10px rgba(0,0,0,0.06); margin-bottom:1rem;">
        {content_html}
    </div>
    """, unsafe_allow_html=True)


# ── Pages ─────────────────────────────────────────────────────

if page == "Home":
    st.markdown("""
    <div style="text-align:center; padding: 2.5rem 0 2rem 0;">
        <div style="font-size:3.5rem; margin-bottom:0.5rem;">🌿</div>
        <div style="font-size:3rem; font-weight:800; color:#1A2D3D; line-height:1.1;">
            Greendle<span style="color:#89cd8f;">.</span>
        </div>
        <div style="font-size:1.25rem; color:#6B7280; margin-top:0.6rem; font-weight:400;">
            Smarter meals, less waste.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div style="background:white;border-radius:16px;padding:1.5rem;box-shadow:0 2px 10px rgba(0,0,0,0.06);"><div style="color:#89cd8f;font-weight:700;font-size:1rem;margin-bottom:0.6rem;">The Problem</div><div style="color:#4B5563;line-height:1.6;">The average household wastes ~30% of groceries, costing <strong style="color:#1A2D3D;">$1,500 annually</strong> and increasing landfill emissions.</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="background:white;border-radius:16px;padding:1.5rem;box-shadow:0 2px 10px rgba(0,0,0,0.06);"><div style="color:#89cd8f;font-weight:700;font-size:1rem;margin-bottom:0.6rem;">The Solution</div><div style="color:#4B5563;line-height:1.6;">A smart meal planner that turns your leftover ingredients into healthy, personalized recipes — and tracks your impact.</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_heading("How it works")

    c1, c2, c3 = st.columns(3)
    features = [
        (c1, "📋", "Smart Input",       "Enter your ingredients and available cooking time."),
        (c2, "📖", "Full Guidance",     "Step-by-step instructions with nutrition info."),
        (c3, "📊", "Track Your Impact", "See how much food waste you've prevented."),
    ]
    for col, icon, title, desc in features:
        with col:
            st.markdown(f"""
            <div style="background:white; border-radius:14px; padding:1.5rem; text-align:center;
                        box-shadow:0 2px 10px rgba(0,0,0,0.06);">
                <div style="background:#D4EDE1; border-radius:10px; width:48px; height:48px;
                            display:flex; align-items:center; justify-content:center;
                            margin:0 auto 0.9rem auto; font-size:1.3rem;">{icon}</div>
                <div style="font-weight:700; color:#1A2D3D; margin-bottom:0.35rem;">{title}</div>
                <div style="color:#6B7280; font-size:0.88rem; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><p style='text-align:center; color:#9CA3AF; font-size:0.9rem;'>Use the sidebar to get started →</p>", unsafe_allow_html=True)


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

            with st.spinner("Searching recipes..."):
                recipes = search_by_ingredients(
                    ingredients, diet=diet, intolerances=intolerances or None
                )

            if not recipes:
                st.warning("No recipes found. Try different ingredients.")
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
        st.markdown(f"<p style='color:#89cd8f; font-weight:600; margin-bottom:1rem;'>Found {len(results)} recipes</p>", unsafe_allow_html=True)

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
                    st.markdown(f"<span style='color:#89cd8f; font-size:0.85rem;'>🥗 {', '.join(d.capitalize() for d in diets[:3])}</span>", unsafe_allow_html=True)

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
    section_heading("My Sustainability Dashboard")
    st.markdown("<p style='color:#6B7280; margin-top:-1rem; margin-bottom:1.5rem;'>Track how much food waste you've prevented by cooking from your available ingredients.</p>", unsafe_allow_html=True)

    ratings = st.session_state.get("ratings", {})
    results = st.session_state.get("search_results", [])

    if not ratings:
        st.markdown('<div style="background:white;border-radius:16px;padding:1.5rem;box-shadow:0 2px 10px rgba(0,0,0,0.06);text-align:center;color:#6B7280;">Your stats will appear here once you rate some recipes.</div>', unsafe_allow_html=True)
    else:
        # Build per-recipe stats for rated recipes
        # Each ingredient ~150g; CO2e: 2.5 kg per kg of food saved
        recipe_cache  = st.session_state.get("recipe_cache", {})
        recipe_lookup = {str(r["id"]): r for r in results}
        recipe_lookup.update(recipe_cache)
        labels, kg_saved, co2_saved, meals = [], [], [], []

        for recipe_id, rating in ratings.items():
            recipe = recipe_lookup.get(recipe_id)
            if not recipe:
                continue
            n_ingredients = len(recipe.get("extendedIngredients", [])) or 6
            food_kg = round(n_ingredients * 0.15, 2)
            co2_kg  = round(food_kg * 2.5, 2)
            short_title = recipe["title"][:25] + ("…" if len(recipe["title"]) > 25 else "")
            labels.append(short_title)
            kg_saved.append(food_kg)
            co2_saved.append(co2_kg)
            meals.append(1)

        total_kg  = round(sum(kg_saved), 2)
        total_co2 = round(sum(co2_saved), 2)
        total_meals = len(meals)

        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Meals cooked", total_meals)
        c2.metric("Food saved", f"{total_kg} kg")
        c3.metric("CO₂ avoided", f"{total_co2} kg")

        st.markdown("<br>", unsafe_allow_html=True)

        # Chart toggle
        chart_type = st.radio("Show", ["Food saved (kg)", "CO₂ avoided (kg)"], horizontal=True)
        y_values = kg_saved if chart_type == "Food saved (kg)" else co2_saved
        y_label  = "Food saved (kg)" if chart_type == "Food saved (kg)" else "CO₂ avoided (kg)"

        fig = go.Figure(go.Bar(
            x=labels,
            y=y_values,
            marker_color="#89cd8f",
            marker_line_width=0,
            text=[f"{v} kg" for v in y_values],
            textposition="outside",
            textfont=dict(color="#1A2D3D", size=12),
        ))
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter, sans-serif", color="#1A2D3D"),
            yaxis=dict(title=y_label, gridcolor="#F0F0F0", zeroline=False),
            xaxis=dict(title="Recipe", tickangle=-20),
            margin=dict(t=20, b=60, l=40, r=20),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<p style='color:#9CA3AF; font-size:0.8rem;'>Estimates based on average ingredient weights (~150g each) and a food waste CO₂ factor of 2.5 kg CO₂e per kg.</p>", unsafe_allow_html=True)

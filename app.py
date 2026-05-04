import streamlit as st
from api import search_by_ingredients, get_recipe_by_id
from recommender import rank_recipes

st.set_page_config(
    page_title="Greendle",
    page_icon="🌿",
    layout="wide"
)

# ── Sidebar navigation ───────────────────────────────────────
st.sidebar.title("🌿 Greendle")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "My Profile", "Find Recipes", "My Dashboard"]
)

# ── Pages ────────────────────────────────────────────────────

if page == "Home":
    st.title("🌿 Greendle")
    st.subheader("Smarter meals, less waste.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.error("### ⚠️ The Problem")
        st.write(
            "The average household wastes ~30% of groceries, "
            "costing **$1,500 annually** and increasing landfill emissions."
        )

    with col2:
        st.success("### ✏️ The Solution")
        st.write(
            "A smart meal planner that turns your leftover ingredients "
            "into healthy, personalized recipes — and tracks your impact."
        )

    st.markdown("---")
    st.markdown("### How it works")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**📋 Smart Input**\n\nEnter your ingredients and available time.")
    with c2:
        st.info("**📖 Full Guidance**\n\nStep-by-step instructions with nutrition info.")
    with c3:
        st.info("**📊 Track Your Impact**\n\nSee how much food waste you've prevented.")

    st.markdown("---")
    st.markdown("*Use the sidebar to get started →*")


elif page == "My Profile":
    st.title("👤 My Profile")
    st.write("Tell us about your preferences so we can personalize your recipes.")

    st.markdown("### Dietary Restrictions")
    col1, col2 = st.columns(2)
    with col1:
        vegan       = st.checkbox("Vegan")
        vegetarian  = st.checkbox("Vegetarian")
        gluten_free = st.checkbox("Gluten-free")
    with col2:
        dairy_free  = st.checkbox("Dairy-free")
        nut_free    = st.checkbox("Nut-free")
        halal       = st.checkbox("Halal")

    st.markdown("### Taste Preferences")
    cuisine = st.multiselect(
        "Favourite cuisines",
        ["Italian", "Mexican", "Asian", "Mediterranean", "American", "French", "Indian"]
    )
    spice_level = st.slider("Spice tolerance", 0, 5, 2, help="0 = no spice, 5 = very spicy")

    st.markdown("### Allergies")
    allergies = st.text_input("List any allergies (comma-separated)", placeholder="e.g. peanuts, shellfish")

    if st.button("💾 Save Profile"):
        st.session_state["profile"] = {
            "vegan": vegan,
            "vegetarian": vegetarian,
            "gluten_free": gluten_free,
            "dairy_free": dairy_free,
            "nut_free": nut_free,
            "halal": halal,
            "cuisine": cuisine,
            "spice_level": spice_level,
            "allergies": [a.strip() for a in allergies.split(",") if a.strip()],
        }
        st.success("Profile saved! ✅")


elif page == "Find Recipes":
    st.title("🔍 Find Recipes")
    st.write("Enter the ingredients you have and we'll find matching recipes.")

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

    if st.button("🔎 Search Recipes"):
        if not ingredients_input:
            st.warning("Please enter at least one ingredient.")
        else:
            ingredients = [i.strip() for i in ingredients_input.split(",") if i.strip()]

            # Build filters from saved profile
            profile = st.session_state.get("profile", {})
            diet = None
            if profile.get("vegan"):
                diet = "vegan"
            elif profile.get("vegetarian"):
                diet = "vegetarian"

            intolerances = []
            if profile.get("gluten_free"):
                intolerances.append("gluten")
            if profile.get("dairy_free"):
                intolerances.append("dairy")
            if profile.get("nut_free"):
                intolerances.append("tree nut")

            with st.spinner("Searching recipes..."):
                recipes = search_by_ingredients(
                    ingredients,
                    diet=diet,
                    intolerances=intolerances or None
                )

            if not recipes:
                st.warning("No recipes found. Try different ingredients.")
            else:
                ratings = st.session_state.get("ratings", {})
                ranked = rank_recipes(recipes, ingredients, ratings)

                with st.spinner("Loading full recipe details..."):
                    full_recipes = []
                    for r in ranked:
                        cache_key = f"full_{r['id']}"
                        if cache_key not in st.session_state:
                            full = get_recipe_by_id(r["id"])
                            st.session_state[cache_key] = full if full else r
                        full_recipes.append(st.session_state[cache_key])

                st.session_state["search_results"] = full_recipes
                st.session_state["search_ingredients"] = ingredients

    # Display results (persists across reruns)
    if "search_results" in st.session_state:
        results = st.session_state["search_results"]
        st.success(f"Found {len(results)} recipes!")

        for recipe in results:
            col1, col2 = st.columns([1, 3])
            with col1:
                if recipe.get("image"):
                    st.image(recipe["image"], width=150)
            with col2:
                st.markdown(f"### {recipe['title']}")
                ready_in = recipe.get("readyInMinutes", "?")
                servings = recipe.get("servings", "?")
                st.caption(f"⏱ {ready_in} min  ·  🍽 {servings} servings")
                diets = recipe.get("diets", [])
                if diets:
                    st.caption("🥗 " + ", ".join(d.capitalize() for d in diets[:3]))

            with st.expander("📖 View full recipe"):
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
                if st.button("💾 Save Rating", key=f"save_{recipe['id']}"):
                    if "ratings" not in st.session_state:
                        st.session_state["ratings"] = {}
                    st.session_state["ratings"][str(recipe["id"])] = rating
                    st.success("Rating saved! It will influence future recommendations.")

            st.markdown("---")


elif page == "My Dashboard":
    st.title("📊 My Sustainability Dashboard")
    st.write("Track how much food waste you've prevented over time.")

    ratings = st.session_state.get("ratings", {})
    results = st.session_state.get("search_results", [])

    if not results:
        st.info("Your stats will appear here once you start searching and rating recipes.")
    else:
        total_rated = len(ratings)
        st.metric("Recipes rated", total_rated)
        st.metric("Recipes found so far", len(results))
        if total_rated:
            avg = sum(ratings.values()) / total_rated
            st.metric("Average rating", f"{avg:.1f} / 5")

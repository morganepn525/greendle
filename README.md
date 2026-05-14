# Greendle

Greendle is a smart meal planning web app built with Python and Streamlit. The idea is simple: instead of going to the store and buying ingredients for a specific recipe, you start with what you already have at home and find something good to cook with it. This helps reduce food waste, which is a bigger problem than most people realise: the average household throws away about 30% of the food it buys.

---

## What the app does

- **Find Recipes**: enter the ingredients you have and get recipe suggestions ranked by how well they match
- **My Profile**: save your dietary preferences (vegan, gluten-free, halal, etc.) so results are filtered automatically
- **My Dashboard**: see a health score for every meal you have rated, based on nutritional quality and food processing level
- **For You**: personalised recipe suggestions based on your highest-rated meals

---

## How to run it

**1. Clone the repository**
```
git clone https://github.com/morganepn525/greendle.git
cd greendle
```

**2. Install dependencies**
```
pip install -r requirements.txt
```

**3. Add your Spoonacular API key**

Create a file called `.env` in the project folder and add the following line:
```
SPOONACULAR_KEY=your_key_here
```

You can get a free API key at [spoonacular.com/food-api](https://spoonacular.com/food-api). The free plan includes 150 requests per day, which is enough to use the app normally.

**4. Launch the app**
```
streamlit run app.py
```

The app will open automatically in your browser.

---

## Project structure

| File | Description |
|---|---|
| `app.py` | Main Streamlit application, all pages and UI |
| `api.py` | Communication with the Spoonacular API |
| `recommender.py` | Content-based ML recommender (TF-IDF + cosine similarity) |
| `cf_recommender.py` | Collaborative filtering recommender (TruncatedSVD on Food.com dataset) |
| `nutriscore.py` | Health score calculator (nutrition + NOVA food processing level) |
| `data/` | JSON files for storing ratings, profile, and cached recipes |

---

## AI usage declaration

In accordance with the University of St. Gallen referencing rules for generative AI, we declare the following:

Parts of this project were developed with the assistance of **Claude Code by Anthropic** (claude-sonnet-4-6). AI assistance was used for:

- Writing and debugging Python code across all modules
- Designing the structure of the recommendation algorithm
- Styling the user interface with CSS

All AI-generated code was reviewed, tested, and adapted by the team. The overall concept, problem framing, and project decisions were made by the team independently.

---

## Team

University of St. Gallen - Grundlagen und Methoden der Informatik, Group Project, Spring 2026

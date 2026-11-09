# Curato AI — Customer Recommendation System

Flask serves the website and API. Three Python recommendation methods are available:

- **Popularity ranking:** community ratings stabilized by a review-count prior.
- **User-based collaborative filtering:** similar customers' ratings predict preference.
- **Cart & wishlist matching:** content-based recommendations from products a customer saves. This replaces SVD.

## Run locally

Use Python 3.12 (see `.python-version`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.
Open `http://127.0.0.1:8000`. No external AI service or API key is needed.
Node dependencies are only for optional Vercel analytics; no frontend build is required.

## Try the shopping model

1. Select a customer, or **New customer (no history)**.
2. Browse products in **Cart & wishlist**. Add headphones, cameras, or other products to either list.
3. Choose **Cart & wishlist**, then **Generate recommendations**.
4. Open **Why recommended?** to see the saved item responsible for a suggestion.
5. Remove or change saved products and generate again to see the results change.

Names and categories become sparse TF-IDF features. Cosine similarity compares candidates with saved products. Known categories must agree to prevent unrelated matches on words such as “Travel”. Cart weight is 2; wishlist weight is 1. A product in both lists counts once at cart weight. The output is a similarity percentage, not a predicted star rating or a purchase probability. Popularity and collaborative models retain 1–5 rating scores.

All models exclude already-rated, cart, and wishlist products. The shopping model returns only positive matches, so it can return fewer products than requested. Without usable saved-product metadata, it explicitly falls back to popularity. To include products without ratings, keep the minimum rating count at 0.

## Sample and persistence

The bundled synthetic teaching dataset has **20 customers (U001–U020), 60 products, and 390 ratings**. Every customer has a different history length, ranging from 10 to 29 ratings. For example, U001 has 10, U002 has 15, U003 has 20, U004 has 13, and U005 has 14.

Synthetic cart/wishlist examples are included and can be edited. Ratings and shopping changes are stored in SQLite under ignored `runtime/`, separately for each dataset. Sample restoration preserves saved sample changes. Existing custom imported datasets are not silently overwritten or truncated to 20 customers. Import a ratings CSV and optional product names/categories from **Dataset & analysis**. Names or categories are needed for shopping similarity; ratings-only imports still support the other two methods.

Serverless fallbacks mirror the CSV files. On Vercel, runtime state uses `/tmp/curato-ai` and is temporary per instance; shared durable multi-instance storage would be needed for production shopping accounts. Customer IDs in this demo are selectable profiles, not authenticated accounts.

To reproduce the bundled data and embedded fallback after intentional changes:

```bash
python scripts/generate_sample.py
```

## Validate

```bash
.venv/bin/python -m unittest discover -s tests -v
node --check public/app.js
.venv/bin/python -m pip check
```

Tests use isolated temporary datasets and cover distinct history lengths, model relevance, cart weights, exclusions, persistence, new customers, imports, and fallback behavior.

## Source tour

- `backend/models/activity.py`: product features, weighted shopping signals, cosine scores, explanation sources.
- `backend/models/popularity.py`: Bayesian weighted popularity ranking.
- `backend/models/collaborative.py`: user-based collaborative filtering.
- `backend/models/engine.py`: candidate filtering and model selection.
- `backend/store.py`: dataset-scoped SQLite state.
- `backend/app.py`: API and website routes.
- [Model design and research](docs/recommendation-design.md): formulas, choices, references, and limitations.

Create a feature branch before source changes; do not commit directly to `main`.

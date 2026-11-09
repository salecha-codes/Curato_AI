# Cart and wishlist recommendations

## Why replace SVD?

SVD stands for singular value decomposition. The previous model compressed a customer–product rating matrix into latent factors and reconstructed estimated ratings. Those hidden numerical factors are harder to explain and do not directly capture “this customer saved these headphones.”

The replacement is a **content-based recommendation model with implicit interest signals**. Cart and wishlist membership express interest without asking for a rating. Product metadata supplies the similarity features. It works for a new customer as soon as they save a product and for catalog items that have no ratings.

## Research and implementation choice

The methods follow the scikit-learn documentation:

1. [TF-IDF term weighting](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting): term frequency multiplied by smoothed inverse document frequency; L2 normalization.
2. [Cosine similarity](https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity): the normalized dot product, commonly used with TF-IDF vectors.

The corresponding upstream documentation was consulted during implementation: [feature extraction source](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/feature_extraction.rst) and [metrics source](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/metrics.rst).

A content approach fits this small catalog because names and categories already exist, while genuine cart-event training data does not. Training another collaborative model on fabricated cart logs would give an unwarranted impression of learned shopping behavior. Sample shopping entries here are clearly labeled synthetic and are used as editable inputs. No external service or additional ML dependency is required; NumPy and SciPy implement the sparse operations directly.

## Features and ranking

For each catalog product:

- Tokenize the case-folded name into word counts (tokens longer than one character).
- Add one category feature with term frequency 2. Name and category features use separate namespaces.
- Omit generated `Product <ID>` placeholder names and `Uncategorized` features.
- Apply `idf(t) = log((1 + number_of_products) / (1 + products_containing_t)) + 1`.
- L2-normalize the resulting sparse vector.

For a customer, every unique saved item contributes a product vector. A cart item has weight 2 and a wishlist item weight 1. An item in both lists uses the maximum weight, rather than double counting. The score is:

```text
score(candidate) = sum(weight(saved) × cosine(candidate, saved)) / sum(weights)
```

When both products have known categories, their categories must agree. This prevents a shared adjective such as “Travel” from connecting cameras to headphones. If either category is missing, name similarity is still available. Incompatible-category contributions are zero.

The weights are not renormalized per candidate: the score remains an average of individual cosine similarities, in [0, 1]. The website displays it as a percentage **match**. This is not a predicted rating or a purchase probability. Different model score types are also identified in the API and CSV export.

Products are sorted by descending score, with product ID breaking ties. Only positive matches are returned. Already-rated and currently saved products are excluded. The minimum rating count filter still applies; its default is 0 to allow unrated catalog items. For each result the explanation identifies the saved item with the strongest weighted contribution and its source list.

If there are no usable saved-product features, popularity is an explicit fallback. If usable interests exist but all matching products are excluded, the result is empty rather than padded with unrelated products. Popularity and user-based filtering remain available as separate rating-based baselines.

The implementation computes sparse cosine matrix-vector products rather than storing a dense product-by-product similarity matrix. Explanation contributions are computed one saved product at a time.

## Persistence and data

The sample generator is deterministic (seed 42), with 20 customer IDs, 60 products, and 390 ratings. Rating counts are a permutation of 10 through 29. This changes history lengths, not the 1–5 meaning of a rating. Each product has at least one sample rating.

Cart/wishlist overrides are stored separately from ratings in SQLite, keyed by dataset fingerprint, customer ID, product ID, and list kind. Explicit removals override seeded entries after restart. Updates are idempotent. Dataset switches isolate shopping state; returning to the sample restores its saved shopping state. Adding shopping signals never creates fabricated rating rows or retrains rating models.

The source CSVs and embedded serverless constants are generated together. Existing runtime data and imported datasets are preserved. On Vercel, SQLite lives in temporary per-instance storage; this demo does not claim durable production account storage or authentication.

## Validation and limits

The regression suite exercises all 20 sample profiles across all three models, plus controlled headphone/camera examples. It checks cart weighting, duplicate signal handling, source explanations, exclusions, empty metadata, unrated candidates, explicit fallback, API validation, persistence across restart, import isolation, sample restoration, rating updates, and embedded-data parity. Browser validation exercises the actual add/remove and customer selection workflow.

These checks establish correct behavior, not commercial recommendation quality. No real shopping outcomes or labeled relevance benchmark are available. The weights are understandable demonstration defaults, not optimized business parameters. Metadata matching does not capture price, stock, images, synonyms, product compatibility, purchases, or recency. Missing categories or broad categories can still cause weak name matches. Production evaluation would need consented interaction data, time-based holdouts, precision/recall or ranking metrics, and user outcome measurement before tuning weights or adding learned embeddings.

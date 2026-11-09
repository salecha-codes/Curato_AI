"""Content recommendations from explicit cart and wishlist signals.

Sparse TF-IDF name features + category features; cosine matching against a
weighted interest profile. No rating predictions or dense item-item matrix.
"""
from collections import Counter
import re
import numpy as np
from scipy.sparse import csr_matrix, diags


class ActivityModel:
    WEIGHTS = {'cart': 2.0, 'wishlist': 1.0}

    def __init__(self, catalog):
        self.items = [p['product_id'] for p in catalog]
        self.index = {pid: i for i, pid in enumerate(self.items)}
        self.categories = [p['category'].casefold().strip() for p in catalog]
        documents = []
        for p in catalog:
            # Placeholder names and missing categories carry no semantic signal.
            name = '' if p['name'] == f"Product {p['product_id']}" else p['name']
            tokens = Counter('name:' + t for t in re.findall(r'[^\W_]+', name.casefold()) if len(t) > 1)
            if p['category'].casefold() != 'uncategorized':
                tokens['category:' + p['category'].casefold().strip()] = 2
            documents.append(tokens)
        df = Counter(t for doc in documents for t in doc)
        vocab = {t: i for i, t in enumerate(sorted(df))}
        rows, cols, values = [], [], []
        for i, doc in enumerate(documents):
            for token, tf in doc.items():
                rows.append(i); cols.append(vocab[token])
                values.append(tf * (np.log((1 + len(documents)) / (1 + df[token])) + 1))
        matrix = csr_matrix((values, (rows, cols)), shape=(len(documents), len(vocab)), dtype=float)
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        self.features = (diags(np.divide(1., norms, out=np.zeros_like(norms), where=norms > 0)) @ matrix).tocsr()
        self.usable = norms > 0

    def predict(self, signals):
        seeds = []
        for pid, kinds in sorted(signals.items()):
            i = self.index.get(pid)
            if i is not None and self.usable[i]:
                kind = max(kinds, key=lambda k: self.WEIGHTS[k])
                seeds.append((i, kind, self.WEIGHTS[kind]))
        if not seeds:
            return None, {}
        # Accumulate weighted cosine matches without a dense item-item matrix.
        # Known categories must agree: e.g. "Travel Camera" is not a similar
        # alternative to "Travel Headphones" merely because both say "Travel".
        total_weight = sum(w for _, _, w in seeds)
        scores = np.zeros(len(self.items))
        sources = {}
        for i, kind, weight in seeds:
            similarities = (self.features @ self.features.getrow(i).T).tocoo()
            for row, similarity in zip(similarities.row, similarities.data):
                category, seed_category = self.categories[row], self.categories[i]
                if category != 'uncategorized' and seed_category != 'uncategorized' and category != seed_category:
                    continue
                scores[row] += float(similarity * weight / total_weight)
                contribution = float(similarity * weight)
                if contribution > sources.get(int(row), {}).get('contribution', -1):
                    sources[int(row)] = {'product_id': self.items[i], 'kind': kind,
                                         'similarity': round(float(similarity), 4), 'contribution': contribution}
        return np.clip(scores, 0, 1), sources

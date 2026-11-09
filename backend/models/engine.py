"""Shared rating matrix, rating baselines and cart/wishlist content matching."""
import time
import numpy as np
from scipy.sparse import csr_matrix
from . import popularity
from .collaborative import UserCollaborative
from .activity import ActivityModel

METHODS = {'popularity', 'collaborative', 'activity'}


class RecommendationEngine:
    def __init__(self, ratings, catalog=None, users=None, items=None):
        start = time.perf_counter()
        self.users = sorted(ratings.user_id.unique()) if users is None else users
        self.items = sorted(p['product_id'] for p in catalog) if catalog is not None else sorted(ratings.product_id.unique())
        if items is not None:
            self.items = items
        self.user_index = {u:i for i,u in enumerate(self.users)}
        self.item_index = {p:i for i,p in enumerate(self.items)}
        self.matrix = csr_matrix((ratings.rating.to_numpy(float),
                                 ([self.user_index[u] for u in ratings.user_id], [self.item_index[p] for p in ratings.product_id])),
                                 shape=(len(self.users),len(self.items)))
        self.popularity, self.averages, self.counts, self.global_mean = popularity.fit(self.matrix)
        counts = np.diff(self.matrix.indptr)
        self.means = np.divide(np.asarray(self.matrix.sum(axis=1)).ravel(), counts,
                               out=np.full(len(self.users), self.global_mean), where=counts > 0)
        self.cf = UserCollaborative(self.matrix, self.means, self.popularity)
        records = {p['product_id']: p for p in (catalog or [])}
        self.activity = ActivityModel([records.get(pid, {'product_id': pid, 'name': f'Product {pid}', 'category': 'Uncategorized'}) for pid in self.items])
        self.training_ms = round((time.perf_counter()-start)*1000, 1)

    def recommend(self, user_id, method, n=8, min_ratings=0, signals=None):
        if method not in METHODS:
            raise ValueError('Unknown method. Choose popularity, collaborative, or activity.')
        signals = signals or {}
        index = self.user_index.get(user_id)
        neighbors, sources, fallback = [], {}, None
        if method == 'activity':
            scores, sources = self.activity.predict(signals)
            if scores is None:
                scores = self.popularity.copy()
                fallback = 'popularity'
        elif method == 'collaborative' and index is not None:
            scores, neighbors = self.cf.predict(index)
            if not neighbors:
                fallback = 'popularity'
        else:
            scores = self.popularity.copy()
            if method != 'popularity':
                fallback = 'popularity'
        seen = set(self.matrix.getrow(index).indices) if index is not None else set()
        seen.update(self.item_index[p] for p in signals if p in self.item_index)
        matching = method == 'activity' and fallback is None
        eligible = [i for i in range(len(self.items)) if i not in seen and self.counts[i] >= min_ratings
                    and (not matching or scores[i] > 1e-12)]
        ordered = sorted(eligible, key=lambda i: (-scores[i], self.items[i]))[:n]
        rows = []
        for i in ordered:
            row = {'product_id':self.items[i], 'score':round(float(scores[i]),4),
                   'average_rating':round(float(self.averages[i]),2) if self.counts[i] else None,
                   'rating_count':int(self.counts[i])}
            if matching:
                row['source'] = {k: v for k, v in sources[i].items() if k != 'contribution'}
            rows.append(row)
        return rows, neighbors, fallback

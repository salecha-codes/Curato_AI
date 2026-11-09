"""Run: .venv/bin/python -m unittest discover -s tests -v"""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from backend.app import create_app
from backend.store import Store
from backend.models.activity import ActivityModel
from backend.models.engine import RecommendationEngine
from backend.embedded_data import DEFAULT_RATINGS_CSV, DEFAULT_PRODUCTS_CSV, DEFAULT_ACTIVITY_CSV, DEFAULT_INFO_JSON


ROOT = Path(__file__).resolve().parents[1]


def fixture(root):
    data = root/'data'; data.mkdir()
    for filename, contents in [('ratings.csv', DEFAULT_RATINGS_CSV), ('products.csv', DEFAULT_PRODUCTS_CSV),
                               ('activity.csv', DEFAULT_ACTIVITY_CSV), ('dataset_info.json', DEFAULT_INFO_JSON)]:
        (data/filename).write_text(contents)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.catalog = [
            {'product_id':'A', 'name':'Studio Headphones', 'category':'Audio'},
            {'product_id':'B', 'name':'Travel Headphones', 'category':'Audio'},
            {'product_id':'C', 'name':'Mirrorless Camera', 'category':'Photography'},
            {'product_id':'D', 'name':'Travel Camera', 'category':'Photography'},
            {'product_id':'E', 'name':'Daily Backpack', 'category':'Accessories'},
        ]
        self.model = ActivityModel(self.catalog)

    def test_cart_and_wishlist_each_match_related_products(self):
        for kind in ('cart', 'wishlist'):
            scores, sources = self.model.predict({'A':{kind}})
            self.assertGreater(scores[1], scores[2])
            self.assertEqual(sources[1]['product_id'], 'A')
            self.assertEqual(sources[1]['kind'], kind)
            self.assertTrue(np.all((scores >= 0) & (scores <= 1)))

    def test_shared_generic_word_does_not_cross_known_categories(self):
        scores, _ = self.model.predict({'B':{'cart'}})
        self.assertEqual(scores[3], 0)  # Travel Headphones vs Travel Camera.

    def test_changing_cart_changes_top_category(self):
        audio, _ = self.model.predict({'A':{'cart'}})
        cameras, _ = self.model.predict({'C':{'cart'}})
        self.assertGreater(audio[1], audio[3])
        self.assertGreater(cameras[3], cameras[1])

    def test_cart_weight_and_no_double_counting(self):
        scores, _ = self.model.predict({'A':{'cart'}, 'C':{'wishlist'}})
        a, _ = self.model.predict({'A':{'wishlist'}})
        c, _ = self.model.predict({'C':{'cart'}})
        np.testing.assert_allclose(scores, (2*a+c)/3)
        double, _ = self.model.predict({'A':{'wishlist','cart'}, 'C':{'wishlist'}})
        np.testing.assert_allclose(scores, double)

    def test_missing_metadata_is_not_a_personalization_signal(self):
        model = ActivityModel([{'product_id':'A','name':'Product A','category':'Uncategorized'}])
        self.assertIsNone(model.predict({'A':{'cart'}})[0])

    def test_zero_similarity_not_padded_into_results_and_exclusions(self):
        ratings = pd.DataFrame([{'user_id':'U', 'product_id':'E', 'rating':4}])
        engine = RecommendationEngine(ratings, self.catalog)
        rows, _, fallback = engine.recommend('U', 'activity', signals={'A':{'cart'}})
        self.assertIsNone(fallback)
        self.assertEqual([r['product_id'] for r in rows], ['B'])
        self.assertIsNone(rows[0]['average_rating'])
        self.assertEqual(engine.recommend('U','activity',min_ratings=1,signals={'A':{'cart'}})[0], [])


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fixture(self.root)
        self.env = patch.dict('os.environ'); self.env.start()
        import os
        os.environ.pop('VERCEL', None)
        self.app = create_app(self.root); self.client = self.app.test_client()

    def tearDown(self):
        self.env.stop(); self.temp.cleanup()

    def activity(self, uid='U001', pid='P001', kind='cart', active=True):
        response = self.client.post('/api/activity', json={'user_id':uid,'product_id':pid,'kind':kind,'active':active})
        self.assertEqual(response.status_code, 200, response.json)
        return response.json

    def test_sample_has_exactly_twenty_customers_and_unique_lengths(self):
        data = self.client.get('/api/users').json
        self.assertEqual(data['total'], 20)
        self.assertEqual([u['user_id'] for u in data['users']], [f'U{i:03d}' for i in range(1,21)])
        self.assertEqual(sorted(u['ratings'] for u in data['users']), list(range(10,30)))
        o = self.client.get('/api/overview').json
        self.assertEqual((o['users'],o['products'],o['ratings']), (20,60,390))
        self.assertNotIn('factors', o)
        for filename, embedded in [('ratings.csv',DEFAULT_RATINGS_CSV), ('products.csv',DEFAULT_PRODUCTS_CSV),
                                   ('activity.csv',DEFAULT_ACTIVITY_CSV), ('dataset_info.json',DEFAULT_INFO_JSON)]:
            self.assertEqual((ROOT/'data'/filename).read_text(), embedded)

    def test_sample_recommendations_for_all_customers_and_methods(self):
        for number in range(1,21):
            uid = f'U{number:03d}'
            profile = self.client.get('/api/profile?user_id='+uid).json
            excluded = {p['product_id'] for key in ('ratings','cart','wishlist') for p in profile[key]}
            for method in ('activity','collaborative','popularity'):
                response = self.client.get(f'/api/recommendations?user_id={uid}&method={method}')
                self.assertEqual(response.status_code,200,response.json)
                result = response.json
                self.assertTrue(result['products'], (uid, method))
                self.assertFalse(excluded & {p['product_id'] for p in result['products']})
                scores = [p['score'] for p in result['products']]
                self.assertEqual(scores, sorted(scores, reverse=True))
                if method == 'activity':
                    self.assertEqual(result['effective_method'], 'activity')
                    self.assertEqual(result['score_type'], 'similarity')
                    self.assertTrue(all(0 < v <= 1 for v in scores))

    def test_activity_only_customer_is_personalized_without_ratings(self):
        self.activity(uid='visitor')
        result = self.client.get('/api/recommendations?user_id=visitor&method=activity').json
        self.assertTrue(result['cold_start'])
        self.assertIsNone(result['fallback'])
        self.assertEqual(result['effective_method'], 'activity')
        self.assertTrue(result['products'])
        self.assertTrue(any(u['user_id']=='visitor' and u['ratings']==0 for u in self.client.get('/api/users').json['users']))
        other = self.client.get('/api/profile?user_id=someone_else').json
        self.assertEqual(other['cart'], [])

    def test_seed_removal_addition_idempotency_and_restart(self):
        store = self.app.config['STORE']
        profile = self.client.get('/api/profile?user_id=U001').json
        pid = profile['cart'][0]['product_id']
        before = self.client.get('/api/overview').json['version']
        first = self.activity(pid=pid,active=False)
        self.assertNotEqual(first['version'], before)
        self.assertEqual(first['version'], self.activity(pid=pid,active=False)['version'])
        self.assertNotIn(pid, Store(self.root).signals('U001'))
        self.activity(pid=pid,kind='wishlist')
        self.assertEqual(Store(self.root).signals('U001')[pid], {'wishlist'})
        self.assertEqual(len(store.ratings),390)

    def test_empty_activity_has_explicit_popularity_fallback(self):
        data = self.client.get('/api/recommendations?user_id=unknown&method=activity').json
        self.assertEqual(data['fallback'],'popularity')
        self.assertEqual(data['score_type'],'rating')
        self.assertIn('cart or wishlist',data['fallback_reason'])

    def test_validation_rejects_bad_actions_and_obsolete_model(self):
        for body in [[], {}, {'user_id':'U001','product_id':'missing','kind':'cart','active':True},
                     {'user_id':'U001','product_id':'P001','kind':'view','active':True},
                     {'user_id':'U001','product_id':'P001','kind':'cart','active':'false'},
                     {'user_id':[],'product_id':'P001','kind':'cart','active':True},
                     {'user_id':'U001','product_id':[],'kind':'cart','active':True}]:
            self.assertEqual(self.client.post('/api/activity',json=body).status_code,400)
        self.assertEqual(self.client.get('/api/recommendations?method=svd').status_code,400)
        self.assertEqual(self.client.get('/api/recommendations?top_n=0').status_code,400)

    def test_export_includes_score_semantics(self):
        data = self.client.get('/api/recommendations/export?method=activity&user_id=U001').data.decode()
        rows = list(csv.DictReader(io.StringIO(data)))
        self.assertTrue(rows)
        self.assertTrue(all(r['score_type']=='similarity' for r in rows))
        self.assertNotIn('predicted_rating', rows[0])

    def test_import_isolation_unrated_catalog_and_restore(self):
        self.activity(uid='visitor',pid='P001')
        ratings = b'user_id,product_id,rating\ncustom,A,4\ncustom,B,5\n'
        metadata = b'product_id,name,category\nA,Studio Headphones,Audio\nB,Daily Backpack,Accessories\nC,Travel Headphones,Audio\n'
        response = self.client.post('/api/dataset',data={'ratings':(io.BytesIO(ratings),'ratings.csv'),
                                                      'catalog':(io.BytesIO(metadata),'products.csv')})
        self.assertEqual(response.status_code,200,response.json)
        self.assertEqual(self.client.get('/api/overview').json['products'],3)
        self.assertEqual(self.client.get('/api/profile?user_id=visitor').json['cart'],[])
        self.activity(uid='new',pid='A')
        result = self.client.get('/api/recommendations?user_id=new&method=activity').json
        self.assertEqual(result['products'][0]['product_id'],'C')
        self.assertIsNone(result['products'][0]['average_rating'])
        reloaded = create_app(self.root).test_client()
        self.assertEqual(reloaded.get('/api/profile?user_id=new').json['cart'][0]['product_id'],'A')
        self.assertEqual(self.client.post('/api/dataset/sample').status_code,200)
        self.assertEqual(self.client.get('/api/profile?user_id=visitor').json['cart'][0]['product_id'],'P001')
        self.assertEqual(self.client.get('/api/profile?user_id=new').json['cart'],[])

    def test_serverless_embedded_data_without_disk_csv(self):
        import shutil
        shutil.rmtree(self.root/'data')
        # Separate writable root exercises the same embedded fallback as deployment.
        fallback = create_app(self.root).test_client()
        self.assertEqual(fallback.get('/api/overview').json['ratings'],390)
        self.assertTrue(fallback.get('/api/profile?user_id=U001').json['cart'])

    def test_rating_update_retains_activity_model(self):
        response = self.client.post('/api/ratings',json={'user_id':'U001','product_id':'P001','rating':5})
        self.assertEqual(response.status_code,200,response.json)
        result = self.client.get('/api/recommendations?user_id=U001&method=activity').json
        self.assertEqual(result['effective_method'],'activity')
        self.assertNotIn('P001', [p['product_id'] for p in result['products']])


if __name__ == '__main__':
    unittest.main()

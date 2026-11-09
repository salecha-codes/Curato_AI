"""Dataset state and SQLite rating overrides. Fitting is atomic before activation."""
from pathlib import Path
from io import BytesIO, StringIO
import os
import sqlite3
import json
import hashlib
import csv
import pandas as pd
from .data import read_ratings, read_catalog
from .models import RecommendationEngine
from .embedded_data import DEFAULT_RATINGS_CSV, DEFAULT_PRODUCTS_CSV, DEFAULT_INFO_JSON, DEFAULT_ACTIVITY_CSV


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.data_dir = self.root/'data'
        # Vercel's deployed source is read-only; use its writable temporary directory.
        # Local development continues to use the ignored runtime/ directory.
        self.runtime = Path('/tmp/curato-ai') if os.environ.get('VERCEL') else self.root/'runtime'
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.db = self.runtime/'ratings.sqlite3'
        with sqlite3.connect(self.db) as db:
            db.execute('CREATE TABLE IF NOT EXISTS ratings (dataset_id TEXT, user_id TEXT, product_id TEXT, rating REAL, PRIMARY KEY(dataset_id,user_id,product_id))')
            db.execute('CREATE TABLE IF NOT EXISTS activity (dataset_id TEXT, user_id TEXT, product_id TEXT, kind TEXT, active INTEGER, PRIMARY KEY(dataset_id,user_id,product_id,kind))')
        _, sample_report = read_ratings(BytesIO(DEFAULT_RATINGS_CSV.encode()))
        self.sample_id = sample_report['dataset_id']
        self.active = self.runtime/'active.json'
        self.load()

    def load(self):
        info = json.loads(self.active.read_text()) if self.active.exists() else None
        
        # Load ratings: check runtime override, then disk file, then embedded dataset
        ratings_path = self.runtime/'imported.csv' if info and (self.runtime/'imported.csv').exists() else self.data_dir/'ratings.csv'
        if ratings_path.exists():
            base, report = read_ratings(ratings_path)
        else:
            base, report = read_ratings(BytesIO(DEFAULT_RATINGS_CSV.encode('utf-8')))
            
        self.dataset_id = report['dataset_id']
        if info and 'cleaning_report' in info:
            report = info['cleaning_report']
            
        try:
            with sqlite3.connect(self.db) as db:
                updates = pd.read_sql_query('SELECT user_id,product_id,rating FROM ratings WHERE dataset_id=?', db, params=(self.dataset_id,))
        except Exception:
            updates = pd.DataFrame()
            
        ratings = (pd.concat([base, updates]) if not updates.empty else base).drop_duplicates(['user_id', 'product_id'], keep='last').reset_index(drop=True)
        
        # Load catalog metadata: check runtime, then disk file, then embedded catalog
        metadata = self.runtime/'catalog.csv' if info and (self.runtime/'catalog.csv').exists() else self.data_dir/'products.csv'
        if metadata and metadata.exists():
            catalog = read_catalog(metadata, ratings.product_id)
        else:
            catalog = read_catalog(BytesIO(DEFAULT_PRODUCTS_CSV.encode('utf-8')), ratings.product_id)
            
        # Load dataset info metadata: check active info, then disk file, then embedded info
        dataset_info = info
        if not dataset_info:
            info_path = self.data_dir/'dataset_info.json'
            if info_path.exists():
                dataset_info = json.loads(info_path.read_text())
            else:
                dataset_info = json.loads(DEFAULT_INFO_JSON)
                
        self.activate(ratings, catalog, report, dataset_info)

    def activate(self, ratings, catalog, report, info):
        engine = RecommendationEngine(ratings, catalog)
        self.ratings, self.catalog, self.report, self.info, self.engine = ratings,catalog,report,info,engine
        self.by_id = {p['product_id']:p for p in catalog}
        self.version = hashlib.sha256(ratings.to_csv(index=False).encode()).hexdigest()[:12]
        self.load_activity()

    def import_dataset(self, ratings_bytes, metadata_bytes, name):
        from io import BytesIO
        ratings,report = read_ratings(BytesIO(ratings_bytes))
        catalog = read_catalog(BytesIO(metadata_bytes) if metadata_bytes else None,ratings.product_id)
        info = {'name':name,'source':'User-imported CSV. Product metadata is optional.','synthetic':None,'cleaning_report':report}
        # Fit before changing the active dataset, so invalid imports leave it intact.
        engine = RecommendationEngine(ratings, catalog)
        (self.runtime/'imported.csv').write_text(ratings.to_csv(index=False))
        pd.DataFrame(catalog).to_csv(self.runtime/'catalog.csv',index=False)
        self.active.write_text(json.dumps(info))
        self.dataset_id = report['dataset_id']
        self.ratings,self.catalog,self.report,self.info,self.engine=ratings,catalog,report,info,engine
        self.by_id = {p['product_id']:p for p in catalog}
        self.version = hashlib.sha256(ratings.to_csv(index=False).encode()).hexdigest()[:12]
        self.load_activity()

    def rate(self, uid, pid, rating):
        rows = self.ratings.copy()
        mask = (rows.user_id==uid)&(rows.product_id==pid)
        if mask.any(): rows.loc[mask,'rating']=rating
        else: rows = pd.concat([rows,pd.DataFrame([{'user_id':uid,'product_id':pid,'rating':rating}])],ignore_index=True)
        # Train first, then persist and swap; a failed training cannot corrupt the active model.
        engine = RecommendationEngine(rows, self.catalog)
        with sqlite3.connect(self.db) as db:
            db.execute('INSERT OR REPLACE INTO ratings VALUES (?,?,?,?)',(self.dataset_id,uid,pid,rating))
        self.ratings,self.engine=rows,engine
        self.version=hashlib.sha256(rows.to_csv(index=False).encode()).hexdigest()[:12]

    def restore_sample(self):
        if self.active.exists(): self.active.unlink()
        self.load()

    def load_activity(self):
        self.activity = {}
        if self.dataset_id == self.sample_id:
            for row in csv.DictReader(StringIO(DEFAULT_ACTIVITY_CSV)):
                self.activity[(row['user_id'], row['product_id'], row['kind'])] = True
        with sqlite3.connect(self.db) as db:
            for uid, pid, kind, active in db.execute('SELECT user_id,product_id,kind,active FROM activity WHERE dataset_id=?', (self.dataset_id,)):
                self.activity[(uid, pid, kind)] = bool(active)

    @property
    def customers(self):
        return sorted(set(self.engine.users) | {uid for (uid, pid, kind), active in self.activity.items() if active and pid in self.by_id})

    def signals(self, uid):
        result = {}
        for (user, pid, kind), active in self.activity.items():
            if user == uid and active and pid in self.by_id:
                result.setdefault(pid, set()).add(kind)
        return result

    def set_activity(self, uid, pid, kind, active):
        with sqlite3.connect(self.db) as db:
            db.execute('INSERT OR REPLACE INTO activity VALUES (?,?,?,?,?)', (self.dataset_id, uid, pid, kind, int(active)))
        self.activity[(uid, pid, kind)] = active

    @property
    def state_version(self):
        active = sorted(key for key, enabled in self.activity.items() if enabled)
        return hashlib.sha256((self.version + repr(active)).encode()).hexdigest()[:12]

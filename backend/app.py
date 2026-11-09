"""Flask JSON API and same-origin frontend. Run with python run.py."""
from pathlib import Path
from threading import RLock
from io import StringIO
import csv
import math
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response, g
from werkzeug.exceptions import HTTPException
from .store import Store


def create_app(root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    app = Flask(__name__,static_folder=None)
    app.config['MAX_CONTENT_LENGTH']=3*1024*1024
    store = Store(root)
    app.config['STORE']=store
    lock = RLock()

    API_PREFIXES = ('/api/', '/overview', '/users', '/matrix', '/profile', '/catalog', '/recommendations', '/ratings', '/dataset', '/activity', '/data/')

    def is_api():
        return any(request.path.startswith(p) for p in API_PREFIXES)

    @app.before_request
    def before():
        if is_api():
            lock.acquire();g.locked=True

    @app.teardown_request
    def after(error):
        if getattr(g,'locked',False):lock.release()

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options']='nosniff'
        if is_api():response.headers['Cache-Control']='no-store'
        return response

    @app.errorhandler(ValueError)
    def invalid(error):return jsonify(error=str(error)),400

    @app.errorhandler(HTTPException)
    def http_error(error):
        if is_api():
            return jsonify(error='Upload is too large. Maximum request size is 3 MB.' if error.code==413 else error.description),error.code
        return error

    @app.errorhandler(Exception)
    def handle_exception(e):
        import traceback
        traceback.print_exc()
        if isinstance(e, HTTPException):
            return jsonify(error=e.description), e.code
        return jsonify(error=f'Internal Server Error: {str(e)}', type=type(e).__name__), 500

    def integer(key, default, low, high):
        try:v=int(request.args.get(key,default))
        except (TypeError,ValueError):raise ValueError(f'{key} must be an integer.')
        if not low<=v<=high:raise ValueError(f'{key} must be between {low} and {high}.')
        return v

    def catalog_record(pid):
        p=dict(store.by_id[pid]);i=store.engine.item_index[pid]
        p.update(average_rating=round(float(store.engine.averages[i]),2) if store.engine.counts[i] else None,rating_count=int(store.engine.counts[i]))
        return p

    @app.get('/api/overview')
    @app.get('/overview')
    def overview():
        r,e=store.ratings,store.engine
        histogram=r.rating.round().clip(1,5).value_counts()
        counts=r.groupby('product_id').size().sort_values(ascending=False).head(6)
        return jsonify(dataset=store.info,version=store.state_version,report=store.report,
                       users=len(store.customers),products=len(e.items),ratings=len(r),average_rating=round(float(r.rating.mean()),2),
                       sparsity=round((1-e.matrix.nnz/(e.matrix.shape[0]*e.matrix.shape[1]))*100,2),
                       training_ms=e.training_ms,
                       categories=sorted(set(p['category'] for p in store.catalog)),
                       rating_distribution=[{'rating':i,'count':int(histogram.get(i,0))} for i in range(1,6)],
                       popular_products=[{'name':store.by_id[pid]['name'],'count':int(count)} for pid,count in counts.items()],
                       activity_events=sum(bool(active) for active in store.activity.values()),
                       customer_rating_counts=[{'user_id':uid, 'ratings':int((r.user_id==uid).sum())} for uid in store.customers])

    @app.get('/api/users')
    @app.get('/users')
    def users():
        q=request.args.get('q','').lower()
        counts=store.ratings.groupby('user_id').rating.agg(['size','mean'])
        data=[{'user_id':u,'ratings':int(counts.loc[u,'size']) if u in counts.index else 0,
               'average':round(float(counts.loc[u,'mean']),2) if u in counts.index else None,
               'cart':sum('cart' in kinds for kinds in store.signals(u).values()),
               'wishlist':sum('wishlist' in kinds for kinds in store.signals(u).values())} for u in store.customers if q in u.lower()]
        return jsonify(users=data[:500],total=len(data),limit=500)

    @app.get('/api/matrix')
    @app.get('/matrix')
    def matrix():
        users=store.engine.users[:12];items=store.engine.items[:16]
        cells=store.engine.matrix[:len(users),:len(items)].toarray().tolist()
        return jsonify(users=users,products=items,values=cells)

    @app.get('/api/profile')
    @app.get('/profile')
    def profile():
        uid=request.args.get('user_id','NEW')
        rows=store.ratings[store.ratings.user_id==uid]
        history=[]
        for row in rows.itertuples(index=False):
            p=catalog_record(row.product_id);p['user_rating']=float(row.rating);history.append(p)
        signals=store.signals(uid)
        return jsonify(user_id=uid,cold_start=rows.empty,ratings=history,
                       cart=[catalog_record(pid) for pid,kinds in sorted(signals.items()) if 'cart' in kinds],
                       wishlist=[catalog_record(pid) for pid,kinds in sorted(signals.items()) if 'wishlist' in kinds])

    @app.get('/api/catalog')
    @app.get('/catalog')
    def catalog():
        query=request.args.get('q','').lower();cat=request.args.get('category','All')
        records=[catalog_record(p['product_id']) for p in store.catalog if (cat=='All' or cat==p['category']) and query in (p['name']+' '+p['product_id']+' '+p['category']).lower()]
        return jsonify(products=records[:1000],total=len(records),limit=1000)

    def recommendation_data():
        uid=request.args.get('user_id','U001');method=request.args.get('method','activity')
        n=integer('top_n',8,1,30);minimum=integer('min_ratings',0,0,10000)
        signals=store.signals(uid)
        rows,neighbors,fallback=store.engine.recommend(uid,method,len(store.engine.items),minimum,signals)
        cold=uid not in store.engine.user_index
        matching=method=='activity' and fallback is None
        output=[]
        for row in rows:
            p=dict(store.by_id[row['product_id']]);p.update(row);p['method']=fallback or method
            if matching:
                source=p['source'];source['name']=store.by_id[source['product_id']]['name']
                p['reason']=f"Similar in name or category to {source['name']} ({source['product_id']}) in your {source['kind']}. Cart signals have weight 2; wishlist signals have weight 1."
            elif fallback or method=='popularity':p['reason']=f'Weighted community rating from {p["rating_count"]} ratings.'
            else:p['reason']=f'Similarity-weighted predictions from up to {len(neighbors)} nearby customers; unrated-by-neighbor items use popularity.'
            output.append(p)
        return {'user_id':uid,'method':method,'effective_method':fallback or method,'cold_start':cold,
                'fallback':fallback,'fallback_reason':('Add cart or wishlist products with names or categories to personalize your recommendations.' if method=='activity' else 'Not enough overlapping rating history.') if fallback else None,
                'score_type':'similarity' if matching else 'rating',
                'products':output[:n],'eligible_products':len(output),'version':store.state_version,
                'neighbors':[dict(x,user_id=store.engine.users[x['index']]) for x in neighbors[:8]],
                'explanation':('Scores show weighted content similarity from 0 to 100%, not purchase probability or predicted star ratings.' if matching else 'Scores are estimated 1–5 ratings, not confidence percentages.') + ' Already-rated, cart and wishlist products are excluded.'}

    @app.get('/api/recommendations')
    @app.get('/recommendations')
    def recommendations():return jsonify(recommendation_data())

    @app.get('/api/recommendations/export')
    @app.get('/recommendations/export')
    def export():
        data=recommendation_data();f=StringIO();writer=csv.writer(f)
        writer.writerow(['product_id','name','method','score','score_type','rating_count'])
        def safe(s):
            s=str(s);return "'"+s if s.startswith(('=','+','-','@')) else s
        for p in data['products']:writer.writerow([safe(p['product_id']),safe(p['name']),data['effective_method'],p['score'],data['score_type'],p['rating_count']])
        return Response(f.getvalue(),mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=recommendations.csv'})

    @app.post('/api/activity')
    @app.post('/activity')
    def activity():
        body=request.get_json(silent=True)
        if not isinstance(body,dict):raise ValueError('Send a JSON object.')
        uid=body.get('user_id');pid=body.get('product_id');kind=body.get('kind');active=body.get('active')
        if not isinstance(uid,str) or not uid.strip() or len(uid.strip())>160:raise ValueError('Customer ID must be 1–160 characters.')
        if not isinstance(pid,str) or pid not in store.by_id:raise ValueError('Product ID is not in the active catalog.')
        if kind not in ('cart','wishlist'):raise ValueError('Choose cart or wishlist.')
        if not isinstance(active,bool):raise ValueError('active must be true or false.')
        store.set_activity(uid.strip(),pid,kind,active)
        return jsonify(message='Shopping activity saved.',version=store.state_version)

    @app.post('/api/ratings')
    @app.post('/ratings')
    def rate():
        body=request.get_json(silent=True) or {}
        uid=str(body.get('user_id','')).strip();pid=str(body.get('product_id','')).strip()
        if not uid or len(uid)>160:raise ValueError('Customer ID must be 1–160 characters.')
        if pid not in store.by_id:raise ValueError('Product ID is not in the active catalog.')
        try:rating=float(body.get('rating'))
        except (ValueError,TypeError):raise ValueError('Choose a rating from 1 to 5.')
        if not math.isfinite(rating) or not 1<=rating<=5:raise ValueError('Choose a rating from 1 to 5.')
        store.rate(uid,pid,rating)
        return jsonify(message='Rating saved to SQLite. Models retrained.',version=store.state_version)

    @app.post('/api/dataset')
    @app.post('/dataset')
    def upload():
        file=request.files.get('ratings')
        if not file:raise ValueError('Choose a ratings CSV first.')
        metadata=request.files.get('catalog')
        store.import_dataset(file.read(),metadata.read() if metadata and metadata.filename else None,Path(file.filename).name[:100])
        return jsonify(message='Dataset cleaned, loaded, and models trained.',report=store.report)

    @app.post('/api/dataset/sample')
    @app.post('/dataset/sample')
    def restore():
        store.restore_sample();return jsonify(message='Sample dataset restored. Saved sample ratings are retained.')

    @app.get('/api/data/<kind>')
    @app.get('/data/<kind>')
    def download(kind):
        if kind not in ['ratings','products']:return jsonify(error='Unknown sample file'),404
        filepath = root/'data'/f'{kind}.csv'
        if filepath.exists():
            return send_from_directory(root/'data',f'{kind}.csv',as_attachment=True)
        from .embedded_data import DEFAULT_RATINGS_CSV, DEFAULT_PRODUCTS_CSV
        csv_content = DEFAULT_RATINGS_CSV if kind == 'ratings' else DEFAULT_PRODUCTS_CSV
        return Response(csv_content, mimetype='text/csv', headers={'Content-Disposition': f'attachment; filename={kind}.csv'})

    @app.get('/')
    def index():
        filepath = root/'public'/'index.html'
        if filepath.exists():
            return send_from_directory(root/'public','index.html')
        return send_from_directory(Path.cwd()/'public','index.html')

    @app.get('/<path:path>')
    def frontend(path):
        target = root/'public'/path
        if target.is_file():
            return send_from_directory(root/'public',path)
        alt_target = Path.cwd()/'public'/path
        if alt_target.is_file():
            return send_from_directory(Path.cwd()/'public',path)
        if path.startswith('api/') or is_api():
            return jsonify(error='Endpoint not found'),404
        if (root/'public'/'index.html').exists():
            return send_from_directory(root/'public','index.html')
        return send_from_directory(Path.cwd()/'public','index.html')

    return app

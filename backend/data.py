"""CSV validation and catalog loading, kept separate from recommendation math."""
from io import BytesIO
from pathlib import Path
import hashlib
import pandas as pd
import numpy as np

MAX_ROWS = 200_000
ALIASES = {'userId': 'user_id', 'user': 'user_id', 'productId': 'product_id',
           'prod_id': 'product_id', 'item_id': 'product_id', 'asin': 'product_id', 'ratings': 'rating'}


def read_ratings(source):
    raw = source.read() if hasattr(source, 'read') else Path(source).read_bytes()
    if not raw.strip():
        raise ValueError('The ratings file is empty.')
    try:
        frame = pd.read_csv(BytesIO(raw), dtype=str, nrows=MAX_ROWS+1)
        frame.columns = [str(c).strip().lstrip('\ufeff') for c in frame.columns]
        frame = frame.rename(columns=ALIASES)
        if not {'user_id', 'product_id', 'rating'} <= set(frame.columns):
            frame = pd.read_csv(BytesIO(raw), header=None, dtype=str, nrows=MAX_ROWS+1)
            if frame.shape[1] not in (3, 4):
                raise ValueError('Use user_id,product_id,rating headers or a headerless Amazon CSV with 3 or 4 columns.')
            # Distinguish genuinely headerless records from unknown column headers.
            if pd.isna(pd.to_numeric(frame.iloc[0, 2], errors='coerce')):
                raise ValueError('Unknown headers. Required: user_id, product_id, rating. prod_id is also supported.')
            frame = frame.rename(columns={0: 'user_id', 1: 'product_id', 2: 'rating'})
    except (pd.errors.ParserError, UnicodeError, pd.errors.EmptyDataError) as exc:
        raise ValueError('Cannot read the CSV. Use a UTF-8 comma-separated file.') from exc
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate or conflicting column headers.')
    if len(frame) > MAX_ROWS:
        raise ValueError(f'Local demo limit: {MAX_ROWS:,} rows. Please sample your dataset first.')
    frame = frame[['user_id', 'product_id', 'rating']].copy()
    original_count = len(frame)
    for col in ['user_id', 'product_id']:
        frame[col] = frame[col].fillna('').str.strip()
    frame['rating'] = pd.to_numeric(frame.rating, errors='coerce')
    valid = frame.user_id.ne('') & frame.product_id.ne('') & frame.user_id.str.len().le(160) & frame.product_id.str.len().le(160)
    valid &= np.isfinite(frame.rating) & frame.rating.between(1, 5)
    frame = frame[valid].copy()
    dropped = original_count-len(frame)
    duplicate_count = int(frame.duplicated(['user_id', 'product_id']).sum())
    frame = frame.groupby(['user_id', 'product_id'], as_index=False, sort=True).rating.mean()
    if frame.empty:
        raise ValueError('No valid ratings remain. IDs must be nonempty and ratings must be finite numbers from 1 to 5.')
    if frame.user_id.nunique() > 20_000 or frame.product_id.nunique() > 10_000:
        raise ValueError('Local demo limit: 20,000 users and 10,000 products. Please sample your dataset.')
    digest = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()[:20]
    return frame, {'input_rows': original_count, 'invalid_rows_removed': dropped,
                   'duplicate_pairs_averaged': duplicate_count, 'clean_rows': len(frame), 'dataset_id': digest}


def read_catalog(source, product_ids):
    catalog = pd.read_csv(source, dtype=str) if source else pd.DataFrame(columns=['product_id'])
    catalog = catalog.rename(columns=ALIASES)
    if 'product_id' not in catalog:
        raise ValueError('Product metadata CSV needs a product_id column.')
    catalog['product_id'] = catalog.product_id.fillna('').str.strip()
    if catalog.product_id.eq('').any() or catalog.product_id.str.len().gt(160).any():
        raise ValueError('Product metadata IDs must be 1–160 characters.')
    if len(set(product_ids) | set(catalog.product_id)) > 10_000:
        raise ValueError('Local demo limit: 10,000 products. Please sample your catalog.')
    if catalog.product_id.duplicated().any():
        raise ValueError('Product metadata must contain one row per product_id.')
    records = catalog.set_index('product_id').to_dict('index')
    result = []
    for pid in sorted(set(product_ids) | set(records)):
        row = records.get(pid, {})
        def value(key, fallback):
            v = row.get(key)
            return fallback if v is None or pd.isna(v) or str(v).strip() == '' else str(v)
        result.append({'product_id': pid, 'name': value('name', f'Product {pid}'),
                       'category': value('category', 'Uncategorized')})
    return result

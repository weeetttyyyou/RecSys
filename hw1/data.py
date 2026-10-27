import gzip
import json
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCES = {'reviews.json.gz': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz', 'books.json.gz': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_books_comics_graphic.json.gz'}


def download():
    folder = ROOT / 'data'
    folder.mkdir(exist_ok=True)
    for name, url in SOURCES.items():
        path = folder / name
        if not path.exists():
            part = path.with_suffix('.part')
            urlretrieve(url, part)
            part.rename(path)


def records(name):
    with gzip.open(ROOT / 'data' / name, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def load_ratings():
    download()
    cache = ROOT / 'data' / 'ratings.csv'
    if cache.exists():
        return pd.read_csv(cache, dtype={'user_id': str, 'item_id': str})
    rows = [(r.get('user_id'), r.get('book_id'), r.get('rating'), r.get('date_updated'))
            for r in records('reviews.json.gz')]
    ratings = pd.DataFrame(rows, columns=['user_id', 'item_id', 'rating', 'updated'])
    dates = pd.to_datetime(ratings.pop('updated'), format='%a %b %d %H:%M:%S %z %Y', utc=True, errors='coerce')
    ratings['timestamp'] = dates.astype('int64') // 10**9
    ratings.loc[dates.isna(), 'timestamp'] = np.nan
    ratings.to_csv(cache, index=False)
    return ratings


def prepare():
    raw = load_ratings()
    missing = raw.isna().sum().to_dict()
    valid = raw.user_id.notna() & raw.item_id.notna() & raw.rating.between(1, 5) & raw.timestamp.gt(0)
    clean = raw[valid].sort_values(['timestamp', 'user_id', 'item_id'], kind='stable')
    duplicates = int(clean.duplicated(['user_id', 'item_id']).sum())
    clean = clean.drop_duplicates(['user_id', 'item_id']).copy()
    borders = clean.timestamp.quantile([0.8, 0.9], interpolation='nearest').to_numpy()
    clean['period'] = np.where(clean.timestamp < borders[0], 'train', np.where(clean.timestamp < borders[1], 'validation', 'test'))
    train_raw = clean[clean.period == 'train']
    popularity = train_raw.groupby('item_id').size().rename('count').reset_index()
    catalog = sorted(popularity.sort_values(['count', 'item_id'], ascending=[False, True]).head(3500).item_id)
    activity = train_raw[train_raw.item_id.isin(catalog)].groupby('user_id').size().rename('count').reset_index()
    activity = activity[activity['count'] >= 4].sort_values(['count', 'user_id'], ascending=[False, True])
    user_ids = sorted(activity.head(12000).user_id)
    chosen = clean[clean.user_id.isin(user_ids) & clean.item_id.isin(catalog)].copy()
    chosen['user'] = chosen.user_id.map({key: index for index, key in enumerate(user_ids)})
    chosen['item'] = chosen.item_id.map({key: index for index, key in enumerate(catalog)})
    parts = [chosen[chosen.period == name].copy() for name in ['train', 'validation', 'test']]
    excluded = []
    for name in ['validation', 'test']:
        period = clean[clean.period == name]
        kept = period.user_id.isin(user_ids) & period.item_id.isin(catalog)
        excluded.append({'period': name, 'before_filter': len(period), 'retained': int(kept.sum()),
                         'new_users': period.loc[~period.user_id.isin(train_raw.user_id), 'user_id'].nunique(),
                         'new_items': period.loc[~period.item_id.isin(train_raw.item_id), 'item_id'].nunique()})
    catalog_set = set(catalog)
    books = [{'item_id': r['book_id'], 'title': r.get('title'), 'pages': r.get('num_pages'),
              'year': r.get('publication_year'), 'language': r.get('language_code')}
             for r in records('books.json.gz') if r.get('book_id') in catalog_set]
    metadata = pd.DataFrame(books).drop_duplicates('item_id').set_index('item_id').reindex(catalog)
    metadata['pages'] = pd.to_numeric(metadata.pages, errors='coerce').where(lambda x: x.between(1, 5000))
    metadata['year'] = pd.to_numeric(metadata.year, errors='coerce').where(lambda x: x.between(1800, 2017))
    info = {'raw_rows': len(raw), 'missing': missing, 'invalid_rows': int((~valid).sum()),
            'duplicate_pairs': duplicates, 'clean_rows': len(clean), 'rows': len(chosen),
            'users': len(user_ids), 'items': len(catalog),
            'train_rows': len(parts[0]), 'validation_rows': len(parts[1]), 'test_rows': len(parts[2]),
            'cutoffs': [int(x) for x in borders], 'excluded': excluded,
            'sparsity': 1 - len(chosen) / (len(user_ids) * len(catalog))}
    return raw, chosen, *parts, metadata, info


if __name__ == '__main__':
    *_, info = prepare()
    print(json.dumps(info, ensure_ascii=False, indent=2))

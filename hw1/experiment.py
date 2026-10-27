import itertools
import json
import time
from pathlib import Path
import pandas as pd
from threadpoolctl import threadpool_limits
from data import prepare
from ranking import Evaluation
from book_features import examples
from models.user_cf import UserCF
from models.implicit_als import ImplicitALS
from models.graded_als import GradedALS
from models.ease import EASE
from models.fm import BookFM

ROOT = Path(__file__).resolve().parent
MODELS = {'UserCF': UserCF, 'iALS': ImplicitALS, 'GradedALS': GradedALS, 'EASE': EASE, 'FM': BookFM}


def settings():
    variants = [('UserCF', {'neighbors': k}) for k in [20, 60, 120]]
    variants += [('EASE', {'regularization': x}) for x in [50, 250, 1000]]
    for name in ['iALS', 'GradedALS']:
        variants += [(name, {'dimensions': size, 'regularization': reg})
                     for size, reg in itertools.product([24, 48], [.5, 5.])]
    variants += [('FM', {'factors': size, 'reg': reg}) for size, reg in itertools.product([16, 32], [1e-5, 1e-4])]
    return variants


def run():
    _, _, train, validation, test, metadata, info = prepare()
    shape = (info['users'], info['items'])
    validation_set = Evaluation(train, validation, shape)
    search = []
    prepared = None
    with threadpool_limits(limits=1):
        for name, parameters in settings():
            print('Validation:', name, parameters, flush=True)
            model = MODELS[name](**parameters)
            started = time.perf_counter()
            if name == 'FM':
                if prepared is None:
                    prepared = examples(train, shape, seed=17, metadata=metadata)
                model.fit(train, shape, prepared=prepared,
                          validation=lambda fitted: validation_set.evaluate(fitted)['NDCG@10'])
            else:
                model.fit(train, shape)
            elapsed = time.perf_counter() - started
            row = dict(model=name, parameters=parameters, seconds=elapsed, **validation_set.evaluate(model))
            if name == 'FM':
                row.update(best_epoch=model.best_epoch, history=model.history)
            search.append(row)
            print({k: v for k, v in row.items() if k != 'history'}, flush=True)
        best = {name: max((row for row in search if row['model'] == name), key=lambda row: row['NDCG@10'])
                for name in MODELS}
        chosen = {name: row['parameters'].copy() for name, row in best.items()}
        chosen['FM']['epochs'] = best['FM']['best_epoch']
        chosen['GradedALS_matched'] = chosen['iALS'].copy()
        chosen['FM_ID'] = {**chosen['FM'], 'metadata': False}
        full_train = pd.concat([train, validation], ignore_index=True)
        test_set = Evaluation(full_train, test, shape)
        fm_data = None
        final = []
        cached = {}
        example = None
        for label, parameters in chosen.items():
            name = 'FM' if label == 'FM_ID' else label.replace('_matched', '')
            key = (name, json.dumps(parameters, sort_keys=True))
            if key in cached:
                final.append({**cached[key], 'model': label})
                continue
            print('Test:', label, parameters, flush=True)
            model = MODELS[name](**parameters)
            started = time.perf_counter()
            if name == 'FM':
                if fm_data is None:
                    fm_data = examples(full_train, shape, seed=17, metadata=metadata)
                model.fit(full_train, shape, prepared=fm_data)
            else:
                model.fit(full_train, shape)
            elapsed = time.perf_counter() - started
            row = dict(model=label, parameters=parameters, seconds=elapsed, **test_set.evaluate(model))
            final.append(row)
            cached[key] = row
            if name == 'EASE':
                example = {'user': int(test_set.users[0]),
                           'items': model.recommend(test_set.users[:1], test_set.seen)[0].tolist()}
            print(row, flush=True)
    report = dict(seed=17, data=info, validation_users=len(validation_set.users), test_users=len(test_set.users),
                  search=search, chosen=chosen, final=final, example=example)
    folder = ROOT / 'results'
    folder.mkdir(exist_ok=True)
    (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    pd.DataFrame([{k: json.dumps(v) if k == 'parameters' else v for k, v in row.items() if k != 'history'}
                  for row in search]).to_csv(folder / 'validation.csv', index=False)
    pd.DataFrame([{**row, 'parameters': json.dumps(row['parameters'])} for row in final]).to_csv(folder / 'test.csv', index=False)
    return report


if __name__ == '__main__':
    run()

# Домашнее задание 1

Коротков Станислав Игоревич

Сравнение пяти моделей на [Goodreads, Comics & Graphic](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html). Код находится в `models/`, результаты и выводы в `experiments.ipynb`.

Для запуска использовал Python 3.9.6:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_notebook.py
```

Данные скачиваются автоматически и не входят в архив. По умолчанию используются сохраненные результаты, для обучения заново нужно поставить `RETRAIN = True` в ноутбуке или запустить `python experiment.py`.

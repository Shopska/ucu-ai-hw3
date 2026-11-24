# HW3: AI-Generated Text Detection (UCU AI 2026)

Авторка: Шопська Анастасія (Kaggle: `shopska`)

Класифікація текстів (EN/DE) на `human`, `ai` і `fusion`. Метрика: macro-F1.

## Фінальна модель

Стекінг (логістична регресія) поверх TF-IDF + LogReg і XLM-R, дотренованої на 80k текстах:
**0.925** macro-F1 на фіксованому val (out-of-fold), **0.927** на публічному лідерборді.

Обрані сабміти на Kaggle:
1. стекінг на XLM-R 80k (`hw3-03-ensemble`, `submission_stacking.csv`): public 0.927;
2. стекінг на XLM-R 32k (`hw3-03-ensemble_32k`, `submission_stacking.csv`): public 0.910.

## Файли

| Файл | Що робить | Залізо | Час |
|---|---|---|---|
| `HW_3_report_Shopska.pdf` | технічний звіт | | |
| `hw3-01-eda-baseline.ipynb` | фіксований спліт (`split.csv`), EDA, TF-IDF бейзлайн | CPU | ~25 хв |
| `hw3-02-xlmr.ipynb` | XLM-R на 32k текстах; head-only / head+tail / chunks | GPU T4 (Colab) | ~1 год |
| `hw3-02b-xlmr-80k.ipynb` | XLM-R на 80k текстах, 1 епоха | GPU T4 (Kaggle) | ~1.2 год |
| `hw3-03-ensemble_32k.ipynb` | ансамбль, зсуви класів, стекінг, абляція на XLM-R 32k | CPU | хвилини |
| `hw3-03-ensemble.ipynb` | те саме на XLM-R 80k, фінальні сабміти | CPU | хвилини |

## Як відтворити

Ноутбуки запускаються на Kaggle по черзі; кожен бере як inputs дані змагання і outputs попередніх:

1. `hw3-01-eda-baseline` → `split.csv`, `tfidf_lr_{val,test}_proba.npy`, `submission_tfidf.csv`.
2. `hw3-02-xlmr` (input: `hw3-01-eda-baseline`) → `xlmr_*_proba.npy`, `val_ids.csv`.
   Запускався в Colab: перша клітинка завантажує дані через Kaggle API (`kaggle.json`); на Kaggle її треба видалити.
3. `hw3-02b-xlmr-80k` (input: `hw3-01-eda-baseline`) → ті самі файли для моделі 80k.
4. `hw3-03-ensemble_32k` (inputs: `hw3-01-eda-baseline` + outputs `hw3-02-xlmr`) і
   `hw3-03-ensemble` (inputs: `hw3-01-eda-baseline` + outputs `hw3-02b-xlmr-80k`) → `submission_*.csv`.

Seed усюди 42. Усі моделі порівнюються на одному фіксованому val (8 000 текстів).

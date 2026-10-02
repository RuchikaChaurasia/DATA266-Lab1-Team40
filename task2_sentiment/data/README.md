# Task 2 data — Yelp Polarity (shared raw dataset)
(The lab brief's folder sketch labels this "IMDB"; the task itself uses Yelp Polarity.)

Downloaded automatically by the notebook from Hugging Face (`fancyzhx/yelp_polarity`): 560,000 train and
38,000 test reviews, label 0 = negative, 1 = positive.

Offline fallback: put the original fast.ai CSVs `train.csv` and `test.csv` (no header; label 1 = negative,
2 = positive) in this folder and set `data.local_csv_dir: task2_sentiment/data` in the member's config.

# DS_project

1. ``fake.ipynb`` is almost the same as ``M2``.
2. ``wiki.ipynb`` collects wikinews, writes them in database and generates a view (``news.csv``) which includes news contents from wikinews and fakenews.
3. ``baseline`` generates ``subnews.csv`` from ``news.csv``. Since it called 'baseline', no news is processed (clean, remove stopwordsand stem...)
4. ``transformer`` will process ``subnews.csv`` for next modelling. For saving time, a ziped, processed subnews ([``processed_sub_news.zip``](https://drive.google.com/file/d/1qDjSOm9bmLT24m4jok1rfdTEDSEIeIu0/view?usp=sharing)) dataset is provided .
5. ``transformer`` runs faster on TPU.

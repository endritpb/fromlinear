import numpy as np, pyarrow.parquet as pq
from extra.helpers import fetch

def fetch_amazon_polarity():
  from sklearn.feature_extraction.text import TfidfVectorizer
  vectorizer = TfidfVectorizer(max_features=100000, ngram_range=(1, 2), dtype=np.float32, lowercase=True)
  BASE = "https://huggingface.co/datasets/fancyzhx/amazon_polarity/resolve/main/amazon_polarity"
  parse = lambda url: (lambda t: (t['content'].to_pylist(), np.array(t['label'].to_pylist(), np.int32)))(pq.read_table(fetch(url)))
  (X_train, Y_train), (X_test, Y_test) = parse(f"{BASE}/train-00000-of-00004.parquet"), parse(f"{BASE}/test-00000-of-00001.parquet")
  return vectorizer.fit_transform(X_train), Y_train, vectorizer.transform(X_test), Y_test
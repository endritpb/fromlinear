import os, pyarrow.parquet as pq, numpy as np
from extra.helpers import fetch, BoW

vectorizer = BoW(max_features=25000, binary=True, ngram_range=(1, 2), norm="l2")
def fetch_amazon_polarity():
  BASE = "https://huggingface.co/datasets/fancyzhx/amazon_polarity/resolve/main/amazon_polarity"
  parse = lambda url, n: (lambda t: (
      t['content'][:n if n is not None else len(t['content'])].to_pylist(),
      np.array(t['label'][:n if n is not None else len(t['label'])].to_pylist(), np.int32)
    ))(pq.read_table(fetch(url)))
  X_train, Y_train = parse(f"{BASE}/train-00000-of-00004.parquet", int(os.getenv("N")) if os.getenv("N") else None)
  X_test, Y_test = parse(f"{BASE}/test-00000-of-00001.parquet", 30000)
  return vectorizer.fit_transform(X_train), Y_train, vectorizer.transform(X_test), Y_test
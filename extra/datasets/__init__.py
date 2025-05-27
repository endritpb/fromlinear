import gzip, numpy as np, pyarrow.parquet as pq
from extra.helpers import fetch

def fetch_amazon_polarity():
  from sklearn.feature_extraction.text import TfidfVectorizer
  vectorizer = TfidfVectorizer(max_features=100000, ngram_range=(1, 2), dtype=np.float32, lowercase=True)
  BASE = "https://huggingface.co/datasets/fancyzhx/amazon_polarity/resolve/main/amazon_polarity"
  parse = lambda url: (lambda t: (t['content'].to_pylist(), np.array(t['label'].to_pylist(), np.int32)))(pq.read_table(fetch(url)))
  (X_train, Y_train), (X_test, Y_test) = parse(f"{BASE}/train-00000-of-00004.parquet"), parse(f"{BASE}/test-00000-of-00001.parquet")
  return vectorizer.fit_transform(X_train), Y_train, vectorizer.transform(X_test), Y_test

def fetch_mnist():
  # https://github.com/tinygrad/tinygrad/blob/master/extra/datasets/__init__.py#L6
  BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/" 
  parse = lambda file: np.frombuffer(gzip.open(fetch(file)).read(), dtype=np.uint8).copy()
  X_train = parse(f"{BASE_URL}train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_train = parse(f"{BASE_URL}train-labels-idx1-ubyte.gz")[8:].astype(np.int8)
  X_test = parse(f"{BASE_URL}t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_test = parse(f"{BASE_URL}t10k-labels-idx1-ubyte.gz")[8:].astype(np.int8)
  return X_train, Y_train, X_test, Y_test

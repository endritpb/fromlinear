import gzip
import numpy as np
from extra.helpers import fetch

BASE_URL_MNIST = "https://storage.googleapis.com/cvdf-datasets/mnist/"
def fetch_mnist():
  parse = lambda f: np.frombuffer(gzip.open(f).read(), dtype=np.uint8).copy()
  X_train = parse(fetch(f"{BASE_URL_MNIST}train-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28*28)).astype(np.float32)
  y_train = parse(fetch(f"{BASE_URL_MNIST}train-labels-idx1-ubyte.gz"))[8:].astype(np.int8)
  X_test = parse(fetch(f"{BASE_URL_MNIST}t10k-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28*28)).astype(np.float32)
  y_test = parse(fetch(f"{BASE_URL_MNIST}t10k-labels-idx1-ubyte.gz"))[8:].astype(np.int8)
  return X_train, y_train, X_test, y_test

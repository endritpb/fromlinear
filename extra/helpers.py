import os, re, hashlib, urllib.request, platform, pathlib, tempfile, tqdm
import numpy as np
from collections import Counter
from typing import Optional, Union

CACHE_DIR:str = os.path.join(os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/Library/Caches" if platform.system() == "Darwin" else "~/.cache")), "fromlinear")

# https://github.com/tinygrad/tinygrad/blob/master/tinygrad/helpers.py#L247
def fetch(url:str, name:Optional[Union[pathlib.Path,str]]=None, allow_caching=not os.getenv("DISABLE_HTTP_CACHE")) -> pathlib.Path:
  if name is not None and (isinstance(name, pathlib.Path)): fp = pathlib.Path(name)
  else: fp = pathlib.Path(CACHE_DIR) / "downloads" / (name or hashlib.md5(url.encode('utf-8')).hexdigest())
  if not fp.is_file() or not allow_caching:
    (_dir:=fp.parent).mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url), timeout=10) as r:
      assert r.status==200, r.status
      length = int(r.headers.get('content-length', 0))
      pbar:tqdm = tqdm.tqdm(total=length, unit='B', unit_scale=True, desc=f"{url}")
      with tempfile.NamedTemporaryFile(dir=_dir, delete=False) as f:
        while chunk := r.read(16384): pbar.update(f.write(chunk))
        f.close()
        pathlib.Path(f.name).rename(fp)
      if length and (file_size:=os.stat(fp).st_size) < length:
        raise RuntimeError(f"fetch size incomplete, {file_size} < {length}")
  return fp

_TOKENIZE = lambda s: re.findall(r'\b\w+\b', s.lower())
def _ngrams(toks, lo, hi):
  for n in range(lo, hi + 1):
    for i in range(len(toks) - n + 1):
      yield ' '.join(toks[i:i + n])

class BoW:
  def __init__(self, max_features=None, *, binary=False, ngram_range=(1, 1), norm=None):
    self.mf, self.bin, self.nr, self.norm = max_features, binary, ngram_range, norm
    self.vocab = {}
  def fit(self, data):
    c = Counter()
    lo, hi = self.nr
    for s in data:
      c.update(_ngrams(_TOKENIZE(s), lo, hi))
    self.vocab = {w: i for i, (w, _) in enumerate(c.most_common(self.mf))}
  def transform(self, data):
    X = np.zeros((len(data), len(self.vocab)), np.float32)
    lo, hi = self.nr
    for i, s in enumerate(data):
      for g in _ngrams(_TOKENIZE(s), lo, hi):
        j = self.vocab.get(g)
        if j is not None:
          X[i, j] = 1 if self.bin else X[i, j] + 1
      if self.norm:
        d = np.linalg.norm(X[i]) if self.norm == 'l2' else np.abs(X[i]).sum()
        if d:
          X[i] /= d
    return X
  def fit_transform(self, data):
    self.fit(data)
    return self.transform(data)
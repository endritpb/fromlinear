import os, hashlib, urllib.request, platform, pathlib, tempfile, tqdm
from typing import Optional, Union

def sparse(x):
  if hasattr(x, "toarray"):
    return x.toarray().ravel()
  return x 

CACHE_DIR:str = os.path.join(os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/Library/Caches" if platform.system() == "Darwin" else "~/.cache")), "fromlinear")
def fetch(url:str, name:Optional[Union[pathlib.Path,str]]=None, allow_caching=not os.getenv("DISABLE_HTTP_CACHE")) -> pathlib.Path:
  # https://github.com/tinygrad/tinygrad/blob/master/tinygrad/helpers.py#L247
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

 
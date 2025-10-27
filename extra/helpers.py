import os, pathlib, tempfile, hashlib, urllib.request, tqdm

CACHE_DIR:str = os.path.join(os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/Library/Caches")), "fromlinear")

# https://github.com/tinygrad/tinygrad/blob/master/tinygrad/helpers.py#L331
def fetch(url:str, name:pathlib.Path|str|None=None, subdir:str|None=None, 
          allow_caching:bool=not os.getenv("DISABLE_HTTP_CACHE")):
  if name is not None and (isinstance(name, pathlib.Path) or "/" in name): fp = pathlib.Path(name)
  else: fp = pathlib.Path(CACHE_DIR) / "downloads" / (subdir or "") / (name or hashlib.md5(url.encode('utf-8')).hexdigest())
  if not fp.is_file() or not allow_caching:
    (_dir:=fp.parent).mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url), timeout=10) as request:
      assert request.status == 200, request.status
      length = int(request.headers.get('content-length', 0))
      progress_bar:tqdm.tqdm = tqdm.tqdm(total=length, unit="B", unit_scale=True, desc=f"{url}")
      with tempfile.NamedTemporaryFile(dir=_dir, delete=False) as file:
        while chunk := request.read(16384): progress_bar.update(file.write(chunk))
        file.close()
        pathlib.Path(file.name).rename(fp)
      progress_bar.update()
      if (file_size:=os.stat(fp).st_size) < length: 
        raise RuntimeError(f"fetch size incomplete, {file_size} < {length}")
  return fp

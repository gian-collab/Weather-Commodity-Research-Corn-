from __future__ import annotations
from pathlib import Path
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from weatherlab.provenance import ArtifactMeta, now_utc, sha256_file, write_meta

DEFAULT_TIMEOUT = 90

def session(retries: int = 5, backoff: float = 0.8) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=retries, connect=retries, read=retries, status=retries,
                  backoff_factor=backoff, status_forcelist=(429,500,502,503,504),
                  allowed_methods=frozenset(['GET','POST']))
    s.mount('https://', HTTPAdapter(max_retries=retry))
    s.headers.update({'User-Agent':'WeatherCommodityLab/0.2 research; contact=local-user'})
    return s

def get_json(url: str, params: dict | None = None, timeout: int = DEFAULT_TIMEOUT) -> dict:
    r = session().get(url, params=params, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if isinstance(j, dict) and j.get('error'):
        raise RuntimeError(f"API error from {url}: {j.get('reason', j)}")
    return j

def download(url, path, source, license=None, timeout=DEFAULT_TIMEOUT):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with session().get(url, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        tmp = path.with_suffix(path.suffix + '.part')
        with open(tmp,'wb') as f:
            for chunk in resp.iter_content(1<<20):
                if chunk: f.write(chunk)
        tmp.replace(path)
    write_meta(str(path)+'.meta.json', ArtifactMeta(source, now_utc(), license=license, sha256=sha256_file(path)))
    return path

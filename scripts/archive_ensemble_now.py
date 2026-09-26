from pathlib import Path
import sys, json
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from weatherlab.data.ecmwf_archive import archive_latest
print(json.dumps(archive_latest(),indent=2,default=str))

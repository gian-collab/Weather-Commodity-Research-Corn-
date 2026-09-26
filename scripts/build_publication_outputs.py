from pathlib import Path
import sys, json
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from weatherlab.reporting.publication import build_publication_outputs

manifest=build_publication_outputs(ROOT/'outputs/results', ROOT/'outputs/publication')
print(json.dumps(manifest, indent=2, default=str))

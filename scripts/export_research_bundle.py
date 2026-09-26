from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
target=ROOT/'outputs/weather_article_research_v0_4_2.zip'
include=[
    ROOT/'outputs/results',
    ROOT/'outputs/publication',
    ROOT/'configs',
    ROOT/'docs/FROZEN_FINDINGS.md',
    ROOT/'README.md',
    ROOT/'AUDIT.md',
    ROOT/'RESEARCH_GOVERNANCE.md',
    ROOT/'ARTICLE_BLUEPRINT.md',
    ROOT/'CHANGELOG.md',
    ROOT/'CITATION.cff',
]
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for p in include:
        if not p.exists():
            continue
        if p.is_dir():
            for f in p.rglob('*'):
                if f.is_file():
                    z.write(f,f.relative_to(ROOT))
        else:
            z.write(p,p.relative_to(ROOT))
print(target)

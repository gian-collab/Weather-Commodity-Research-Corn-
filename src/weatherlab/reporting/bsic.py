from __future__ import annotations
from pathlib import Path
import json
import matplotlib.pyplot as plt

BSIC_COLORS=["#38329A", "#8EC6FF", "#601E66", "#2F2984", "#0E0B54"]
MAX_WORD_WIDTH=7.32

def apply_bsic(fig, ax, title=None, sources=("BSIC",)):
    axes=list(ax.flat) if hasattr(ax,'flat') else [ax]
    for a in axes:
        a.set_prop_cycle(color=BSIC_COLORS)
        if title and len(axes)==1: a.set_title(title, fontweight='bold', fontstyle='italic', fontsize=12)
        a.tick_params(labelsize=9)
        a.grid(False)
    src=list(sources)
    if 'BSIC' not in src: src.insert(0,'BSIC')
    fig.text(.5,.005, ('Sources: ' if len(src)>1 else 'Source: ')+', '.join(src), ha='center', fontsize=8)
    return fig, ax

def export_figure(fig, stem, meta=None):
    stem=Path(stem); stem.parent.mkdir(parents=True, exist_ok=True)
    if fig.get_size_inches()[0] > MAX_WORD_WIDTH + .01:
        raise ValueError(f"Figure width exceeds BSIC Word limit {MAX_WORD_WIDTH}in")
    fig.savefig(stem.with_suffix('.svg'), dpi=1200, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), dpi=240, bbox_inches='tight')
    if meta: stem.with_suffix('.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
    plt.close(fig)

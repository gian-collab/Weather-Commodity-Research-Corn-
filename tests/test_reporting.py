import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from weatherlab.reporting.bsic import apply_bsic, export_figure, MAX_WORD_WIDTH

def test_bsic_export(tmp_path):
    fig,ax=plt.subplots(figsize=(6,3)); ax.plot([1,2],[2,3]); apply_bsic(fig,ax,"Test",["Unit test"]); export_figure(fig,tmp_path/"f")
    assert (tmp_path/"f.svg").exists() and (tmp_path/"f.png").exists()

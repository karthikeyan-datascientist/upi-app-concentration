"""Workflow diagrams for the final report (Figures 'End-to-end workflow' and 'Workflow by research question')."""
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif']})
SHADE = {'data': '#d9d9d9', 'proc': '#ffffff', 'model': '#a6a6a6', 'out': '#f2f2f2'}

def box(ax, x, y, w, h, text, kind, fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=SHADE[kind], ec='black', lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, linespacing=1.15)
def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=9, lw=1, color='black', shrinkA=0, shrinkB=0))

# ---------------- Figure A: end-to-end
fig, ax = plt.subplots(figsize=(6.5, 5.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 9); ax.axis('off')
rows = [
    [('1. Data source\nNPCI "UPI Applications"\nmonthly Excel tables\n(Jan 2022–Sep 2026)', 'data'), ('2. Data ingestion\nmanual download (54 files)\ncombine_apps.py parses\nlayouts and text numbers', 'proc'), ('3. Preprocessing\nduplicates, name harmonization,\nhash cells, missing months,\nTPAP flag', 'proc')],
    [('4. Storage\npanel CSV + SQLite\n(3,829 entity-months;\n223 entities)', 'data'), ('5. Exploratory analysis\nsummaries, uni/bi/multivariate\nplots, outlier screen,\nsample-size checks', 'proc'), ('6. Feature engineering\nlagged share and growth,\nlog volume, peak and\ncalendar terms, TPAP', 'proc')],
    [('7. Models by RQ\nRQ1 segmented regression\nRQ2 fractional logit\nRQ3 variance + survival\nRQ4 forecasting', 'model'), ('8. Evaluation\nHAC and bootstrap intervals,\nrolling-origin hold-out,\ntrain/test comparison, DM test', 'model'), ('9. Deployment\nmonthly monitoring brief\n(HHI, 30% window, forecast);\ndashboard planned', 'out')],
]
xs = [0.3, 3.65, 7.0]; ys = [6.6, 3.7, 0.8]; W, H = 2.7, 1.85
for r, row in enumerate(rows):
    for c, (t, k) in enumerate(row): box(ax, xs[c], ys[r], W, H, t, k)
    for c in range(2): arrow(ax, xs[c] + W, ys[r] + H / 2, xs[c + 1], ys[r] + H / 2)
arrow(ax, xs[2] + W / 2, ys[0], xs[2] + W / 2, ys[0] - 0.6); ax.plot([xs[2] + W / 2, xs[0] + W / 2], [ys[0] - 0.6, ys[0] - 0.6], color='black', lw=1); arrow(ax, xs[0] + W / 2, ys[0] - 0.6, xs[0] + W / 2, ys[1] + H)
arrow(ax, xs[2] + W / 2, ys[1], xs[2] + W / 2, ys[1] - 0.6); ax.plot([xs[2] + W / 2, xs[0] + W / 2], [ys[1] - 0.6, ys[1] - 0.6], color='black', lw=1); arrow(ax, xs[0] + W / 2, ys[1] - 0.6, xs[0] + W / 2, ys[2] + H)
ax.legend(handles=[Patch(fc=SHADE['data'], ec='black', label='Data stage'), Patch(fc=SHADE['proc'], ec='black', label='Processing stage'), Patch(fc=SHADE['model'], ec='black', label='Modelling stage'), Patch(fc=SHADE['out'], ec='black', label='Output stage')],
          loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=4, frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig('outputs/figures/workflow_end_to_end.png', dpi=220); plt.close(fig)

# ---------------- Figure B: workflow per RQ
chains = [
    ('RQ1', ['HHI series\n(51 months)', 'Add t, Post, slope,\nPaytm indicator', 'OLS segmented\nregression', 'Newey–West errors;\nplacebo breaks; DiD', 'Level and slope\ntests (5%)']),
    ('RQ2', ['15-app panel\n(711 app-months)', 'Share, peak,\nchallenger flags', 'Fractional logit,\napp effects', 'Cluster bootstrap\n(499 draws); AME', 'Peak effects on\nshare and ticket']),
    ('RQ3', ['App growth g\n(641 app-months)', 'Build g, runs\n(spells)', 'ICC, AR(1);\nKaplan–Meier, Cox', 'LR test; robust\nand clustered SE', 'Persistence and\nduration tests']),
    ('RQ4', ['Lagged features\n(8 features)', 'Rolling-origin\nsplit (15 months)', 'Naive, logit,\nXGBoost, MLP', 'RMSE, MAE, DM;\nprecision, PR-AUC', 'Best forecaster;\nfixed-split check']),
]
fig, ax = plt.subplots(figsize=(6.5, 6.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
kinds = ['data', 'proc', 'model', 'model', 'out']; bw, bh, gap = 1.45, 1.35, 0.38
for r, (lab, steps) in enumerate(chains):
    y = 8.3 - r * 2.5; ax.text(0.02, y + bh / 2, lab, fontsize=10, fontweight='bold', va='center')
    for i, t in enumerate(steps):
        x = 0.85 + i * (bw + gap); box(ax, x, y, bw, bh, t, kinds[i], fs=6.2)
        if i < 4: arrow(ax, x + bw, y + bh / 2, x + bw + gap, y + bh / 2)
ax.legend(handles=[Patch(fc=SHADE['data'], ec='black', label='Input data'), Patch(fc=SHADE['proc'], ec='black', label='Preparation'), Patch(fc=SHADE['model'], ec='black', label='Model and checks'), Patch(fc=SHADE['out'], ec='black', label='Hypothesis result')],
          loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig('outputs/figures/workflow_by_rq.png', dpi=220); plt.close(fig)

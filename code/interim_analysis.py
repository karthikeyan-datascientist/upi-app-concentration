"""Interim analysis: EDA, preliminary models for RQ1-RQ4, and figures.

Usage: python code/interim_analysis.py      (run code/combine_apps.py first)
Reads  data/processed/upi_app_panel.csv
Writes outputs/interim_results.json, outputs/tables/*.csv, outputs/figures/*.png
Random seed 42 for all stochastic steps. All values are preliminary.
"""
import json, os, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
rng = np.random.default_rng(42)
os.makedirs('outputs/figures', exist_ok=True); os.makedirs('outputs/tables', exist_ok=True)
R = {}
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'], 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
GR = ['black', '#555555', '#999999']

# ------------------------------------------------------------------ data
a = pd.read_csv('data/processed/upi_app_panel.csv', parse_dates=['month']).rename(columns={'vol_mn': 'vol', 'val_cr': 'val'})
a['tot'] = a.groupby('month').vol.transform('sum'); a['share'] = a.vol / a.tot * 100
months_all = pd.date_range('2022-01-01', '2026-09-01', freq='MS'); have = set(a.month.unique())
named = a[~a.is_others]
R['n_months'] = int(a.month.nunique()); R['n_entities'] = int(a.entity.nunique()); R['n_rows'] = len(a)
K = 1.959964 + 0.841621; Z = 1.959964

# system series
def sysrow(g):
    s = g.sort_values('vol', ascending=False)
    gv = g[g.val.notna()]
    sv = gv.val / gv.val.sum() * 100
    return pd.Series(dict(hhi=(g.share ** 2).sum(), top2=s.share.iloc[:2].sum(), top3=s.share.iloc[:3].sum(), tpap=g[g.tpap].share.sum(),
                          pp=g[g.entity == 'PhonePe'].share.sum(), gp=g[g.entity == 'Google Pay'].share.sum(), pt=g[g.entity == 'Paytm'].share.sum(),
                          tot=g.tot.iloc[0], napps=len(g), hhi_val=(sv ** 2).sum(), val_tot=gv.val.sum(), n_val=len(gv), n_all=len(g)))
S = named.groupby('month').apply(sysrow).reset_index()
S['ticket'] = a.groupby('month').val.sum().values * 1e7 / (a.groupby('month').vol.sum().values * 1e6)
S.loc[S.month.isin(pd.to_datetime(['2022-02-01', '2022-03-01'])), 'ticket'] = np.nan  # value missing for the two leaders
S['t'] = (S.month.dt.year - 2022) * 12 + S.month.dt.month - 1
brk = pd.Timestamp('2025-01-01'); t0 = int(S[S.month == brk].t.iloc[0])
S['post'] = (S.month >= brk).astype(int); S['tp'] = S.post * (S.t - t0); S['paytm'] = (S.month >= '2024-03-01').astype(int)
# value HHI unreliable when PhonePe/Google Pay value is missing (Feb-Mar 2022): drop those months for that outcome
S.loc[S.n_val < S.n_all - 3, 'hhi_val'] = np.nan
S.loc[S.month.isin(pd.to_datetime(['2022-02-01', '2022-03-01'])), 'hhi_val'] = np.nan
S.to_csv('outputs/system_series.csv', index=False)

# app-level panel: 15 largest apps with >= 24 usable months
st = named.groupby('entity').agg(mo=('month', 'nunique'), ms=('share', 'mean')).query('mo>=24').sort_values('ms', ascending=False)
top = list(st.index[:15]); R['top15'] = top
P = a[a.entity.isin(top)].copy().merge(S[['month', 't']], on='month').sort_values(['entity', 'month'])
P['logit'] = np.log((P.share / 100) / (1 - P.share / 100)); P['lnV'] = np.log(P.vol)
P['peak'] = P.month.dt.month.isin([3, 10, 11]).astype(int); P['moy'] = P.month.dt.month
P['challenger'] = (~P.entity.isin(['PhonePe', 'Google Pay'])).astype(int)
P['ticket'] = P.val * 1e7 / (P.vol * 1e6); P['lnticket'] = np.log(P.ticket)
P['prev_m'] = P.groupby('entity').month.shift(1)
P['gap'] = (P.month.dt.year - P.prev_m.dt.year) * 12 + P.month.dt.month - P.prev_m.dt.month
P['dlnV'] = P.groupby('entity').lnV.diff(); P.loc[P.gap != 1, 'dlnV'] = np.nan
sysg = a.groupby('month').vol.sum().apply(np.log).reset_index(); sysg.columns = ['month', 'lnT']; sysg['dT'] = sysg.lnT.diff()
sysg['gp'] = (sysg.month.dt.year - sysg.month.shift(1).dt.year) * 12 + sysg.month.dt.month - sysg.month.shift(1).dt.month; sysg.loc[sysg.gp != 1, 'dT'] = np.nan
P = P.merge(sysg[['month', 'dT']], on='month'); P['g'] = P.dlnV - P.dT
P['share_l'] = P.groupby('entity').share.shift(1); P.loc[P.gap != 1, 'share_l'] = np.nan
P['d_share'] = P.share - P.share_l
R['panel_n'] = len(P); R['peak_n'] = int(P.peak.sum()); R['top15_cov'] = float(P.groupby('month').share.sum().mean())

# ------------------------------------------------------------------ EDA tables
def desc(x):
    return dict(mean=x.mean(), sd=x.std(), min=x.min(), median=x.median(), p99=x.quantile(.99), max=x.max(), n=int(x.notna().sum()))
T5 = {'Monthly app-reported volume (million)': desc(S.tot), 'HHI (0-10,000)': desc(S.hhi), 'Top-two share (%)': desc(S.top2),
      'App share (%), 15 apps': desc(P.share), 'App volume (million), 15 apps': desc(P.vol), 'Ticket size (Rs), 15 apps': desc(P.ticket),
      'App-specific growth g (log points)': desc(P.g)}
R['T5'] = T5
# group table
def grp(e):
    if e in ('PhonePe', 'Google Pay'): return 'Leaders (PhonePe, Google Pay)'
    if e == 'Paytm': return 'Paytm'
    return 'Other TPAPs' if a[a.entity == e].tpap.max() else 'Bank and NPCI apps'
a['grp'] = a.entity.map(grp); P['grp'] = P.entity.map(grp)
G = []
for k, d in a.groupby('grp'):
    dp = P[P.grp == k]
    G.append(dict(group=k, entities=int(d.entity.nunique()), rows=len(d), mean_share=float(dp.share.mean()) if len(dp) else np.nan,
                  pooled_share=float(d.vol.sum() / a.vol.sum() * 100), median_ticket=float(dp.ticket.median()) if len(dp) else np.nan,
                  mean_g=float(dp.g.mean()) if len(dp) else np.nan, sd_g=float(dp.g.std()) if len(dp) else np.nan))
R['T6'] = G
pd.DataFrame(G).to_csv('outputs/tables/group_summary.csv', index=False)
# first rows of panel (for report)
R['panel_head'] = a[a.month == '2026-09-01'].sort_values('vol', ascending=False).head(6)[['month', 'entity', 'vol', 'val', 'share', 'tpap']].assign(month=lambda d: d.month.dt.strftime('%Y-%m')).to_dict('records')
# cleaning counts
R['dup_rows_combined'] = int(a.n_rows.gt(1).sum()); R['zero_vol_rows'] = int((a.vol == 0).sum()); R['val_missing'] = int(a.val.isna().sum() - 0)
q1, q3 = P.g.quantile([.25, .75]); iqr = q3 - q1; lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
out = P[(P.g < lo) | (P.g > hi)]
R['g_iqr'] = dict(q1=q1, q3=q3, lo=lo, hi=hi, n_out=len(out), n=int(P.g.notna().sum()), top_out=out.assign(a=out.g.abs()).sort_values('a', ascending=False).head(6)[['entity', 'month', 'g']].assign(month=lambda d: d.month.dt.strftime('%Y-%m')).to_dict('records'))
out_sys = S.hhi.diff().abs(); R['hhi_max_jump'] = dict(val=float(out_sys.max()), month=str(S.loc[out_sys.idxmax(), 'month'].date()))
# sys totals
R['tot_first'] = float(S.tot.iloc[0]); R['tot_last'] = float(S.tot.iloc[-1]); R['tot_max'] = float(S.tot.max()); R['tot_max_m'] = str(S.loc[S.tot.idxmax(), 'month'].date())
for k in ['hhi', 'top2', 'top3', 'pp', 'gp', 'pt', 'tpap', 'ticket']:
    R[k + '_first'] = float(S[k].iloc[0]); R[k + '_last'] = float(S[k].iloc[-1])
R['hhi_max'] = float(S.hhi.max()); R['hhi_max_m'] = str(S.loc[S.hhi.idxmax(), 'month'].date()); R['hhi_min'] = float(S.hhi.min())
R['pt_min'] = float(S.pt.min()); R['pt_min_m'] = str(S.loc[S.pt.idxmin(), 'month'].date()); R['pp_max'] = float(S.pp.max()); R['pp_min'] = float(S.pp.min()); R['gp_max'] = float(S.gp.max())
_d = S.dropna(subset=['ticket']); R['corr_vol_ticket'] = float(np.corrcoef(_d.tot, _d.ticket)[0, 1]); R['ticket_growth_pct'] = float(S.ticket.iloc[-1] / S.ticket.iloc[0] * 100 - 100)
R['pre_n'] = int((S.post == 0).sum()); R['post_n'] = int((S.post == 1).sum())
# Navi / challengers
for e in ['Navi', 'super.money', 'FamApp']:
    d = P[P.entity == e]
    R['ent_' + e] = dict(first=str(d.month.min().date()), months=int(len(d)), first_share=float(d.share.iloc[0]), last_share=float(d.share.iloc[-1]), max_share=float(d.share.max()))

# ------------------------------------------------------------------ RQ1: segmented regression
def its(y, cols, data=S, lag=2, brk_=None):
    d = data.dropna(subset=[y]).copy()
    X = sm.add_constant(d[cols]); m = sm.OLS(d[y], X).fit(); h = m.get_robustcov_results('HAC', maxlags=lag)
    ix = list(X.columns); r = m.resid; rho = np.corrcoef(r.values[:-1], r.values[1:])[0, 1]
    out = dict(n=len(d), r2=m.rsquared, sd=float(np.sqrt(m.scale)), rho=rho, b=float(m.params['post']), se=float(h.bse[ix.index('post')]), p=float(h.pvalues[ix.index('post')]),
               ci=[float(x) for x in h.conf_int()[ix.index('post')]], params={k: float(v) for k, v in m.params.items()})
    if 'tp' in cols: out.update(bs=float(m.params['tp']), bs_se=float(h.bse[ix.index('tp')]), bs_p=float(h.pvalues[ix.index('tp')]))
    return out, m
specs = {}
specs['A'], _ = its('hhi', ['t', 'post'])
specs['B'], _ = its('hhi', ['t', 'post', 'tp'])
specs['C'], mC = its('hhi', ['t', 'post', 'tp', 'paytm'])
specs['D'], _ = its('hhi', ['t', 'post', 'tp', 'paytm'], lag=4)
specs['E'], _ = its('hhi_val', ['t', 'post', 'tp', 'paytm'])
specs['F'], _ = its('top2', ['t', 'post', 'tp', 'paytm'])
# sensitivity: drop Feb-Jun 2024 transition months
Sx = S[~S.month.between('2024-02-01', '2024-06-01')]
specs['G'], _ = its('hhi', ['t', 'post', 'tp'], data=Sx)
R['specs'] = specs
# exploratory Dec 2022 event
S2 = S.copy(); b2 = pd.Timestamp('2022-12-01'); t2 = int(S2[S2.month == b2].t.iloc[0])
S2['post'] = (S2.month >= b2).astype(int); S2['tp'] = S2.post * (S2.t - t2)
R['dec22'], _ = its('hhi', ['t', 'post', 'tp', 'paytm'], data=S2); R['dec22_pre'] = int((S2.post == 0).sum())
# placebo-in-time over candidate breaks (>=10 usable months each side)
cands = []
for i in range(10, len(S) - 10):
    bm = S.month.iloc[i]; d = S.copy(); d['post'] = (d.month >= bm).astype(int); d['tp'] = d.post * (d.t - int(S.t.iloc[i]))
    o, _ = its('hhi', ['t', 'post', 'tp', 'paytm'], data=d); cands.append((bm, o['b'], o['p']))
pl = pd.DataFrame(cands, columns=['month', 'b', 'p']); pl['rank'] = pl.b.abs().rank(ascending=False)
R['placebo'] = dict(n=len(pl), rank_jan25=float(pl[pl.month == brk]['rank'].iloc[0]), b_jan25=float(pl[pl.month == brk].b.iloc[0]),
                    top3=pl.sort_values('rank').head(3).assign(month=lambda d: d.month.dt.strftime('%Y-%m')).to_dict('records'))
pl.to_csv('outputs/tables/placebo_breaks.csv', index=False)
# sample size inputs (RQ1)
sd, rho = specs['B']['sd'], specs['B']['rho']; vif = (1 + rho) / (1 - rho); n1 = 4 * vif * (Z * sd / 100) ** 2
se_a = sd * np.sqrt(vif * (1 / R['pre_n'] + 1 / R['post_n']))
R['ss1'] = dict(sd=sd, rho=rho, vif=vif, n=n1, n_ceil=int(np.ceil(n1)), se_ach=se_a, hw=Z * se_a, mde=K * se_a, mean_pre=float(S[S.month < brk].hhi.mean()))

# ------------------------------------------------------------------ RQ2: fractional logit with app fixed effects
def frac_fit(df, inter=False, peakvar='peak'):
    f = f'share_f ~ C(entity) + t + {peakvar}' + (f' + {peakvar}:challenger' if inter else '')
    d = df.copy(); d['share_f'] = d.share / 100
    return smf.glm(f, d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.entity)[0]})
def ame(m, df, var='peak', inter=False):
    d1 = df.copy(); d0 = df.copy(); d1[var] = 1; d0[var] = 0
    return float(((m.predict(d1) - m.predict(d0)).mean()) * 100)
def boot_ame(df, inter=False, B=499, var='peak'):
    ents = df.entity.unique(); vals = []; vals_i = []
    for _ in range(B):
        pick = rng.choice(ents, len(ents), replace=True)
        parts = [df[df.entity == e].assign(entity=f'{e}_{k}') for k, e in enumerate(pick)]
        d = pd.concat(parts);
        try:
            m = frac_fit(d, inter, var);
            if inter:
                d1 = d.copy(); d0 = d.copy(); d1[var] = 1; d0[var] = 0
                ch = d.challenger == 1
                vals.append(float(((m.predict(d1) - m.predict(d0))[ch].mean()) * 100)); vals_i.append(float(((m.predict(d1) - m.predict(d0))[~ch].mean()) * 100))
            else: vals.append(ame(m, d, var))
        except Exception: continue
    return np.array(vals), np.array(vals_i)
Pm = P.dropna(subset=['share']).copy()
m2a = frac_fit(Pm); a2 = ame(m2a, Pm); b2a, _ = boot_ame(Pm)
m2b = frac_fit(Pm, inter=True)
d1 = Pm.copy(); d0 = Pm.copy(); d1['peak'] = 1; d0['peak'] = 0; diff = (m2b.predict(d1) - m2b.predict(d0)) * 100
a_ch = float(diff[Pm.challenger == 1].mean()); a_ld = float(diff[Pm.challenger == 0].mean())
b_ch, b_ld = boot_ame(Pm, inter=True)
# alt peak definitions
Pm['mar'] = (Pm.month.dt.month == 3).astype(int); Pm['fest'] = Pm.month.dt.month.isin([10, 11]).astype(int)
m2c = frac_fit(Pm, peakvar='mar'); m2d = frac_fit(Pm, peakvar='fest')
# ticket size
Pt = Pm.dropna(subset=['lnticket'])
mt = smf.ols('lnticket ~ C(entity) + t + peak', Pt).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(Pt.entity)[0]})
mti = smf.ols('lnticket ~ C(entity) + t + peak + peak:challenger', Pt).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(Pt.entity)[0]})
q = lambda lst: [float(np.nanpercentile(lst, 2.5)), float(np.nanpercentile(lst, 97.5))]
R['rq2'] = dict(n=len(Pm), n_peak=int(Pm.peak.sum()), peak_b=float(m2a.params['peak']), peak_se=float(m2a.bse['peak']), peak_p=float(m2a.pvalues['peak']), ame=a2, ame_ci=q(b2a), boot_n=len(b2a),
    int_b=float(m2b.params['peak:challenger']), int_se=float(m2b.bse['peak:challenger']), int_p=float(m2b.pvalues['peak:challenger']), ame_ch=a_ch, ame_ld=a_ld, ame_ch_ci=q(b_ch), ame_ld_ci=q(b_ld),
    mar_b=float(m2c.params['mar']), mar_se=float(m2c.bse['mar']), mar_p=float(m2c.pvalues['mar']), mar_ame=ame(m2c, Pm, 'mar'),
    fest_b=float(m2d.params['fest']), fest_se=float(m2d.bse['fest']), fest_p=float(m2d.pvalues['fest']), fest_ame=ame(m2d, Pm, 'fest'),
    tk_n=len(Pt), tk_b=float(mt.params['peak']), tk_se=float(mt.bse['peak']), tk_p=float(mt.pvalues['peak']), tk_pct=float(np.expm1(mt.params['peak']) * 100),
    tki_b=float(mti.params['peak:challenger']), tki_se=float(mti.bse['peak:challenger']), tki_p=float(mti.pvalues['peak:challenger']))
# sample size RQ2 inputs
lin = smf.ols('logit ~ C(entity) + t + peak', Pm).fit(); sig = float(np.sqrt(lin.scale))
Pm['pk_w'] = Pm.peak - Pm.groupby('entity').peak.transform('mean'); sdx = float(Pm.pk_w.std()); n2 = (K * sig / (0.20 * sdx)) ** 2
R['ss2'] = dict(sigma=sig, sdx=sdx, beta=0.2, n=n2, n_ceil=int(np.ceil(n2)), mdb=float(K * sig / (sdx * np.sqrt(len(Pm)))))
# calendar-month profile
mm = smf.ols('logit ~ C(entity) + t + C(moy)', Pm).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(Pm.entity)[0]})
prof = []
for mth in range(1, 13):
    if mth == 1: prof.append((mth, 0.0, 0.0, 0.0)); continue
    k = f'C(moy)[T.{mth}]'; prof.append((mth, float(mm.params[k]), float(mm.conf_int().loc[k, 0]), float(mm.conf_int().loc[k, 1])))
R['moy'] = prof

# ------------------------------------------------------------------ RQ3: persistence and survival
G3 = P.dropna(subset=['g'])
md = smf.mixedlm('g ~ 1', G3, groups=G3.entity).fit(); s2u = float(md.cov_re.iloc[0, 0]); s2e = float(md.scale); icc = s2u / (s2u + s2e)
iccs = []
ents = G3.entity.unique()
for _ in range(200):
    pick = rng.choice(ents, len(ents), replace=True); d = pd.concat([G3[G3.entity == e].assign(entity=f'{e}_{k}') for k, e in enumerate(pick)])
    try:
        mb = smf.mixedlm('g ~ 1', d, groups=d.entity).fit(); u = float(mb.cov_re.iloc[0, 0]); iccs.append(u / (u + float(mb.scale)))
    except Exception: pass
P['g_l'] = P.groupby('entity').g.shift(1); P['lgap'] = P.groupby('entity').gap.shift(1)
H = P.dropna(subset=['g', 'g_l']); H = H[(H.gap == 1) & (H.lgap == 1)]
ar = smf.ols('g ~ g_l + C(entity)', H).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(H.entity)[0]})
rho3 = float(ar.params['g_l']); hl = float(np.log(.5) / np.log(abs(rho3))); ci_r = [float(x) for x in ar.conf_int().loc['g_l']]
n3 = Z ** 2 * (1 - rho3 ** 2) / 0.10 ** 2
sp = []
for e, d in P.groupby('entity'):
    d = d.sort_values('month'); run = 0; start = None
    for _, r in d.iterrows():
        if pd.isna(r.g):
            if run > 0: sp.append((e, run, 0, start)); run = 0
            continue
        if r.g > 0:
            if run == 0: start = r
            run += 1
        else:
            if run > 0: sp.append((e, run, 1, start)); run = 0
    if run > 0: sp.append((e, run, 0, start))
SP = pd.DataFrame([dict(entity=e, dur=du, event=ev, tpap=int(a[a.entity == e].tpap.max()), ln_share=float(np.log(s.share_l if pd.notna(s.share_l) else s.share)), start_peak=int(s.peak)) for e, du, ev, s in sp])
SP.to_csv('outputs/tables/spells.csv', index=False)
kmf = KaplanMeierFitter().fit(SP.dur, SP.event)
d1_ = int(((SP.dur == 1) & (SP.event == 1)).sum()); n1_ = len(SP); d2_ = int(((SP.dur == 2) & (SP.event == 1)).sum()); n2_ = int((SP.dur >= 2).sum())
S1 = 1 - d1_ / n1_; S2_ = S1 * (1 - d2_ / n2_)
cph = CoxPHFitter().fit(SP[['dur', 'event', 'tpap', 'ln_share', 'start_peak']], 'dur', 'event')
cx = cph.summary
lr = logrank_test(SP[SP.tpap == 1].dur, SP[SP.tpap == 0].dur, SP[SP.tpap == 1].event, SP[SP.tpap == 0].event)
R['rq3'] = dict(n_g=len(G3), s2u=s2u, s2e=s2e, icc=icc, icc_ci=q(iccs), icc_boot=len(iccs), n_ar=len(H), rho=rho3, rho_ci=ci_r, rho_p=float(ar.pvalues['g_l']), hl=hl, n_need=n3, n_need_ceil=int(np.ceil(n3)),
    spells=n1_, events=int(SP.event.sum()), med=float(SP.dur.median()), mean=float(SP.dur.mean()), mx=int(SP.dur.max()), longest=sp[int(SP.dur.idxmax())][0], one_month=float((SP.dur == 1).mean()),
    d1=d1_, n1=n1_, d2=d2_, n2=n2_, S1=S1, S2=S2_, epv=30, logrank_p=float(lr.p_value),
    cox={k: dict(hr=float(np.exp(cx.loc[k, 'coef'])), lo=float(cx.loc[k, 'exp(coef) lower 95%']), hi=float(cx.loc[k, 'exp(coef) upper 95%']), p=float(cx.loc[k, 'p'])) for k in cx.index},
    tp_n=int((SP.tpap == 1).sum()), bk_n=int((SP.tpap == 0).sum()), tp_med=float(SP[SP.tpap == 1].dur.median()), bk_med=float(SP[SP.tpap == 0].dur.median()))

# ------------------------------------------------------------------ RQ4: rolling-origin forecasting
F = P.copy().sort_values(['entity', 'month'])
F['lnV_l'] = F.groupby('entity').lnV.shift(1); F['g_l'] = F.groupby('entity').g.shift(1); F['dlnV_l'] = F.groupby('entity').dlnV.shift(1)
F['logit_l'] = np.log((F.share_l / 100) / (1 - F.share_l / 100)); F['sin'] = np.sin(2 * np.pi * F.moy / 12); F['cos'] = np.cos(2 * np.pi * F.moy / 12)
F['tpapf'] = F.entity.map(lambda e: int(a[a.entity == e].tpap.max()))
F['hist_mean'] = F.groupby('entity').share.transform(lambda s: s.shift(1).expanding().mean())
for c in ['g_l', 'dlnV_l', 'lnV_l']: F.loc[F.gap != 1, c] = np.nan
F['y_dir'] = (F.g > 0).astype(float); F.loc[F.g.isna(), 'y_dir'] = np.nan
FE = ['share_l', 'g_l', 'lnV_l', 'dlnV_l', 'peak', 'sin', 'cos', 'tpapf']
D = F.dropna(subset=['share_l', 'share']).copy()
test_months = sorted(D[D.month >= '2025-01-01'].month.unique()); R['rq4_test_months'] = len(test_months)
preds = {k: [] for k in ['naive', 'histmean', 'logit', 'xgb']}; truth = []; meta = []
clf = {k: [] for k in ['naive', 'lr', 'xgb']}; ytrue = []; cmeta = []
GLMX = ['logit_l', 'peak', 'sin', 'cos', 'tpapf']
for m in test_months:
    tr = D[D.month < m]; te = D[D.month == m]
    truth += list(te.share); meta += list(zip(te.entity, te.month))
    preds['naive'] += list(te.share_l)
    preds['histmean'] += list(te.hist_mean.fillna(te.share_l))
    g1 = sm.GLM(tr.share / 100, sm.add_constant(tr[GLMX]), family=sm.families.Binomial()).fit()
    preds['logit'] += list(g1.predict(sm.add_constant(te[GLMX], has_constant='add')) * 100)
    xr = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xr.fit(tr[FE], tr.d_share.clip(-5, 5)); preds['xgb'] += list(te.share_l + xr.predict(te[FE]))
    # direction task
    trc = D[(D.month < m) & D.y_dir.notna()]; tec = D[(D.month == m) & D.y_dir.notna() & D.g_l.notna()]
    if len(tec):
        ytrue += list(tec.y_dir); cmeta += list(zip(tec.entity, tec.month)); clf['naive'] += list((tec.g_l > 0).astype(float))
        lr_ = LogisticRegression(max_iter=1000).fit(trc[GLMX].fillna(0), trc.y_dir); clf['lr'] += list(lr_.predict_proba(tec[GLMX].fillna(0))[:, 1])
        xc = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0, eval_metric='logloss')
        xc.fit(trc[FE], trc.y_dir); clf['xgb'] += list(xc.predict_proba(tec[FE])[:, 1])
truth = np.array(truth); pr = {k: np.array(v) for k, v in preds.items()}
met = {k: dict(rmse=float(np.sqrt(np.mean((truth - v) ** 2))), mae=float(np.mean(np.abs(truth - v)))) for k, v in pr.items()}
def dm(e1, e2):  # Diebold-Mariano on squared errors, h = 1, simple variance
    d = e1 ** 2 - e2 ** 2; tt = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))); return float(tt), float(2 * (1 - stats.norm.cdf(abs(tt))))
for k in ['histmean', 'logit', 'xgb']:
    t_, p_ = dm(truth - pr[k], truth - pr['naive']); met[k].update(dm_t=t_, dm_p=p_, skill=float(1 - met[k]['rmse'] / met['naive']['rmse']))
xr = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0).fit(D[FE], D.d_share.clip(-5, 5))
imp = dict(zip(FE, [float(x) for x in xr.feature_importances_]))
# errors without the largest 3 naive errors
en = np.abs(truth - pr['naive']); idx = np.argsort(-en)[:3]; keep = np.ones(len(truth), bool); keep[idx] = False
for k in pr: met[k]['rmse_trim'] = float(np.sqrt(np.mean((truth[keep] - pr[k][keep]) ** 2)))
R['rq4'] = dict(n=len(truth), months=len(test_months), met=met, imp=imp, n_need=Z ** 2 / (2 * 0.1 ** 2), n_need_ceil=int(np.ceil(Z ** 2 / (2 * .1 ** 2))))
yt = np.array(ytrue); cm = {}
for k, v in clf.items():
    v = np.array(v); pred = (v >= .5).astype(int) if k != 'naive' else v.astype(int)
    tp_ = int(((pred == 1) & (yt == 1)).sum()); fp_ = int(((pred == 1) & (yt == 0)).sum()); fn_ = int(((pred == 0) & (yt == 1)).sum())
    cm[k] = dict(tp=tp_, fp=fp_, fn=fn_, prec=tp_ / max(tp_ + fp_, 1), rec=tp_ / max(tp_ + fn_, 1), flags=int(pred.sum()), pr_auc=float(average_precision_score(yt, v)) if k != 'naive' else float('nan'))
R['rq4_cls'] = dict(n=len(yt), pos=int(yt.sum()), base=float(yt.mean()), met=cm)
# errors by test month
em = pd.DataFrame({'month': [m for _, m in meta], 'naive': (truth - pr['naive']) ** 2, 'logit': (truth - pr['logit']) ** 2, 'xgb': (truth - pr['xgb']) ** 2}).groupby('month').mean().pow(.5)
em.to_csv('outputs/tables/rmse_by_month.csv')

# ------------------------------------------------------------------ figures
def save(fig, name): fig.tight_layout(); fig.savefig(f'outputs/figures/{name}.png', dpi=220); plt.close(fig)
Sx_ = S.set_index('month').reindex(months_all)
# F1 volume and ticket size
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.0))
ax[0].plot(Sx_.index, Sx_.tot / 1000, color='black', marker='o', ms=2.5, lw=1.2); ax[0].set_ylabel('Volume (billion transactions)'); ax[0].set_xlabel('Month'); ax[0].set_title('Total app-reported volume', fontsize=9)
ax[1].plot(Sx_.index, Sx_.ticket, color='black', marker='o', ms=2.5, lw=1.2); ax[1].set_ylabel('Average ticket size (Rs)'); ax[1].set_xlabel('Month'); ax[1].set_title('Value per transaction', fontsize=9)
for x in ax: x.tick_params(axis='x', labelsize=8)
save(fig, 'eda_volume_ticket')
# F2 HHI + top2
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.0))
ax[0].plot(Sx_.index, Sx_.hhi, color='black', marker='o', ms=2.5, lw=1.2); ax[0].axvline(brk, color='gray', ls=':'); ax[0].set_ylabel('HHI'); ax[0].set_title('HHI of volume shares', fontsize=9)
ax[1].plot(Sx_.index, Sx_.top2, color='black', marker='o', ms=2.5, lw=1.2, label='Top two'); ax[1].plot(Sx_.index, Sx_.top3, color='gray', marker='s', ms=2.5, lw=1.2, ls='--', label='Top three'); ax[1].axvline(brk, color='gray', ls=':')
ax[1].set_ylabel('Share of volume (%)'); ax[1].set_title('Top-two and top-three shares', fontsize=9); ax[1].legend(frameon=False, fontsize=8)
for x in ax: x.set_xlabel('Month'); x.tick_params(axis='x', labelsize=8)
save(fig, 'eda_concentration')
# F3 leaders and Paytm (reuse style)
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for col, lab, ls, mk in [('pp', 'PhonePe', '-', 'o'), ('gp', 'Google Pay', '--', 's'), ('pt', 'Paytm', ':', '^')]: ax.plot(Sx_.index, Sx_[col], color='black', ls=ls, lw=1.3, marker=mk, ms=2.5, label=lab)
ax.axhline(30, color='gray', lw=1, ls='-.'); ax.text(months_all[1], 28.3, '30% cap threshold', fontsize=8, color='dimgray', va='top'); ax.axvline(brk, color='black', lw=.9, ls=':')
ax.set_ylabel('Share of UPI volume (%)'); ax.set_xlabel('Month'); ax.legend(frameon=False, fontsize=8, loc='center', bbox_to_anchor=(.62, .30), ncol=3)
save(fig, 'eda_shares')
# F4 cumulative share by rank
fig, ax = plt.subplots(figsize=(6.5, 3.0))
for mth, lab, ls in [('2022-01-01', 'Jan 2022', '-'), ('2024-05-01', 'May 2024 (HHI peak)', '--'), ('2026-09-01', 'Sep 2026', ':')]:
    d = named[named.month == mth].sort_values('share', ascending=False).share.cumsum().values[:15]; ax.plot(range(1, len(d) + 1), d, color='black', ls=ls, marker='o', ms=3, label=lab)
ax.set_xlabel('App rank by volume'); ax.set_ylabel('Cumulative share of volume (%)'); ax.set_ylim(60, 101); ax.legend(frameon=False, fontsize=8, loc='lower right')
save(fig, 'eda_cumulative')
# F5 calendar-month profile (RQ2)
fig, ax = plt.subplots(figsize=(6.5, 3.0)); mths = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for i, (mth, b, lo_, hi_) in enumerate(prof):
    col = '#555555' if mth in (3, 10, 11) else '#bbbbbb'
    ax.bar(i, b, color=col, width=.7); ax.errorbar(i, b, yerr=[[b - lo_], [hi_ - b]], color='black', lw=.8, capsize=2)
ax.axhline(0, color='black', lw=.8); ax.set_xticks(range(12)); ax.set_xticklabels(mths); ax.set_ylabel('Logit share relative to January'); ax.set_xlabel('Calendar month')
save(fig, 'eda_calendar')
# F6 ticket size by app
fig, ax = plt.subplots(figsize=(6.5, 3.2)); order = P.groupby('entity').ticket.median().sort_values().index.tolist()
ax.boxplot([P[P.entity == e].ticket.dropna() for e in order], vert=False, labels=order, flierprops=dict(marker='o', ms=2), medianprops=dict(color='black'))
ax.set_xlabel('Ticket size (Rs per transaction)'); ax.tick_params(axis='y', labelsize=8)
save(fig, 'eda_ticket')
# F7 distribution of g
fig, ax = plt.subplots(figsize=(6.5, 3.0)); ax.hist(P.g.dropna().clip(-1, 1), bins=50, color='#777777', edgecolor='white'); ax.axvline(lo, color='black', ls='--', lw=1); ax.axvline(hi, color='black', ls='--', lw=1)
ax.set_xlabel('App-specific growth g (log points, clipped at ±1)'); ax.set_ylabel('App-months'); save(fig, 'eda_growth')
# F8 ITS counterfactual
Sx2 = S.copy(); Xf = sm.add_constant(Sx2[['t', 'post', 'tp', 'paytm']]); fit = mC.predict(Xf)
cf = mC.predict(sm.add_constant(Sx2.assign(post=0, tp=0)[['t', 'post', 'tp', 'paytm']]))
fig, ax = plt.subplots(figsize=(6.5, 3.2)); ax.plot(S.month, S.hhi, color='black', marker='o', ms=2.5, lw=1.2, label='Observed')
ax.plot(S.month[S.post == 1], cf[S.post == 1], color='gray', ls='--', lw=1.5, label='Counterfactual (pre-period trend)'); ax.plot(S.month, fit, color='#999999', lw=1, ls=':', label='Fitted model')
ax.axvline(brk, color='black', ls=':', lw=.9); ax.set_ylabel('HHI'); ax.set_xlabel('Month'); ax.legend(frameon=False, fontsize=8, loc='upper left'); save(fig, 'rq1_its')
# F9 placebo breaks
fig, ax = plt.subplots(figsize=(6.5, 3.0)); ax.plot(pl.month, pl.b, color='black', marker='o', ms=3, lw=1); ax.axhline(0, color='gray', lw=.8)
ax.scatter([brk], [pl[pl.month == brk].b.iloc[0]], s=70, facecolors='none', edgecolors='black', zorder=5); ax.set_ylabel('Estimated level change (HHI points)'); ax.set_xlabel('Assumed break month'); save(fig, 'rq1_placebo')
# F10 KM by type
fig, ax = plt.subplots(figsize=(6.5, 3.0))
for flag, lab, ls in [(1, 'TPAP apps', '-'), (0, 'Bank and NPCI apps', '--')]:
    d = SP[SP.tpap == flag]; k_ = KaplanMeierFitter().fit(d.dur, d.event, label=lab); k_.plot_survival_function(ax=ax, ci_show=False, color='black', ls=ls)
ax.set_xlabel('Months of above-system growth'); ax.set_ylabel('Share of runs still ongoing'); ax.set_xlim(0, 12); ax.legend(frameon=False, fontsize=8); save(fig, 'rq3_km')
# F11 model comparison
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.0)); names = ['Naive', 'History\nmean', 'Fractional\nlogit', 'XGBoost']; ks = ['naive', 'histmean', 'logit', 'xgb']
ax[0].bar(names, [met[k]['rmse'] for k in ks], color=['#444444', '#999999', '#999999', '#999999']); ax[0].set_ylabel('RMSE (pp)'); ax[0].tick_params(axis='x', labelsize=8)
ax[1].bar(names, [met[k]['mae'] for k in ks], color=['#444444', '#999999', '#999999', '#999999']); ax[1].set_ylabel('MAE (pp)'); ax[1].tick_params(axis='x', labelsize=8)
save(fig, 'rq4_models')

json.dump(R, open('outputs/interim_results.json', 'w'), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
print('done'); print(json.dumps({k: R[k] for k in ['specs']}, default=float)[:1500])

"""Monthly cap-monitoring brief from the processed panel (proof-of-concept deployment).

Usage: python code/monitor.py [YYYY-MM]    default = latest month in the panel
Prints concentration indicators, rolling three-month shares against the 30% threshold (calendar window, available months only),
and a next-month share forecast for the leading apps from the fractional-logit model fitted on all earlier months.
"""
import sys, numpy as np, pandas as pd, statsmodels.api as sm

a = pd.read_csv('data/processed/upi_app_panel.csv', parse_dates=['month'])
a['tot'] = a.groupby('month').vol_mn.transform('sum'); a['share'] = a.vol_mn / a.tot * 100   # total includes 'Others'
a = a[~a.is_others].copy()
last = pd.Timestamp(sys.argv[1] + '-01') if len(sys.argv) > 1 else a.month.max()
cur = a[a.month == last].sort_values('vol_mn', ascending=False)
print(f'Month: {last:%Y-%m}   named apps: {len(cur)}   total volume (Mn): {cur.tot.iloc[0]:,.0f}')
print(f'HHI: {(cur.share ** 2).sum():,.0f}   top-two share: {cur.share.iloc[:2].sum():.1f}%')
w = a[(a.month > last - pd.DateOffset(months=3)) & (a.month <= last)]
tot_w = a[(a.month > last - pd.DateOffset(months=3)) & (a.month <= last)].drop_duplicates('month').tot.sum()
n_m = w.month.nunique()
print(f'\nRolling window: {n_m} available month(s) of the last 3 calendar months')
for e in cur.entity.iloc[:4]:
    s = w[w.entity == e].vol_mn.sum() / tot_w * 100
    print(f'  {e:<12} rolling share {s:5.1f}%   {"ABOVE 30%" if s > 30 else "below 30%"}')
# forecast: fractional logit on lagged logit share, fitted on months before `last`
a = a.sort_values(['entity', 'month']); a['gap'] = a.groupby('entity').month.diff().dt.days
a['share_l'] = a.groupby('entity').share.shift(1); a.loc[~a.gap.between(28, 31), 'share_l'] = np.nan
a['ll'] = np.log(a.share_l / 100 / (1 - a.share_l / 100)); a['peak'] = a.month.dt.month.isin([3, 10, 11]).astype(int)
a['sin'] = np.sin(2 * np.pi * a.month.dt.month / 12); a['cos'] = np.cos(2 * np.pi * a.month.dt.month / 12)
tr = a[(a.month <= last) & a.ll.notna()]; X = ['ll', 'peak', 'sin', 'cos']
g = sm.GLM(tr.share / 100, sm.add_constant(tr[X]), family=sm.families.Binomial()).fit()
nxt = last + pd.DateOffset(months=1)
fut = cur[['entity', 'share']].rename(columns={'share': 'share_l'}); fut['ll'] = np.log(fut.share_l / 100 / (1 - fut.share_l / 100))
fut['peak'] = int(nxt.month in (3, 10, 11)); fut['sin'] = np.sin(2 * np.pi * nxt.month / 12); fut['cos'] = np.cos(2 * np.pi * nxt.month / 12)
fut['forecast'] = g.predict(sm.add_constant(fut[X], has_constant='add')) * 100
print(f'\nForecast shares for {nxt:%Y-%m} (leading apps, percentage points; indicative only):')
print(fut[['entity', 'share_l', 'forecast']].head(5).round(2).to_string(index=False))

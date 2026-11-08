"""Combine NPCI monthly 'UPI Applications' tables into one clean month x app panel.

Usage:  python code/combine_apps.py --raw data/raw --out data/processed
Input : monthly files whose names contain YYYY-Mon (e.g. ...applications-2022-Jan.xlsx)
        or upi_apps_YYYY-MM.xlsx. Files with a lost .xlsx extension are fine.
Output: data/processed/upi_app_panel.csv, data/processed/file_audit.csv
"""
import argparse, glob, os, re, shutil, tempfile
import numpy as np, pandas as pd, openpyxl

MON = {m: i + 1 for i, m in enumerate('Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split())}


def num(x):
    """Parse numbers stored as text ('7,45,552.20', '-', '########')."""
    if x is None: return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip().replace(',', '')
    if s in ('', '-', '–'): return 0.0
    try: return float(s)
    except ValueError: return np.nan


def file_month(path):
    b = os.path.basename(path)
    m = re.search(r'(\d{4})-(\w{3})\b', b)
    if m and m.group(2).title() in MON: return pd.Timestamp(int(m.group(1)), MON[m.group(2).title()], 1)
    m = re.search(r'(\d{4})-(\d{2})', b)
    return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)


def read_file(path):
    tmp = os.path.join(tempfile.mkdtemp(), 't.xlsx'); shutil.copy(path, tmp)
    ws = openpyxl.load_workbook(tmp, data_only=True).active
    title = str(ws.cell(1, 1).value)
    tm = re.search(r"\((\w+)'(\d\d)\)", title)
    title_month = pd.Timestamp(2000 + int(tm.group(2)), MON[tm.group(1)[:3].title()], 1)
    hdr = {}  # header-aware: locate column groups in row 2 (layout differs across years)
    for c in range(1, 16):
        v = ws.cell(2, c).value
        if v: hdr[str(v).split()[0].lower()] = c - 1
    cols = {k: hdr.get(k) for k in ['customer', 'b2c', 'b2b', 'on-us', 'total']}
    rows = []
    for r in ws.iter_rows(min_row=4, values_only=True):
        r = list(r) + [None] * 16
        if r[1] is None or np.isnan(num(r[0])): continue
        g = lambda k, o: num(r[cols[k] + o]) if cols[k] is not None else 0.0
        rows.append(dict(app_raw=str(r[1]).strip(), ci_vol=g('customer', 0), ci_val=g('customer', 1),
                         b2c_vol=g('b2c', 0), b2b_vol=g('b2b', 0), onus_vol=g('on-us', 0),
                         vol=g('total', 0), val=g('total', 1)))
    return title_month, pd.DataFrame(rows)


def key(a): return re.sub(r'[^a-z0-9]+', '', a.lower().replace('#', ''))


ALIAS = {'phonepe': 'PhonePe', 'googlepay': 'Google Pay', 'cred': 'CRED', 'amazonpay': 'Amazon Pay', 'otherapps': 'Others',
         'others': 'Others', 'whatsapp': 'WhatsApp', 'mobikwik': 'MobiKwik', 'slice': 'Slice', 'sliceapp': 'Slice',
         'supermoney': 'super.money', 'navi': 'Navi', 'bharatpe': 'BharatPe', 'bhim': 'BHIM', 'bhimapp': 'BHIM',
         'cheq': 'CheQ', 'jar': 'Jar', 'obopay': 'Obopay', 'groww': 'Groww', 'kotakmahindrabankapps': 'Kotak Bank Apps',
         'kotakbankapps': 'Kotak Bank Apps', 'sbibankapps': 'State Bank of India Apps', 'idfcfirstbankapps': 'IDFC Bank Apps'}
KNOWN_TPAP = {'PhonePe', 'Google Pay', 'Paytm', 'CRED', 'Amazon Pay', 'WhatsApp', 'MobiKwik', 'Navi', 'super.money', 'FamApp',
              'BharatPe', 'Groww', 'Bajaj Finserv', 'Slice', 'Jar', 'CheQ'}


def entity(app):
    k = key(app)
    if 'paytm' in k: return 'Paytm'          # Paytm Payments Bank App / Paytm (OCL) -> one entity
    if k.startswith('fam'): return 'FamApp'  # FamApp / Fampay variants
    if k.startswith('bajajfinserv'): return 'Bajaj Finserv'
    return ALIAS.get(k, app.replace('#', '').strip())


def main(raw, out):
    files = sorted(f for f in glob.glob(os.path.join(raw, '**', '*'), recursive=True) if os.path.isfile(f) and re.search(r'\d{4}-', f))
    data, audit = {}, []
    for f in files:
        fm = file_month(f); tm, df = read_file(f); data[f] = (fm, tm, df)
    # 1) exact-duplicate content under a wrong month -> keep the file whose title matches its name
    sig = {f: tuple(zip(d[2].app_raw, d[2].vol)) for f, d in data.items()}
    drop = set()
    for f, s in sig.items():
        for g, t in sig.items():
            if f < g and s == t:
                bad = f if data[f][1] != data[f][0] else g
                drop.add(bad)
    frames = []
    for f, (fm, tm, df) in data.items():
        status = 'excluded: identical content to another month' if f in drop else ('used (title month differs: relabeled by file name)' if tm != fm else 'used')
        audit.append(dict(file=os.path.basename(f), file_month=fm.date(), title_month=tm.date(), rows=len(df), status=status))
        if f in drop: continue
        df = df.assign(month=fm)
        frames.append(df)
    raw_all = pd.concat(frames, ignore_index=True)
    # 2) stray repeated 'Others' row: drop the second one if it equals the previous month's Others row
    months = sorted(raw_all.month.unique()); prev_others = {}
    keep = []
    for m in months:
        d = raw_all[raw_all.month == m]
        oth = d[d.app_raw.map(key).isin(['others', 'otherapps'])]
        if len(oth) > 1 and m - pd.DateOffset(months=1) in prev_others:
            p = prev_others[m - pd.DateOffset(months=1)]
            stray = oth[(oth.vol == p[0]) & (oth.val == p[1])].index
            d = d.drop(stray)
        o = d[d.app_raw.map(key).isin(['others', 'otherapps'])]
        if len(o): prev_others[m] = (o.vol.iloc[0], o.val.iloc[0])
        keep.append(d)
    df = pd.concat(keep, ignore_index=True)
    # 3) '########' value cells: use customer-initiated value when no other transaction type exists
    m = df.val.isna() & (df.b2c_vol + df.b2b_vol + df.onus_vol == 0) & df.ci_val.notna()
    df.loc[m, 'val'] = df.loc[m, 'ci_val']
    df['entity'] = df.app_raw.map(entity)
    tpap_marked = set(df[df.app_raw.str.contains('#')].entity) | KNOWN_TPAP
    panel = df.groupby(['month', 'entity']).agg(vol_mn=('vol', 'sum'), val_cr=('val', lambda s: s.sum(min_count=1)), n_rows=('app_raw', 'size')).reset_index()
    panel['tpap'] = panel.entity.isin(tpap_marked) & ~panel.entity.str.contains('Bank|BHIM|Others')
    panel['is_others'] = panel.entity.eq('Others')
    panel['share'] = panel.vol_mn / panel.groupby('month').vol_mn.transform('sum') * 100
    os.makedirs(out, exist_ok=True)
    panel.to_csv(os.path.join(out, 'upi_app_panel.csv'), index=False)
    pd.DataFrame(audit).sort_values('file_month').to_csv(os.path.join(out, 'file_audit.csv'), index=False)
    print(f'{panel.month.nunique()} usable months, {len(panel)} entity-months, {panel.entity.nunique()} entities')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--raw', default='data/raw'); ap.add_argument('--out', default='data/processed')
    a = ap.parse_args(); main(a.raw, a.out)

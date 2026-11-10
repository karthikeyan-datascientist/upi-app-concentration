"""Count cell-level data-quality issues in the raw NPCI monthly files (used for the data cleaning log).

Usage: python code/data_audit.py --raw data/raw --out outputs/data_audit.json
"""
import argparse, glob, json, os, re, shutil, tempfile
import numpy as np, openpyxl


def main(raw, out):
    files = sorted(f for f in glob.glob(os.path.join(raw, '**', '*'), recursive=True) if os.path.isfile(f) and re.search(r'\d{4}-', f))
    c = dict(files=len(files), data_rows=0, numeric_cells=0, text_numeric_cells=0, grouped_text_cells=0, dash_cells=0, hash_cells=0,
             blank_cells=0, zero_total_rows=0, header_repeat_rows=0)
    for f in files:
        tmp = os.path.join(tempfile.mkdtemp(), 't.xlsx'); shutil.copy(f, tmp)
        ws = openpyxl.load_workbook(tmp, data_only=True).active
        for r in ws.iter_rows(min_row=4, values_only=True):
            r = list(r) + [None] * 14
            if r[1] is None: continue
            if isinstance(r[0], str) and not re.match(r'^\d+(\.0)?$', r[0].strip()):
                c['header_repeat_rows'] += 1; continue
            c['data_rows'] += 1
            vals = r[2:12]
            for v in vals:
                if v is None: c['blank_cells'] += 1
                elif isinstance(v, (int, float)): c['numeric_cells'] += 1
                else:
                    s = str(v).strip()
                    if s in ('-', '–'): c['dash_cells'] += 1
                    elif '#' in s: c['hash_cells'] += 1
                    elif ',' in s: c['grouped_text_cells'] += 1
                    else: c['text_numeric_cells'] += 1
            try:
                if float(str(vals[8]).replace(',', '')) == 0: c['zero_total_rows'] += 1
            except ValueError: pass
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    json.dump(c, open(out, 'w'), indent=1); print(c)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--raw', default='data/raw'); ap.add_argument('--out', default='outputs/data_audit.json')
    a = ap.parse_args(); main(a.raw, a.out)

"""Load the processed panel into a local SQLite database for reproducible querying.

Usage: python code/build_db.py     (needs data/processed/upi_app_panel.csv from combine_apps.py)
Creates data/processed/upi_apps.db with tables apps_monthly and system_monthly and an index on (month, entity).
"""
import sqlite3, pandas as pd

a = pd.read_csv('data/processed/upi_app_panel.csv', parse_dates=['month'])
a['month'] = a.month.dt.strftime('%Y-%m')
a['tpap'] = a.tpap.astype(int); a['is_others'] = a.is_others.astype(int)
con = sqlite3.connect('data/processed/upi_apps.db')
a.to_sql('apps_monthly', con, if_exists='replace', index=False)
con.execute('CREATE INDEX IF NOT EXISTS ix_month_entity ON apps_monthly (month, entity)')
con.execute('DROP TABLE IF EXISTS system_monthly')
con.execute("""CREATE TABLE system_monthly AS
  SELECT month, SUM(vol_mn) AS vol_mn, SUM(val_cr) AS val_cr, COUNT(*) AS n_entities,
         SUM(CASE WHEN is_others = 0 THEN share * share ELSE 0 END) AS hhi
  FROM apps_monthly GROUP BY month""")
con.commit()
for q in ["SELECT COUNT(*) AS rows_, COUNT(DISTINCT month) AS months, COUNT(DISTINCT entity) AS entities FROM apps_monthly",
          "SELECT month, ROUND(hhi,0) AS hhi FROM system_monthly ORDER BY month DESC LIMIT 3",
          "SELECT entity, ROUND(vol_mn,1) AS vol_mn, ROUND(share,2) AS share FROM apps_monthly WHERE month='2026-09' AND is_others=0 ORDER BY vol_mn DESC LIMIT 5"]:
    print(q); print(pd.read_sql(q, con).to_string(index=False)); print()
con.close()

"""Preliminary analysis used to size the study (sample size inputs, HHI series, RQ1-RQ4 preliminary values).

Usage: python code/prelim_analysis.py      (run code/combine_apps.py first)
Reads  data/processed/upi_app_panel.csv ; writes outputs/results.json and outputs/figures/fig1.png, fig2.png
All values here are preliminary and only used to size the study; they are not final estimates.
"""
import pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf, json, warnings, os
from lifelines import KaplanMeierFitter
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
os.makedirs('outputs/figures', exist_ok=True)
R = {}
a = pd.read_csv('data/processed/upi_app_panel.csv', parse_dates=['month']).rename(columns={'vol_mn': 'vol', 'val_cr': 'val'})
R['months'] = int(a.month.nunique()); R['entities'] = int(a.entity.nunique()); R['entity_months'] = len(a)
a['tot']=a.groupby('month').vol.transform('sum'); a['share']=a.vol/a.tot*100
named=a[~a.is_others]
# system series, HHI = sum over named apps of squared share (share of total incl Others)
S=named.groupby('month').apply(lambda g: pd.Series(dict(hhi=(g.share**2).sum(),top2=g.sort_values('vol',ascending=False).share.iloc[:2].sum(),
   top3=g.sort_values('vol',ascending=False).share.iloc[:3].sum(),tpap=g[g.tpap].share.sum(),pp=g[g.entity=='PhonePe'].share.sum(),gp=g[g.entity=='Google Pay'].share.sum(),
   pt=g[g.entity=='Paytm'].share.sum(),tot=g.tot.iloc[0],napps=len(g)))).reset_index()
S['t']=(S.month.dt.year-2022)*12+S.month.dt.month-1
brk=pd.Timestamp('2025-01-01'); t0=int(S[S.month==brk].t.iloc[0])
S['post']=(S.month>=brk).astype(int); S['tp']=S.post*(S.t-t0); S['paytm']=(S.month>='2024-03-01').astype(int)
R['hhi_first']=float(S.hhi.iloc[0]); R['hhi_last']=float(S.hhi.iloc[-1]); R['hhi_max']=float(S.hhi.max()); R['hhi_max_month']=str(S.loc[S.hhi.idxmax(),'month'].date())
R['hhi_min']=float(S.hhi.min()); R['hhi_dec24']=float(S[S.month=='2024-12-01'].hhi.iloc[0]); R['hhi_jan25']=float(S[S.month==brk].hhi.iloc[0])
for k in ['pp','gp','pt','top2','top3','tpap']: R[k+'_first']=float(S[k].iloc[0]); R[k+'_last']=float(S[k].iloc[-1])
R['pp_max']=float(S.pp.max()); R['pp_max_m']=str(S.loc[S.pp.idxmax(),'month'].date()); R['pt_min']=float(S.pt.min())
R['pre_n']=int((S.post==0).sum()); R['post_n']=int((S.post==1).sum())
def fit(cols,y='hhi',lag=2):
    X=sm.add_constant(S[cols]); m=sm.OLS(S[y],X).fit(); h=m.get_robustcov_results('HAC',maxlags=lag)
    r=m.resid; rho=np.corrcoef(r.values[:-1],r.values[1:])[0,1]
    ci=h.conf_int()[list(X.columns).index('post')]
    return dict(b=float(m.params['post']),se=float(h.bse[list(X.columns).index('post')]),p=float(h.pvalues[list(X.columns).index('post')]),ci=[float(ci[0]),float(ci[1])],
        sd=float(np.sqrt(m.scale)),rho=float(rho),r2=float(m.rsquared),params={k:float(v) for k,v in m.params.items()},
        bslope=float(m.params['tp']) if 'tp' in m.params else None, pslope=float(h.pvalues[list(X.columns).index('tp')]) if 'tp' in X.columns else None)
R['its_seg']=fit(['t','post','tp']); R['its_seg_paytm']=fit(['t','post','tp','paytm'])
R['its_lvl']=fit(['t','post'])
R['its_seg_lag4']=fit(['t','post','tp'],lag=4)
# sample size RQ1
K=1.959964+0.841621; z=1.959964; K=z+0.841621
sd=R['its_seg']['sd']; rho=R['its_seg']['rho']; vif=(1+rho)/(1-rho); h=100
n1=4*vif*(z*sd/h)**2
se_ach=sd*np.sqrt(vif*(1/R['pre_n']+1/R['post_n']))
R['ss1']=dict(K=K,sd=sd,rho=rho,vif=vif,h=h,n=n1,n_ceil=int(np.ceil(n1)),se_ach=se_ach,hw_ach=z*se_ach,mde=K*se_ach,mean_pre=float(S[S.month<brk].hhi.mean()))
# panel set
st=named.groupby('entity').agg(mo=('month','nunique'),ms=('share','mean')).query('mo>=24').sort_values('ms',ascending=False)
top=list(st.index[:15]); R['top15']=top; R['top15_months']={e:int(st.loc[e,'mo']) for e in top}; R['top15_ms']={e:float(st.loc[e,'ms']) for e in top}
P=a[a.entity.isin(top)].copy().merge(S[['month','t']],on='month')
P['logit']=np.log((P.share/100)/(1-P.share/100)); P['lnV']=np.log(P.vol); P['peak']=P.month.dt.month.isin([3,10,11]).astype(int)
P['ticket']=P.val*1e7/(P.vol*1e6)
R['panel_n']=len(P); R['peak_n']=int(P.peak.sum()); R['top15_share_cov']=float(P.groupby('month').share.sum().mean())
f=smf.ols('logit~C(entity)+t+peak',data=P).fit(cov_type='cluster',cov_kwds={'groups':P.entity.astype('category').cat.codes})
Pd=P.copy(); Pd['pk_w']=Pd.peak-Pd.groupby('entity').peak.transform('mean')
sigma=float(np.sqrt(f.scale)); sdx=float(Pd.pk_w.std()); beta=0.20; n2=(K*sigma/(beta*sdx))**2
R['ss2']=dict(sigma=sigma,sdx=sdx,beta=beta,n=n2,n_ceil=int(np.ceil(n2)),mdb=K*sigma/(sdx*np.sqrt(len(P))),prelim_b=float(f.params['peak']),prelim_se=float(f.bse['peak']))
# growth deviation
P=P.sort_values(['entity','month']); P['prev_m']=P.groupby('entity').month.shift(1)
P['gap']=(P.month.dt.year-P.prev_m.dt.year)*12+P.month.dt.month-P.prev_m.dt.month
P['dlnV']=P.groupby('entity').lnV.diff(); P.loc[P.gap!=1,'dlnV']=np.nan
sysg=a.groupby('month').vol.sum().apply(np.log).reset_index(); sysg.columns=['month','lnT']; sysg['dT']=sysg.lnT.diff()
sysg['gp']=(sysg.month.dt.year-sysg.month.shift(1).dt.year)*12+sysg.month.dt.month-sysg.month.shift(1).dt.month; sysg.loc[sysg.gp!=1,'dT']=np.nan
P=P.merge(sysg[['month','dT']],on='month'); P['g']=P.dlnV-P.dT
G=P.dropna(subset=['g']); md=smf.mixedlm('g~1',G,groups=G.entity).fit(); s2u=float(md.cov_re.iloc[0,0]); s2e=float(md.scale)
P['g_l']=P.groupby('entity').g.shift(1); P['lgap']=P.groupby('entity').gap.shift(1)
H=P.dropna(subset=['g','g_l']); H=H[(H.gap==1)&(H.lgap==1)]
ar=smf.ols('g~g_l+C(entity)',data=H).fit(); rho3=float(ar.params['g_l']); hl=float(np.log(0.5)/np.log(abs(rho3)))
n3=z**2*(1-rho3**2)/0.10**2
R['rq3']=dict(growth_n=len(G),s2u=s2u,s2e=s2e,icc=s2u/(s2u+s2e),rho=rho3,hl=hl,n_ar=len(H),n_need=n3,n_need_ceil=int(np.ceil(n3)))
sp=[]
for e,gdf in P.groupby('entity'):
    gdf=gdf.sort_values('month'); run=0
    for _,r in gdf.iterrows():
        if pd.isna(r.g):
            if run>0: sp.append((e,run,0)); run=0
            continue
        if r.g>0: run+=1
        else:
            if run>0: sp.append((e,run,1)); run=0
    if run>0: sp.append((e,run,0))
sp=pd.DataFrame(sp,columns=['e','dur','event'])
d1=int(((sp.dur==1)&(sp.event==1)).sum()); n1_=len(sp); d2=int(((sp.dur==2)&(sp.event==1)).sum()); n2_=int((sp.dur>=2).sum())
S1=1-d1/n1_; S2=S1*(1-d2/n2_)
R['spell']=dict(n=n1_,events=int(sp.event.sum()),med=float(sp.dur.median()),mx=int(sp.dur.max()),d1=d1,n1=n1_,d2=d2,n2=n2_,S1=S1,S2=S2,epv=3*10)
R['spell']['longest_app']=sp.loc[sp.dur.idxmax(),'e']
# RQ4 naive
P['share_l']=P.groupby('entity').share.shift(1)
T=P[(P.gap==1)&(P.month>='2025-01-01')].dropna(subset=['share_l'])
rm=float(np.sqrt(((T.share-T.share_l)**2).mean())); ma=float((T.share-T.share_l).abs().mean())
n4=z**2/(2*0.10**2)
R['rq4']=dict(rmse=rm,mae=ma,n=len(T),months=int(T.month.nunique()),n_need=n4,n_need_ceil=int(np.ceil(n4)))
T2=T.dropna(subset=['g','g_l']); y=(T2.g>0).astype(int); pred=(T2.g_l>0).astype(int)
tp_=int(((y==1)&(pred==1)).sum()); fp=int(((y==0)&(pred==1)).sum()); fn=int(((y==1)&(pred==0)).sum())
R['rq4'].update(tp=tp_,fp=fp,fn=fn,prec=tp_/(tp_+fp),rec=tp_/(tp_+fn),ncls=len(T2),pos=int(y.sum()))
R['final_N']=max(R['ss2']['n_ceil'],R['rq3']['n_need_ceil'],R['rq4']['n_need_ceil'])
R['missing_months']=['2022-07','2022-11','2026-03','2026-05','2026-06','2026-07']
# panel unbalance
R['entities_all_months']=int((a.groupby('entity').month.nunique()==51).sum())
R['tpap_apps_top15']=sum(1 for e in top if a[a.entity==e].tpap.max())
json.dump(R,open('outputs/results.json','w'),indent=1,default=float)
S.to_csv('outputs/system_series.csv',index=False)
# ---- figures
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','Times New Roman','DejaVu Serif'],'font.size':10})
allm=pd.date_range('2022-01-01','2026-09-01',freq='MS'); have=set(S.month)
fig,ax=plt.subplots(figsize=(6.5,3.4))
Sx=S.set_index('month').reindex(allm)
ax.plot(Sx.index,Sx.hhi,color='black',lw=1.4,marker='o',ms=3,label='Monthly HHI (volume)')
# fitted segmented
fit_=sm.OLS(S.hhi,sm.add_constant(S[['t','post','tp']])).fit(); S['fit']=fit_.fittedvalues
for flag in [0,1]:
    d=S[S.post==flag]; ax.plot(d.month,d.fit,color='gray',lw=1.6,ls='--',label='Segmented fit' if flag==0 else None)
ax.axvline(brk,color='black',lw=0.9,ls=':'); ax.text(brk,Sx.hhi.min()-40,' Jan 2025\n cap deadline\n extension',fontsize=8,va='bottom',ha='left')
for m in allm:
    if m not in have: ax.axvline(m,color='lightgray',lw=3,alpha=0.6,zorder=0)
ax.set_ylabel('HHI (0–10,000)'); ax.set_xlabel('Month'); ax.legend(frameon=False,fontsize=8,loc='upper left'); ax.spines[['top','right']].set_visible(False)
ax.set_ylim(Sx.hhi.min()-120,Sx.hhi.max()+120)
plt.tight_layout(); plt.savefig('outputs/figures/fig1.png',dpi=220); plt.close()
fig,ax=plt.subplots(figsize=(6.5,3.4))
for col,lab,ls,mk in [('pp','PhonePe','-','o'),('gp','Google Pay','--','s'),('pt','Paytm',':','^')]:
    ax.plot(Sx.index,Sx[col],color='black',ls=ls,lw=1.3,marker=mk,ms=2.5,label=lab)
ax.axhline(30,color='gray',lw=1,ls='-.'); ax.text(allm[1],28.3,'30% cap threshold',fontsize=8,color='dimgray',va='top')
ax.axvline(brk,color='black',lw=0.9,ls=':')
ax.set_ylabel('Share of UPI volume (%)'); ax.set_xlabel('Month'); ax.legend(frameon=False,fontsize=8,loc='center',bbox_to_anchor=(0.62,0.30),ncol=3); ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('outputs/figures/fig2.png',dpi=220); plt.close()
print(json.dumps({k:R[k] for k in ['months','entities','entity_months','hhi_first','hhi_last','hhi_max','hhi_max_month','pre_n','post_n']},indent=0))
for k in ['its_seg','its_seg_paytm','its_lvl','its_seg_lag4','ss1','ss2','rq3','spell','rq4']: print(k,json.dumps(R[k],default=float))
print(R['final_N'],R['panel_n'],R['peak_n'],R['entities_all_months'],R['tpap_apps_top15'])

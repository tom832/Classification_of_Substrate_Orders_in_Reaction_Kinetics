#!/usr/bin/env python3
"""Secondary check: fit integrated reduced rate laws to the SAME original points.

Fixed primary background/window from kinetic_validation.py. Unweighted
concentration residuals; descriptive working-iid AICc, accompanied by lag-1
residual correlation. Does not replace the rate-space Table S13 comparison.

Methods references (also cited in SI Section 10):
Local-polynomial differentiation: Savitzky and Golay, Anal. Chem. 1964,
36, 1627-1639. DOI: 10.1021/ac60214a047.
Small-sample AIC correction: Hurvich and Tsai, Biometrika 1989,
76, 297-307. DOI: 10.1093/biomet/76.2.297.
Kinetic model discrimination: Taylor et al., React. Chem. Eng. 2021,
6, 1404-1411. DOI: 10.1039/D1RE00098E."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import wrightomega
from kinetic_validation import DEFAULT_SOURCE, MODELS, SUBSTRATES, COLORS, extract_rates


def concentration(model,p,t):
    x0=p[0]
    if model=='S1': return np.maximum(x0-p[1]*t,0)
    if model=='S3': return x0*np.exp(-p[1]*t)
    v,k=p[1:3]
    if model=='S2' or (model=='S5' and p[3]==0):
        return k*wrightomega(x0/k-v*t/k+np.log(x0/k))
    if model=='S4':
        return k/wrightomega(k/x0+np.log(k/x0)+v*t)
    side=p[3]
    if v==0:return x0*np.exp(-side*t)
    # Solve the integrated S5 law in log-concentration with vector bisection.
    A=v+side*k
    B=side*x0/A
    # g(u)=time for x=x0*exp(-u); stable log1p for small side rate.
    def elapsed(u):
        difference=np.log1p(B*(-np.expm1(-u))/(1+B*np.exp(-u)))
        return (k/A)*u+(v/A)*(difference/side)
    lo=np.zeros_like(t);hi=np.minimum(t*(v/k+side),740)+1e-12
    for _ in range(48):
        mid=(lo+hi)/2
        later=elapsed(mid)>t
        hi=np.where(later,mid,hi);lo=np.where(later,lo,mid)
    return x0*np.exp(-(lo+hi)/2)


def fit_concentration(model,t,y,previous):
    def encode(p):
        return np.array([p[0],np.log(max(p[1],1e-12))]+(
            [np.log(p[2])] if model in ('S2','S4','S5') else [])+([p[3]] if model=='S5' else []))
    def decode(q):
        return np.array([q[0],np.exp(q[1])]+(
            [np.exp(q[2])] if model in ('S2','S4','S5') else [])+([q[3]] if model=='S5' else []))
    lo=[.25,-14];hi=[2.,14]
    starts=[];candidates=[]
    if model in ('S1','S3'):
        starts=[[1.,rate] for rate in (.5,1,3,10)]
    else:
        lo+=[np.log(1e-4)];hi+=[np.log(1e4)]
        starts=[[1.,(1+k)*2,k] for k in (.001,.03,.3,3,100,10000)]
    if model=='S5':
        lo+=[0];hi+=[100]
        starts=[p+[b] for p in starts for b in (0.,1.,5.)]
        s2=previous['S2'];embedded=np.r_[s2,0.]
        candidates.append((np.sum((concentration('S5',embedded,t)-y)**2),embedded,True))
        starts += [list(s2)+[b] for b in (0.,.1,1.)]
        # Pure first-order limit V=0 is also feasible in S5.
        p=np.array([previous['S3'][0],0.,1.,previous['S3'][1]])
        candidates.append((np.sum((concentration('S5',p,t)-y)**2),p,True))
    for start in starts:
        q0=np.clip(encode(start),lo,hi)
        opt=least_squares(lambda q:concentration(model,decode(q),t)-y,q0,
                          bounds=(lo,hi),ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=1000)
        if opt.success:
            candidates.append((float(opt.fun@opt.fun),decode(opt.x),True))
    if not candidates:raise RuntimeError(f'No converged {model} fit')
    return min(candidates,key=lambda v:v[0])[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    df=pd.read_csv(args.source,float_precision='round_trip')
    rows=[];predictions=[];params=[]
    for sid in SUBSTRATES:
        t,y=df[[f'time_{sid}',f'S_{sid}']].dropna().to_numpy().T
        rates,meta=extract_rates(t,y)
        end=int(rates.end_row.iloc[-1])+1
        tt=(t[:end]-t[0])/meta['time_span'];yy=(y[:end]-meta['baseline'])/meta['signal_span']
        previous={}
        for model in ['S1','S2','S3','S4','S5']:
            p=fit_concentration(model,tt,yy,previous);previous[model]=p
            pred=concentration(model,p,tt);res=yy-pred;sse=float(res@res);n=len(yy)
            k=len(p)+1 # model initial condition plus rate coefficients plus noise variance
            aicc=n*(np.log(2*np.pi)+1+np.log(sse/n))+2*k+2*k*(k+1)/(n-k-1)
            rows.append(dict(substrate=sid,model=model,n_points=n,SSE=sse,
                             R2_centered=1-sse/np.sum((yy-yy.mean())**2),AICc=aicc,
                             residual_lag1=np.corrcoef(res[:-1],res[1:])[0,1],
                             K_boundary=bool(model in ('S2','S4','S5') and (p[2]<.0001001 or p[2]>9990))))
            params.append(dict(substrate=sid,model=model,normalized_parameters=json.dumps(p.tolist())))
            for ti,yi,pi in zip(t[:end],y[:end],meta['baseline']+meta['signal_span']*pred):
                predictions.append(dict(substrate=sid,model=model,time=ti,observed=yi,predicted=pi,residual=yi-pi))
        s2=[r for r in rows if r['substrate']==sid and r['model']=='S2'][0]['SSE']
        assert rows[-1]['SSE']<=s2+1e-9
        print(sid,'done',flush=True)
    stats=pd.DataFrame(rows)
    stats['delta_AICc']=stats.AICc-stats.groupby('substrate').AICc.transform('min')
    stats.to_csv(args.out/'concentration_comparison.csv',index=False)
    pd.DataFrame(params).to_csv(args.out/'concentration_parameters.csv',index=False)
    preds=pd.DataFrame(predictions);preds.to_csv(args.out/'concentration_predictions.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':10})
    for residual in (False,True):
        fig,axes=plt.subplots(2,4,figsize=(14,7))
        for ax,sid in zip(axes.flat,SUBSTRATES):
            if not residual:
                obs=preds.query('substrate==@sid and model=="S1"')
                ax.plot(obs.time,obs.observed,'.',ms=3,color='0.35')
            else:ax.axhline(0,color='0.5',lw=.7)
            for model in MODELS:
                f=preds.query('substrate==@sid and model==@model')
                ax.plot(f.time,f.residual if residual else f.predicted,color=COLORS[model],label=model,lw=1.3)
            ax.set(title=sid,xlabel='Time (input units)',ylabel='Signal residual' if residual else 'Supplied signal')
            ax.tick_params(direction='out',length=4,width=1,color='black')
        axes.flat[-1].axis('off');axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',frameon=False)
        fig.suptitle('Secondary integrated-curve check; working-iid errors, inspect autocorrelation')
        fig.tight_layout();name='concentration_residuals' if residual else 'concentration_fits'
        fig.savefig(args.out/(name+'.png'),dpi=220);fig.savefig(args.out/(name+'.pdf'));plt.close(fig)


if __name__=='__main__':main()

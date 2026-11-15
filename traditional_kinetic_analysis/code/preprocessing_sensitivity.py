#!/usr/bin/env python3
"""Finite, explicitly recorded sensitivity analysis for current 1g/1h.

No data edits, random smoothing factors, class-directed stopping, or filtering
of negative rate estimates. Reports every configuration in analysis_plan.json.

Methods references (also cited in SI Section 10):
Local-polynomial differentiation: Savitzky and Golay, Anal. Chem. 1964,
36, 1627-1639. DOI: 10.1021/ac60214a047.
Small-sample AIC correction: Hurvich and Tsai, Biometrika 1989,
76, 297-307. DOI: 10.1093/biomet/76.2.297.
Kinetic model discrimination: Taylor et al., React. Chem. Eng. 2021,
6, 1404-1411. DOI: 10.1039/D1RE00098E."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import nnls, minimize_scalar
from scipy.signal import savgol_filter
from kinetic_validation import MODELS, PARAM_COUNT, predict_rate, mad_sigma

HERE=Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Method:
    baseline:str
    name:str
    window:int
    degree:int
    stride:int
    floor:float=.05

    @property
    def key(self):return f'{self.baseline}__{self.name}__floor{self.floor:g}'


def background(y,name):
    if name=='zero':return 0.0
    fraction={'tail05':.05,'tail10':.10,'tail15':.15}[name]
    return float(np.median(y[-max(5,int(np.ceil(fraction*len(y)))):]))


def extract(t,y,m):
    t=np.asarray(t,dtype=float);y=np.asarray(y,dtype=float)
    if not np.isfinite(t).all() or not np.isfinite(y).all() or (np.diff(t)<=0).any():
        raise ValueError('Invalid observations')
    baseline=background(y,m.baseline);scale=y[0]-baseline;duration=t[-1]-t[0]
    if scale<=0:raise ValueError('Nonpositive signal span')
    tau=(t-t[0])/duration;x=(y-baseline)/scale
    sigma=max(mad_sigma(np.diff(y,n=2))/np.sqrt(6)/scale,1e-6)
    variances=np.zeros(len(y));coverage=np.zeros(len(y));rows=[];D=[]
    for start in range(0,len(y),m.stride):
        stop=min(len(y),start+m.window)
        if stop-start<max(4,m.degree+2):break
        tt=tau[start:stop];yy=x[start:stop];half=(tt[-1]-tt[0])/2
        u=(tt-tt.mean())/half
        X=np.vander(u,m.degree+1,increasing=True);P=np.linalg.pinv(X)
        beta=P@yy
        xx=float(beta[0])
        if xx<m.floor:break
        res=yy-X@beta
        var=max(float(res@res/(len(tt)-m.degree-1)),sigma*sigma)
        derivative=np.zeros(len(y));derivative[start:stop]=-P[1]/half
        D.append(derivative)
        variances[start:stop]+=var;coverage[start:stop]+=1
        rows.append(dict(start_row=start,end_row=stop-1,t=float(t[start:stop].mean()),
                         x=xx,rate=float(derivative@x),local_variance=var))
    if len(rows)<6:raise ValueError(f'Only {len(rows)} rate estimates; cannot compare all five by AICc')
    D=np.asarray(D)
    point_var=np.divide(variances,coverage,out=np.full_like(variances,sigma*sigma),where=coverage>0)
    cov=(D*point_var)@D.T
    L=cholesky(cov,lower=True) # no diagonal-only approximation for overlapping SG
    invL=solve_triangular(L,np.eye(len(rows)),lower=True)
    frame=pd.DataFrame(rows);frame['rate_se']=np.sqrt(np.diag(cov))
    meta=dict(baseline=baseline,signal_span=scale,time_span=duration,noise_sigma=sigma,
              input_points=len(y),rate_points=len(frame),negative_rates=int((frame.rate<0).sum()),
              logdet_cov=2*np.log(np.diag(L)).sum(),cov_condition=float(np.linalg.cond(cov)),
              last_time=float(frame.t.iloc[-1]))
    return frame,invL,meta


def fit_profile(model,x,y,invL,s2=None):
    yw=invL@y
    if model in ('S1','S3'):
        basis=np.ones_like(x) if model=='S1' else x
        bw=invL@basis;a=max(0,float(bw@yw/(bw@bw)))
        return np.array([a])
    def at(logK):
        K=np.exp(logK);q=x/(K+x)
        if model=='S4':q=q*x
        B=np.column_stack((q,x)) if model=='S5' else q[:,None]
        BW=invL@B;norm=np.linalg.norm(BW,axis=0)
        coef,_=nnls(BW/norm,yw,maxiter=100);coef/=norm
        res=BW@coef-yw
        return float(res@res),np.array([coef[0],K]+([coef[1]] if model=='S5' else []))
    grid=np.linspace(np.log(1e-4),np.log(1e4),65);scores=[at(v)[0] for v in grid]
    candidates=[at(grid[0]),at(grid[-1]),at(grid[np.argmin(scores)])]
    for i in range(1,len(grid)-1):
        if scores[i]<=scores[i-1] and scores[i]<=scores[i+1]:
            opt=minimize_scalar(lambda v:at(v)[0],bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-9})
            candidates.append(at(opt.x))
    if model=='S5':
        p=np.r_[s2,0.];r=invL@(predict_rate('S5',p,x)-y)
        candidates.append((float(r@r),p))
    return min(candidates,key=lambda z:z[0])[1]


def fit_all(frame,invL,meta):
    x=frame.x.to_numpy();y=frame.rate.to_numpy();n=len(x);fits={};rows=[]
    ones=invL@np.ones(n);mean=(ones@(invL@y))/(ones@ones)
    z0=invL@(y-mean);sst=float(z0@z0)
    for model in MODELS:
        p=fit_profile(model,x,y,invL,fits.get('S2'));fits[model]=p
        r=y-predict_rate(model,p,x);z=invL@r;Q=float(z@z)
        k=PARAM_COUNT[model]+1
        aicc=n*(np.log(2*np.pi)+1+np.log(max(Q/n,1e-300)))+meta['logdet_cov']+2*k+2*k*(k+1)/(n-k-1)
        rho=float(np.corrcoef(z[:-1],z[1:])[0,1]) if np.std(z)>0 else np.nan
        rows.append(dict(model=model,n_rate=n,Q=Q,R2_GLS=1-Q/sst if sst>0 else np.nan,
                         AICc=float(aicc),whitened_residual_lag1=rho,
                         K_boundary=bool(model in ('S2','S4','S5') and (p[1]<=1.001e-4 or p[1]>=1e4/1.001)),
                         parameters=json.dumps(p.tolist()),mean_parameters=PARAM_COUNT[model]))
    df=pd.DataFrame(rows);df['delta_AICc']=df.AICc-df.AICc.min()
    assert df.set_index('model').loc['S5','Q']<=df.set_index('model').loc['S2','Q']+1e-8
    return fits,df


def analyze(sid,t,y,m,repeats,seed):
    frame,invL,meta=extract(t,y,m);fits,stats=fit_all(frame,invL,meta)
    stats.insert(0,'substrate',sid);stats.insert(1,'configuration',m.key)
    frame.insert(0,'substrate',sid);frame.insert(1,'configuration',m.key)
    for model,p in fits.items():
        frame[model+'_pred']=predict_rate(model,p,frame.x)
        frame[model+'_residual']=frame.rate-frame[model+'_pred']
    trend=savgol_filter(y,11,2,mode='interp');res=y-trend;res-=res.mean()
    length=int(np.ceil(len(y)**(1/3)));rng=np.random.default_rng(seed);boot=[];fail=[]
    for b in range(repeats):
        offset=int(rng.integers(length))
        multipliers=np.repeat(rng.choice([-1,1],size=int(np.ceil((len(y)+offset)/length))),length)[offset:offset+len(y)]
        simulated=trend+res*multipliers
        try:
            f,L,mm=extract(t,simulated,m);pp,ss=fit_all(f,L,mm)
            row=ss.set_index('model')
            boot.append(dict(replicate=b,best=ss.loc[ss.AICc.idxmin(),'model'],
                             delta_S5_S2=row.loc['S5','AICc']-row.loc['S2','AICc'],
                             k_side=pp['S5'][2]/mm['time_span']))
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as e:
            fail.append(dict(replicate=b,error=str(e)))
    stat=stats.set_index('model');delta=stat.loc['S5','AICc']-stat.loc['S2','AICc']
    summary_meta={k:v for k,v in meta.items() if k!='baseline'}
    summary=dict(substrate=sid,configuration=m.key,**asdict(m),baseline_value=meta['baseline'],**summary_meta,
                 best=stats.loc[stats.AICc.idxmin(),'model'],delta_S5_minus_S2=delta,
                 k_side=fits['S5'][2]/meta['time_span'],S5_K_boundary=bool(stat.loc['S5','K_boundary']),
                 successful_bootstrap=len(boot),failed_bootstrap=len(fail))
    if boot:
        bb=pd.DataFrame(boot)
        low,hi=np.quantile(bb.k_side,[.025,.975])
        summary.update(k_side_low=low,k_side_high=hi,
                       bootstrap_S5_lowest_fraction=float((bb.best=='S5').mean()),
                       bootstrap_delta_lt_minus2_fraction=float((bb.delta_S5_S2 < -2).mean()))
    for row in boot:row.update(substrate=sid,configuration=m.key)
    return summary,stats,frame,boot,fail


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--bootstrap',type=int,default=100)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((HERE/'analysis_plan.json').read_text())
    source=HERE/'input_data_current.csv';payload=source.read_bytes()
    assert hashlib.sha256(payload).hexdigest()==plan['source_sha256']
    df=pd.read_csv(source,float_precision='round_trip')
    configurations=[]
    for b in plan['rate_baselines']:
        for method in plan['rate_methods']:configurations.append(Method(b,**method))
        configurations.append(Method(b,'linear7',7,1,7,.10))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(args.out/'run.log')])
    summaries=[];statistics=[];points=[];boot=[];errors=[];start=time.monotonic();done=0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        active={}
        for i,sid in enumerate(plan['targets']):
            t,y=df[[f'time_{sid}',f'S_{sid}']].dropna().to_numpy().T
            for m in configurations:
                f=ex.submit(analyze,sid,t,y,m,args.bootstrap,plan['bootstrap']['seed']+i)
                active[f]=(sid,m)
        for f in as_completed(active):
            sid,m=active[f];done+=1
            try:
                summ,st,fr,bb,fail=f.result();summaries.append(summ);statistics.append(st);points.append(fr);boot.extend(bb)
                if fail:errors.append(dict(substrate=sid,configuration=m.key,bootstrap_failures=fail))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as e:
                errors.append(dict(substrate=sid,configuration=m.key,error=str(e)))
            elapsed=time.monotonic()-start
            if done%8==0 or done==len(active):
                logging.info('%d/%d configurations elapsed=%.1fs ETA=%.1fs',done,len(active),elapsed,elapsed*(len(active)-done)/done)
    pd.DataFrame(summaries).sort_values(['substrate','configuration']).to_csv(args.out/'configuration_summary.csv',index=False)
    pd.concat(statistics,ignore_index=True).sort_values(['substrate','configuration','model']).to_csv(args.out/'all_model_comparisons.csv',index=False)
    pd.concat(points,ignore_index=True).sort_values(['substrate','configuration','t']).to_csv(args.out/'rates_predictions_residuals.csv',index=False)
    pd.DataFrame(boot).to_csv(args.out/'bootstrap_results.csv',index=False)
    summary=dict(attempted=len(active),completed=len(summaries),errors=errors,source_sha256=plan['source_sha256'],
                 plan_sha256=hashlib.sha256((HERE/'analysis_plan.json').read_bytes()).hexdigest(),
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 bootstrap_per_configuration=args.bootstrap,
                 covariance='D diag(local pooled point variance) D.T; overlap retained; working Gaussian covariance',
                 note='All configurations are exploratory sensitivity checks, not independent hypothesis tests.')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    assert source.read_bytes()==payload
    logging.info('Complete: %s',args.out)


if __name__=='__main__':main()

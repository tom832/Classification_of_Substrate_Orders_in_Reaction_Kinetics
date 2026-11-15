#!/usr/bin/env python3
"""Unsmooth concentration fits under iid and stationary AR(1) errors.

Same data window/response for S1-S5 at each fixed background. All results,
including optimizer and bound diagnostics, are retained. AR(1) is a working
error model, not a remedy for arbitrary mean-model misspecification.

Methods references (also cited in SI Section 10):
Local-polynomial differentiation: Savitzky and Golay, Anal. Chem. 1964,
36, 1627-1639. DOI: 10.1021/ac60214a047.
Small-sample AIC correction: Hurvich and Tsai, Biometrika 1989,
76, 297-307. DOI: 10.1093/biomet/76.2.297.
Kinetic model discrimination: Taylor et al., React. Chem. Eng. 2021,
6, 1404-1411. DOI: 10.1039/D1RE00098E."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import logging
import os
from pathlib import Path
import time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar, least_squares
from concentration_check import concentration, fit_concentration
from preprocessing_sensitivity import Method, extract
from kinetic_validation import MODELS

HERE=Path(__file__).resolve().parent.parent


def innovation_residual(r,rho):
    return np.r_[np.sqrt(1-rho*rho)*r[0],r[1:]-rho*r[:-1]]


def profile_n2ll(r,rho):
    z=innovation_residual(r,rho);q=float(z@z);n=len(r)
    return float(n*(np.log(2*np.pi)+1+np.log(max(q/n,1e-300)))-np.log1p(-rho*rho))


def fit_ar1(model,t,y,iid,previous):
    def encode(p):
        return np.array([p[0],np.log(max(p[1],np.exp(-14)))]+(
            [np.log(max(1e-4,p[2]))] if model in ('S2','S4','S5') else [])+([p[3]] if model=='S5' else []))
    def decode(q):
        return np.array([q[0],np.exp(q[1])]+(
            [np.exp(q[2])] if model in ('S2','S4','S5') else [])+([q[3]] if model=='S5' else []))
    low=[.25,-14];high=[2,14]
    if model in ('S2','S4','S5'):low.append(np.log(1e-4));high.append(np.log(1e4))
    if model=='S5':low.append(0);high.append(100)
    starts=[iid]
    if model in ('S2','S4','S5'):
        for K in (.001,.1,1,100):
            starts.append(np.array([iid[0],2*(1+K),K]+([.5] if model=='S5' else [])))
    candidates=[];audit=[]
    if model=='S5':
        p,rho,_=previous['S2'];p=np.r_[p,0.]
        candidates.append((profile_n2ll(y-concentration(model,p,t),rho),p,rho,'S2_boundary'))
        starts.append(p)
        q,rho,_=previous['S3'];p=np.array([q[0],0.,1.,q[1]])
        candidates.append((profile_n2ll(y-concentration(model,p,t),rho),p,rho,'S3_boundary'))
    def objective(q):return profile_n2ll(y-concentration(model,decode(q[:-1]),t),q[-1])
    for k,start in enumerate(starts):
        for rho0 in (0.,.6):
            q=encode(start);rho=rho0
            # Coordinate descent gives a well-scaled deterministic starting point.
            for _ in range(6):
                opt=least_squares(lambda zz:innovation_residual(y-concentration(model,decode(zz),t),rho),
                                  np.clip(q,low,high),bounds=(low,high),max_nfev=400,
                                  ftol=1e-9,xtol=1e-9,gtol=1e-9)
                q=opt.x;r=y-concentration(model,decode(q),t)
                fit_rho=minimize_scalar(lambda v:profile_n2ll(r,v),bounds=(-.95,.95),method='bounded',options={'xatol':1e-8})
                newrho=float(fit_rho.x)
                if abs(newrho-rho)<1e-7:rho=newrho;break
                rho=newrho
            joint=minimize(objective,np.r_[q,rho],method='L-BFGS-B',
                           bounds=list(zip(low+[-.95],high+[.95])),
                           options=dict(maxiter=1000,ftol=1e-11,gtol=1e-6,maxls=30))
            audit.append(dict(start=k,rho_start=rho0,success=bool(joint.success),
                              objective=float(joint.fun),message=str(joint.message)))
            if joint.success and np.isfinite(joint.fun):
                candidates.append((float(joint.fun),decode(joint.x[:-1]),float(joint.x[-1]),'joint_optimizer'))
    if not candidates:raise RuntimeError(f'No converged joint {model} AR1 fit')
    _,p,rho,origin=min(candidates,key=lambda r:r[0])
    return p,rho,dict(selected_origin=origin,starts=audit)


def analyze(sid,t,y,baseline):
    rates,_,meta=extract(t,y,Method(baseline,'linear7',7,1,7))
    end=int(rates.end_row.iloc[-1])+1
    tt=(t[:end]-t[0])/meta['time_span'];yy=(y[:end]-meta['baseline'])/meta['signal_span']
    iid={};ar={};rows=[];predictions=[];audit={}
    for model in MODELS:
        p=fit_concentration(model,tt,yy,iid);iid[model]=p
        pp,rho,detail=fit_ar1(model,tt,yy,p,ar);ar[model]=(pp,rho,detail);audit[model]=detail
        for error_model,param,corr in [('iid',p,0.),('AR1',pp,rho)]:
            pred=concentration(model,param,tt);r=yy-pred;n=len(r)
            nll=profile_n2ll(r,corr);k=len(param)+1+int(error_model=='AR1')
            aicc=nll+2*k+2*k*(k+1)/(n-k-1)
            z=innovation_residual(r,corr)
            ac=lambda v:float(np.corrcoef(v[:-1],v[1:])[0,1]) if np.std(v)>0 else np.nan
            rows.append(dict(substrate=sid,baseline=baseline,error_model=error_model,model=model,
                             n_points=n,parameters=json.dumps(param.tolist()),rho=corr,n2loglik=nll,AICc=aicc,
                             R2_centered=1-float(r@r)/float(np.sum((yy-yy.mean())**2)),
                             raw_residual_lag1=ac(r),innovation_lag1=ac(z),
                             K_boundary=bool(model in ('S2','S4','S5') and (param[2]<1.001e-4 or param[2]>1e4/1.001)),
                             rho_boundary=bool(abs(corr)>.949),side_at_zero=bool(model=='S5' and param[3]<1e-6),
                             k_parameters=k,baseline_value=meta['baseline']))
            for ti,obs,hat,rr in zip(t[:end],y[:end],meta['baseline']+meta['signal_span']*pred,z):
                predictions.append(dict(substrate=sid,baseline=baseline,error_model=error_model,model=model,
                                        t=ti,observed=obs,predicted=hat,innovation=rr))
    frame=pd.DataFrame(rows)
    for err in ('iid','AR1'):
        f=frame[frame.error_model==err].set_index('model')
        assert f.loc['S5','n2loglik']<=f.loc['S2','n2loglik']+1e-6
    frame['delta_AICc']=frame.AICc-frame.groupby('error_model').AICc.transform('min')
    return frame,pd.DataFrame(predictions),dict(substrate=sid,baseline=baseline,optimizer=audit)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((HERE/'analysis_plan.json').read_text());source=HERE/'input_data_current.csv'
    raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()==plan['source_sha256']
    df=pd.read_csv(source,float_precision='round_trip')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(args.out/'run.log')])
    tables=[];predictions=[];audit=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        active={}
        for sid in plan['targets']:
            t,y=df[[f'time_{sid}',f'S_{sid}']].dropna().to_numpy().T
            for b in plan['rate_baselines']:
                active[executor.submit(analyze,sid,t,y,b)]=(sid,b)
        for f in as_completed(active):
            sid,b=active[f];st,pr,au=f.result();tables.append(st);predictions.append(pr);audit.append(au)
            elapsed=time.monotonic()-start
            logging.info('%s %s done (%d/%d) elapsed %.1fs ETA %.1fs',sid,b,len(tables),len(active),elapsed,elapsed*(len(active)-len(tables))/len(tables))
    pd.concat(tables,ignore_index=True).to_csv(args.out/'all_model_comparisons.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(args.out/'predictions_and_innovations.csv',index=False)
    (args.out/'summary.json').write_text(json.dumps(dict(source_sha256=plan['source_sha256'],
        plan_sha256=hashlib.sha256((HERE/'analysis_plan.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),audit=audit,
        note='Profile stationary AR1 likelihood includes first residual normalization, determinant and estimated rho.'),indent=2)+'\n')
    assert raw==source.read_bytes()


if __name__=='__main__':main()

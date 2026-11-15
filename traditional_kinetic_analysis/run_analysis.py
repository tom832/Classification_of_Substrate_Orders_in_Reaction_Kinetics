#!/usr/bin/env python3
"""Uniform seven-substrate SG reanalysis, with all fixed sensitivity checks.

Copies of the numerical engines are supplied in code/. Reads the supplied experimental input and creates a fresh output directory.

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
import sys
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'code'))
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from preprocessing_sensitivity import Method, extract, fit_all
from kinetic_validation import MODELS, predict_rate
from concentration_check import concentration, fit_concentration
from correlated_concentration_check import fit_ar1, profile_n2ll, innovation_residual

NAMES = {'S1':['c'], 'S2':['V','K'], 'S3':['k'], 'S4':['a','K'], 'S5':['V','K','k_side']}


def physical(parameter, value, meta):
    scale, T = meta['signal_span'], meta['time_span']
    if parameter == 'K': return value*scale, 'input signal'
    if parameter in ('V','c'): return value*scale/T, 'input signal / input time'
    return value/T, '1 / input time'


def rate_task(sid, t, y, settings, nboot, seed, primary):
    method = Method(**settings)
    frame, invL, meta = extract(t, y, method)
    fits, stats = fit_all(frame, invL, meta)
    for model in MODELS:
        pred = predict_rate(model, fits[model], frame.x)
        frame[model+'_prediction'] = pred
        frame[model+'_residual'] = frame.rate-pred
        frame[model+'_whitened_residual'] = invL@(frame.rate-pred)
    for f in (frame, stats):
        f.insert(0,'substrate',sid); f.insert(1,'configuration',method.key)
    params = []
    for model in MODELS:
        for name, value in zip(NAMES[model],fits[model]):
            converted, unit = physical(name, value, meta)
            params.append(dict(substrate=sid,configuration=method.key,model=model,parameter=name,
                               normalized_estimate=value,input_unit_estimate=converted,unit=unit))
    trend = savgol_filter(y,11,2,mode='interp')
    residual = y-trend; residual -= residual.mean()
    length = int(np.ceil(len(y)**(1/3)))
    rng = np.random.default_rng(seed); boot = []; bootpars = []; errors = []
    for repeat in range(nboot):
        offset = int(rng.integers(length))
        multipliers = np.repeat(rng.choice([-1,1],size=int(np.ceil((len(y)+offset)/length))),length)[offset:offset+len(y)]
        simulated = trend+residual*multipliers
        try:
            ff, LL, mm = extract(t,simulated,method)
            pp, ss = fit_all(ff,LL,mm); indexed = ss.set_index('model')
            boot.append(dict(substrate=sid,configuration=method.key,replicate=repeat,n_rate=len(ff),
                             best=ss.loc[ss.AICc.idxmin(),'model'],
                             delta_S5_minus_S2=indexed.loc['S5','AICc']-indexed.loc['S2','AICc']))
            for model in MODELS:
                for name, value in zip(NAMES[model],pp[model]):
                    converted, unit = physical(name,value,mm)
                    bootpars.append(dict(substrate=sid,configuration=method.key,replicate=repeat,
                                         model=model,parameter=name,normalized_estimate=value,input_unit_estimate=converted))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            errors.append(dict(substrate=sid,configuration=method.key,replicate=repeat,error=str(e)))
    summary = dict(substrate=sid,configuration=method.key,primary=primary,**meta,
                   best=stats.loc[stats.AICc.idxmin(),'model'],bootstrap_requested=nboot,
                   bootstrap_completed=len(boot),bootstrap_failures=len(errors),
                   first_input_row=int(frame.start_row.min()),last_input_row=int(frame.end_row.max()))
    return summary,stats,frame,params,boot,bootpars,errors


def concentration_task(sid,t,y,settings):
    rates,_,meta = extract(t,y,Method(**settings))
    end = int(rates.end_row.max())+1
    tt = (t[:end]-t[0])/meta['time_span']
    yy = (y[:end]-meta['baseline'])/meta['signal_span']
    iid, ar = {}, {}; rows, points, audit = [], [], []
    for model in ('S2','S3','S5'):
        p = fit_concentration(model,tt,yy,iid); iid[model] = p
        pa,rho,detail = fit_ar1(model,tt,yy,p,ar); ar[model] = (pa,rho,detail)
        audit.append(dict(substrate=sid,model=model,**detail))
        for error,pp,rr in [('iid',p,0.),('AR1',pa,rho)]:
            pred = concentration(model,pp,tt); r = yy-pred
            z = innovation_residual(r,rr); n=len(r); k=len(pp)+1+int(error=='AR1')
            nll = profile_n2ll(r,rr)
            corr = lambda q: float(np.corrcoef(q[:-1],q[1:])[0,1])
            rows.append(dict(substrate=sid,model=model,error_model=error,n_points=n,
                             parameters=json.dumps(pp.tolist()),rho=rr,n2loglik=nll,
                             AICc=nll+2*k+2*k*(k+1)/(n-k-1),
                             R2_concentration=1-float(r@r)/float(np.sum((yy-yy.mean())**2)),
                             residual_lag1=corr(r),innovation_lag1=corr(z),
                             K_boundary=bool(model!='S3' and (pp[2]<1.001e-4 or pp[2]>9990)),
                             side_at_zero=bool(model=='S5' and pp[3]<1e-6)))
            for j in range(n):
                points.append(dict(substrate=sid,model=model,error_model=error,input_row=j,
                                   t=t[j],observed=y[j],predicted=meta['baseline']+meta['signal_span']*pred[j],
                                   residual=meta['signal_span']*r[j],innovation=meta['signal_span']*z[j]))
    f = pd.DataFrame(rows)
    f['delta_AICc'] = f.AICc-f.groupby('error_model').AICc.transform('min')
    for _,ss in f.groupby('error_model'):
        ss = ss.set_index('model')
        assert ss.loc['S5','n2loglik'] <= min(ss.loc['S2','n2loglik'],ss.loc['S3','n2loglik'])+1e-6
    return f,pd.DataFrame(points),audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'rerun_results')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(); args.out.mkdir(exist_ok=False,parents=True)
    plan=json.loads((HERE/'analysis_plan.json').read_text())
    source=HERE/'input_data_current.csv'; raw=source.read_bytes()
    sha=lambda x:hashlib.sha256(x).hexdigest()
    assert sha(raw)==plan['source_sha256']
    df=pd.read_csv(source,float_precision='round_trip')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(args.out/'run.log')])
    sets=[plan['primary']]+plan['sensitivity']
    summaries=[];stats=[];rates=[];parameters=[];boot=[];bp=[];errors=[]
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        active={}
        for i,sid in enumerate(plan['substrates']):
            t,y=df[[f'time_{sid}',f'S_{sid}']].dropna().to_numpy().T
            for j,settings in enumerate(sets):
                nboot=plan['bootstrap']['primary' if j==0 else 'sensitivity']
                f=pool.submit(rate_task,sid,t,y,settings,nboot,plan['bootstrap']['seed']+i,j==0)
                active[f]=(sid,settings)
        for n,f in enumerate(as_completed(active),1):
            summ,st,fr,pa,bb,pp,ee=f.result()
            summaries.append(summ);stats.append(st);rates.append(fr)
            parameters.extend(pa);boot.extend(bb);bp.extend(pp);errors.extend(ee)
            elapsed=time.monotonic()-start
            if n%7==0 or n==len(active):
                logging.info('Rates %d/%d; elapsed %.1fs; ETA %.1fs',n,len(active),elapsed,elapsed*(len(active)-n)/n)
    ss=pd.DataFrame(summaries).sort_values(['substrate','configuration'])
    ss.to_csv(args.out/'configuration_summary.csv',index=False)
    st=pd.concat(stats).sort_values(['substrate','configuration','model'])
    st.to_csv(args.out/'all_rate_model_comparisons.csv',index=False)
    pd.concat(rates).sort_values(['substrate','configuration','t']).to_csv(args.out/'rates_predictions_residuals.csv',index=False)
    pd.DataFrame(boot).sort_values(['substrate','configuration','replicate']).to_csv(args.out/'bootstrap_model_comparisons.csv',index=False)
    bpars=pd.DataFrame(bp).sort_values(['substrate','configuration','model','parameter','replicate'])
    bpars.to_csv(args.out/'bootstrap_parameters.csv',index=False)
    keys=['substrate','configuration','model','parameter']
    quant=bpars.groupby(keys)[['normalized_estimate','input_unit_estimate']].quantile([.025,.975]).unstack()
    quant.columns=['normalized_low','normalized_high','input_unit_low','input_unit_high']
    pars=pd.DataFrame(parameters).merge(quant.reset_index(),on=keys)
    pars.to_csv(args.out/'parameter_estimates.csv',index=False)
    primary=Method(**plan['primary']).key
    main=st[st.configuration==primary]
    main.to_csv(args.out/'primary_model_comparison.csv',index=False)
    for value,name in [('R2_GLS','Table_S13a_centered_GLS_R2.csv'),('delta_AICc','Table_S13b_delta_AICc.csv')]:
        main.pivot(index='substrate',columns='model',values=value).reindex(index=plan['substrates'],columns=MODELS).to_csv(args.out/name)
    ss.pivot(index='substrate',columns='configuration',values='best').to_csv(args.out/'all_sensitivity_rankings.csv')

    cc=[];pr=[];audit=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[]
        for sid in plan['substrates']:
            t,y=df[[f'time_{sid}',f'S_{sid}']].dropna().to_numpy().T
            futures.append(pool.submit(concentration_task,sid,t,y,plan['primary']))
        for n,f in enumerate(as_completed(futures),1):
            c,p,a=f.result();cc.append(c);pr.append(p);audit.extend(a)
            elapsed=time.monotonic()-start
            logging.info('Concentration %d/7; elapsed %.1fs; ETA %.1fs',n,elapsed,elapsed*(7-n)/n)
    pd.concat(cc).sort_values(['substrate','error_model','model']).to_csv(args.out/'concentration_model_comparison.csv',index=False)
    pd.concat(pr).sort_values(['substrate','error_model','model','t']).to_csv(args.out/'concentration_predictions_residuals.csv',index=False)
    (args.out/'optimizer_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (args.out/'summary.json').write_text(json.dumps(dict(
        source_sha256=sha(raw),plan_sha256=sha((HERE/'analysis_plan.json').read_bytes()),
        code_sha256={str(f.relative_to(HERE)):sha(f.read_bytes()) for f in [Path(__file__)]+sorted((HERE/'code').glob('*.py'))},
        primary=primary,configurations=len(ss),bootstrap_completed=len(boot),bootstrap_failures=errors,
        note='Exploratory selection acknowledged; these outputs do not establish unique mechanisms.'),indent=2)+'\n')
    assert source.read_bytes()==raw
    logging.info('Complete: %s',args.out)


if __name__=='__main__':main()

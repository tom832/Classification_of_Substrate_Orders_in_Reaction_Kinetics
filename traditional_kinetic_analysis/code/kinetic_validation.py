#!/usr/bin/env python3
"""Reviewer 2, point 2: common-scale rate fitting of the CURRENT supplied CSV.

No ML probabilities, desired classes, stochastic point selection, or fitted
polynomial degree is used to choose data/preprocessing. All source files remain
unchanged. Output is an explicitly new directory.

Run with the existing cosork/my_org_mech Python environment:
  python kinetic_validation.py --out results --bootstrap 200 --workers 4

Concentration and time retain their supplied units in exports. Normalized
coordinates are internal numerical scalings, not a conversion of units.
The input CSV is already processed; this is a conditional reanalysis, not
validation against independent instrument exports or biological replicates.

Methods references (also cited in SI Section 10):
Local-polynomial differentiation: Savitzky and Golay, Anal. Chem. 1964,
36, 1627-1639. DOI: 10.1021/ac60214a047.
Small-sample AIC correction: Hurvich and Tsai, Biometrika 1989,
76, 297-307. DOI: 10.1093/biomet/76.2.297.
Kinetic model discrimination: Taylor et al., React. Chem. Eng. 2021,
6, 1404-1411. DOI: 10.1039/D1RE00098E."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, nnls
from scipy.signal import savgol_filter
import scipy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT/'input_data_current.csv'
SUBSTRATES = ['1a', '1b', '1d', '1e', '1f', '1g', '1h']
MODELS = ['S1', 'S2', 'S3', 'S4', 'S5']
COLORS = dict(zip(MODELS, ['#FFA878', '#79A1E3', '#88D59A', '#E37275', '#AB9FD6']))
PARAM_COUNT = dict(S1=1, S2=2, S3=1, S4=2, S5=3)
PARAM_NAMES = dict(S1=['c'], S2=['V','K'], S3=['k'], S4=['a','K'], S5=['V','K','k_side'])
K_LIMITS = (1e-4, 1e4)  # numerical range in units of observed concentration span


@dataclass(frozen=True)
class Config:
    name: str = 'primary'
    baseline: str = 'tail10'
    block_points: int = 7
    floor: float = 0.05
    weighted: bool = True
    k_min: float = K_LIMITS[0]
    k_max: float = K_LIMITS[1]


CONFIGS = [Config(), Config('block5', block_points=5), Config('block9', block_points=9),
           Config('block11', block_points=11), Config('last_point_baseline', baseline='last'),
           Config('no_baseline_subtraction', baseline='zero'), Config('floor10', floor=.10),
           Config('unweighted_rate', weighted=False),
           Config('wide_K_range', k_min=1e-6, k_max=1e6)]


def mad_sigma(values):
    values = np.asarray(values)
    return float(np.median(np.abs(values - np.median(values))) / .6744897501960817)


def baseline_value(y, method):
    if method == 'zero': return 0.0
    if method == 'last': return float(y[-1])
    if method == 'tail10': return float(np.median(y[-max(5, int(np.ceil(.1*len(y)))):]))
    raise ValueError(method)


def extract_rates(t, y, config=Config()):
    """Disjoint local OLS slopes. No positive-rate deletion or interpolation.

    Rate SE uses max(local residual variance, robust high-frequency variance).
    At a uniform grid the second difference has variance 6*sigma_y**2 under
    independent measurement noise. This is a working variance model; the
    signal-level block-wild bootstrap checks uncertainty under correlation.
    """
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if len(t) < 15 or not np.isfinite(t).all() or not np.isfinite(y).all() or (np.diff(t)<=0).any():
        raise ValueError('Need >=15 paired finite observations with strictly increasing time')
    base = baseline_value(y, config.baseline)
    scale = float(y[0] - base)
    duration = float(t[-1]-t[0])
    if scale <= 0: raise ValueError('Initial signal must exceed the fixed background')
    noise = max(mad_sigma(np.diff(y, n=2))/np.sqrt(6), scale*1e-6)
    normalized_t = (t-t[0])/duration
    normalized_y = (y-base)/scale
    rows = []
    for start in range(0, len(t), config.block_points):
        stop = min(start+config.block_points,len(t))
        if stop-start < 4: break
        tt, yy = normalized_t[start:stop], normalized_y[start:stop]
        tc = tt-tt.mean()
        sxx = tc @ tc
        slope = float(tc @ (yy-yy.mean())/sxx)
        residual = yy-yy.mean()-slope*tc
        variance = max(float(residual @ residual/(len(tt)-2)), (noise/scale)**2)
        concentration = float(yy.mean())
        # End the common observed window at the first block below the floor.
        if concentration < config.floor: break
        rows.append(dict(block=len(rows),start_row=start,end_row=stop-1,n_points=stop-start,
                         t=float(t[start:stop].mean()),x=concentration,
                         concentration=concentration*scale,rate=-slope,
                         rate_se=np.sqrt(variance/sxx),
                         rate_supplied_units=-slope*scale/duration,
                         rate_se_supplied_units=np.sqrt(variance/sxx)*scale/duration))
    result = pd.DataFrame(rows)
    if len(result) < 6: raise ValueError(f'Only {len(result)} usable rate blocks')
    metadata = dict(baseline=base,initial_observed=float(y[0]),signal_span=scale,
                    time_span=duration,input_points=len(y),rate_blocks=len(result),
                    sigma_signal=noise,negative_rates=int((result.rate<0).sum()),
                    baseline_tail_points=max(5,int(np.ceil(.1*len(y)))),
                    last_signal=float(y[-1]),last_fitted_time=float(result.t.iloc[-1]))
    return result, metadata


def predict_rate(model, p, x):
    x = np.asarray(x)
    if model=='S1': return np.full_like(x,p[0],dtype=float)
    if model=='S3': return p[0]*x
    if model=='S2': return p[0]*x/(p[1]+x)
    if model=='S4': return p[0]*x*x/(p[1]+x)
    if model=='S5': return p[0]*x/(p[1]+x)+p[2]*x
    raise ValueError(model)


def fit_model(model, x, y, se, config=Config(), nested_s2=None):
    """Profile K, solve nonnegative linear amplitudes exactly at each K.

    Same objective and weights for every candidate. S5 always explicitly
    includes the optimized S2 solution with k_side=0 as a feasible candidate.
    """
    x,y,se = map(np.asarray,(x,y,se))
    weight_scale = se if config.weighted else np.ones_like(se)
    yw = y/weight_scale
    if model in ('S1','S3'):
        basis = np.ones_like(x) if model=='S1' else x
        bw = basis/weight_scale
        amplitude = max(0,float(bw @ yw/(bw @ bw)))
        return np.array([amplitude]), dict(K_boundary=False,profile_solutions=1)

    def at(logK):
        k = np.exp(logK)
        q = x/(k+x)
        if model=='S4': q *= x
        basis = np.column_stack((q,x)) if model=='S5' else q[:,None]
        bw = basis/weight_scale[:,None]
        # Normalize each basis to prevent K-scale conditioning artifacts.
        norms = np.sqrt((bw*bw).sum(axis=0))
        coefs, _ = nnls(bw/norms,yw,maxiter=100)
        coefs /= norms
        residual = (basis @ coefs-y)/weight_scale
        p = [coefs[0],k]+([coefs[1]] if model=='S5' else [])
        return float(residual @ residual), np.array(p)

    grid=np.linspace(np.log(config.k_min),np.log(config.k_max),65)
    scores=np.array([at(q)[0] for q in grid])
    candidates=[at(grid[0]),at(grid[-1])]
    for i in range(1,len(grid)-1):
        if scores[i] <= scores[i-1] and scores[i] <= scores[i+1]:
            opt=minimize_scalar(lambda v:at(v)[0],bounds=(grid[i-1],grid[i+1]),
                                method='bounded',options=dict(xatol=1e-9))
            candidates.append(at(opt.x))
    candidates.append(at(grid[scores.argmin()]))
    if model=='S5' and nested_s2 is not None:
        p=np.array([nested_s2[0],nested_s2[1],0.0])
        residual=(predict_rate('S5',p,x)-y)/weight_scale
        candidates.append((float(residual @ residual),p))
    _,p=min(candidates,key=lambda c:c[0])
    return p,dict(K_boundary=bool(p[1] <= config.k_min*1.001 or p[1]>=config.k_max/1.001),
                  profile_solutions=len(candidates))


def metrics(model, p, rates, config):
    y=rates.rate.to_numpy();x=rates.x.to_numpy();se=rates.rate_se.to_numpy()
    pred=predict_rate(model,p,x);res=y-pred
    scale=se if config.weighted else np.ones_like(se)
    z=res/scale
    q=float(z@z)
    w=1/scale**2
    mean=float(np.sum(w*y)/w.sum())
    centered=float(np.sum(w*(y-mean)**2))
    sst=float(np.sum((y-y.mean())**2))
    r2=1-float(res@res)/sst if sst>0 else np.nan
    wr2=1-q/centered if centered>0 else np.nan
    # Working Gaussian likelihood with a common unknown residual multiplier.
    # AICc is reported descriptively, not as a definitive mechanistic verdict.
    n=len(y); k=PARAM_COUNT[model]+1
    n2loglik=n*(np.log(2*np.pi)+1+np.log(max(q/n,1e-300)))+2*np.log(scale).sum()
    aic=n2loglik+2*k
    aicc=aic+2*k*(k+1)/(n-k-1) if n>k+1 else np.inf
    acf=float(np.corrcoef(z[:-1],z[1:])[0,1]) if len(z)>2 and np.std(z)>0 else np.nan
    return dict(n_rate=n,mean_parameters=PARAM_COUNT[model],likelihood_parameters=k,
                SSE=float(res@res),weighted_SSE=q,RMSE=float(np.sqrt(np.mean(res**2))),
                R2_centered=r2,R2_weighted=wr2,AICc=float(aicc),residual_lag1=acf,
                residual_DW=float(np.diff(z)@np.diff(z)/q) if q>0 else np.nan)


def fit_all(rates,config):
    fits={};records=[]
    for model in MODELS:
        p,detail=fit_model(model,rates.x,rates.rate,rates.rate_se,config,fits.get('S2'))
        fits[model]=p
        records.append(dict(model=model,**metrics(model,p,rates,config),**detail))
    frame=pd.DataFrame(records)
    frame['delta_AICc']=frame.AICc-frame.AICc.min()
    frame['AICc_weight']=np.exp(-.5*frame.delta_AICc)
    frame['AICc_weight']/=frame.AICc_weight.sum()
    q=frame.set_index('model').weighted_SSE
    if q['S5']>q['S2']+1e-8*max(1,q['S2']):
        raise AssertionError('Nested model constraint failed: S5 SSE > S2 SSE')
    return fits,frame


def physical_parameters(model,p,meta):
    p=np.array(p,dtype=float).copy();s=meta['signal_span'];t=meta['time_span']
    if model in ('S1','S2','S5'): p[0]*=s/t
    else: p[0]/=t
    if model in ('S2','S4','S5'): p[1]*=s
    if model=='S5': p[2]/=t
    return p


def bootstrap_signal(t,y,repeats,seed,config=Config()):
    """Block-wild residual bootstrap of supplied observations, then rerun extraction.

    One fixed quadratic Savitzky-Golay trend (11 points) defines residuals only
    for bootstrap generation. It is NEVER used for the primary rate estimates.
    Blocks of ceil(N**(1/3)) receive independent Rademacher multipliers.
    This propagates background and concentration/slope uncertainty, conditional
    on the supplied processed curve; it is not an independent-replicate CI.
    """
    n=len(y)
    window=min(11,n if n%2 else n-1)
    trend=savgol_filter(y,window_length=window,polyorder=2,mode='interp')
    residual=y-trend;residual-=residual.mean()
    length=int(np.ceil(n**(1/3)))
    rng=np.random.default_rng(seed)
    values=[];failed=[]
    for b in range(repeats):
        offset=int(rng.integers(length))
        mult=np.repeat(rng.choice([-1,1],size=int(np.ceil((n+offset)/length))),length)[offset:offset+n]
        generated=trend+residual*mult
        try:
            rates,meta=extract_rates(t,generated,config)
            fits,stats=fit_all(rates,config)
            for model in MODELS:
                rec=dict(replicate=b,model=model,rate_blocks=len(rates),
                         **{k:float(v) for k,v in zip(PARAM_NAMES[model],physical_parameters(model,fits[model],meta))})
                rec['AICc']=float(stats.set_index('model').loc[model,'AICc'])
                values.append(rec)
        except (ValueError,RuntimeError) as exc:
            failed.append(dict(replicate=b,reason=str(exc)))
    return values,failed,dict(trend_window=window,trend_degree=2,wild_block_length=length)


def analyze_one(sid,t,y,repeats,seed):
    rates,meta=extract_rates(t,y)
    fits,stats=fit_all(rates,Config())
    stats.insert(0,'substrate',sid)
    stats['RMSE_supplied_units']=stats.RMSE*meta['signal_span']/meta['time_span']
    rates.insert(0,'substrate',sid)
    for model,p in fits.items():
        rates[f'{model}_predicted']=predict_rate(model,p,rates.x)
        rates[f'{model}_residual']=rates.rate-rates[f'{model}_predicted']
    sensitivity=[]
    for config in CONFIGS:
        r,m=extract_rates(t,y,config)
        ff,ss=fit_all(r,config)
        ss.insert(0,'substrate',sid);ss.insert(1,'configuration',config.name)
        ss['baseline']=m['baseline'];ss['s5_k_side']=ff['S5'][2]/m['time_span']
        sensitivity.append(ss)
    boot,failed,boot_config=bootstrap_signal(t,y,repeats,seed)
    return sid,dict(rates=rates,metadata=meta,fits=fits,stats=stats,
                    sensitivity=pd.concat(sensitivity,ignore_index=True),
                    bootstrap=boot,bootstrap_failed=failed,bootstrap_config=boot_config)


def draw_figures(results,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':10,
                         'axes.linewidth':1,'xtick.direction':'out','ytick.direction':'out'})
    def style(ax):
        ax.tick_params(axis='both',direction='out',length=4,width=1,color='black',bottom=True,left=True)
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for ax,sid in zip(axes.flat,SUBSTRATES):
        d=results[sid];r=d['rates'];m=d['metadata'];scale=m['signal_span']/m['time_span']
        xgrid=np.linspace(max(.001,min(r.x)),max(r.x),200)
        ax.errorbar(r.x*m['signal_span'],r.rate*scale,yerr=r.rate_se*scale,
                    fmt='o',color='black',ms=3,capsize=2,label='Local slopes ± SE')
        for model in MODELS:
            ax.plot(xgrid*m['signal_span'],predict_rate(model,d['fits'][model],xgrid)*scale,
                    color=COLORS[model],label=model,lw=1.5)
        ax.set(title=sid,xlabel='Background-corrected signal',ylabel='Rate (signal / input-time unit)')
        style(ax)
    axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles,labels,loc='center',frameon=False)
    fig.suptitle('Current supplied curves: common rate-scale fits (conditional reanalysis)')
    fig.tight_layout();fig.savefig(out/'rate_fits.png',dpi=240);fig.savefig(out/'rate_fits.pdf');plt.close(fig)

    fig,axes=plt.subplots(7,5,figsize=(15,16))
    for row,sid in enumerate(SUBSTRATES):
        d=results[sid];r=d['rates']
        for col,model in enumerate(MODELS):
            ax=axes[row,col];z=r[f'{model}_residual']/r.rate_se
            ax.axhline(0,color='0.55',lw=.8);ax.plot(r.t,z,'o-',ms=3,color=COLORS[model])
            if row==0:ax.set_title(model)
            if col==0:ax.set_ylabel(f'{sid}\nResidual / SE')
            if row==6:ax.set_xlabel('Time (input units)')
            style(ax)
    fig.suptitle('Residuals for every candidate; connected points are disjoint local windows')
    fig.tight_layout();fig.savefig(out/'residuals.png',dpi=200);fig.savefig(out/'residuals.pdf');plt.close(fig)

    # Print-size panels: the 35-panel overview above is too dense for an A4 SI.
    for sid in SUBSTRATES:
        d=results[sid];r=d['rates'];m=d['metadata'];scale=m['signal_span']/m['time_span']
        with plt.rc_context({'font.size':8}):
            fig=plt.figure(figsize=(5.7,6.4))
            grid=fig.add_gridspec(3,3,height_ratios=(2.3,1,1))
            top=fig.add_subplot(grid[0,:])
            top.errorbar(r.x*m['signal_span'],r.rate*scale,yerr=r.rate_se*scale,
                         fmt='o',color='black',ms=2.5,capsize=2,label='Rates ± SE')
            xx=np.linspace(max(.001,min(r.x)),max(r.x),200)
            for j,model in enumerate(MODELS):
                top.plot(xx*m['signal_span'],predict_rate(model,d['fits'][model],xx)*scale,
                         color=COLORS[model],label=model,lw=1.3,ls=['--','-','-.',':','--'][j])
            top.set(title=f'{sid}: common rate-scale fits',xlabel='Background-corrected signal',
                    ylabel='Rate (signal / input-time unit)')
            top.legend(ncol=3,frameon=False,fontsize=7);style(top)
            for i,model in enumerate(MODELS):
                ax=fig.add_subplot(grid[1+i//3,i%3])
                ax.axhline(0,color='0.55',lw=.6)
                ax.plot(r.t,r[f'{model}_residual']/r.rate_se,'o-',ms=2,color=COLORS[model],lw=.8)
                ax.set(title=model,xlabel='Input time',ylabel='Residual / SE');style(ax)
            fig.tight_layout(pad=1.2)
            fig.savefig(out/f'fit_residuals_{sid}.png',dpi=240)
            fig.savefig(out/f'fit_residuals_{sid}.pdf');plt.close(fig)

    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for ax,sid in zip(axes.flat,SUBSTRATES):
        d=results[sid];m=d['metadata'];r=d['rates']
        ax.plot(d['t'],d['y'],'.',color='0.35',ms=3,label='Input')
        ax.axhline(m['baseline'],color='0.5',ls='--',label='Fixed tail median')
        for _,block in r.iterrows():
            i,j=int(block.start_row),int(block.end_row)
            tt=d['t'][[i,j]]
            yy=m['baseline']+block.concentration-block.rate_supplied_units*(tt-block.t)
            ax.plot(tt,yy,color=COLORS['S2'],lw=2)
        ax.set(title=sid,xlabel='Time (input units)',ylabel='Supplied signal')
        style(ax)
    axes.flat[-1].axis('off');axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',frameon=False)
    fig.suptitle('Unmodified input values and the fixed local-linear rate extraction')
    fig.tight_layout();fig.savefig(out/'input_and_rate_extraction.png',dpi=240);fig.savefig(out/'input_and_rate_extraction.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=200)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261017)
    args=p.parse_args()
    if args.bootstrap < 0 or args.workers < 1: p.error('Invalid bootstrap/workers')
    if args.out.exists(): p.error('Output already exists; use a NEW directory')
    args.out.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(args.out/'run.log')])
    source_bytes=args.source.read_bytes()
    frame=pd.read_csv(args.source,float_precision='round_trip')
    curves={};source_rows=[]
    for sid in SUBSTRATES:
        pair=frame[[f'time_{sid}',f'S_{sid}']]
        valid=pair.notna().all(axis=1)
        data=pair.loc[valid]
        t,y=data.to_numpy(dtype=float).T
        curves[sid]=(t,y)
        for row,tv,yv in zip(data.index,t,y):
            source_rows.append(dict(substrate=sid,source_csv_row=int(row)+2,time=tv,signal=yv))
    results={};start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(analyze_one,sid,t,y,args.bootstrap,args.seed+i)
                 for i,(sid,(t,y)) in enumerate(curves.items())]
        for future in as_completed(futures):
            sid,d=future.result();results[sid]=d
            d['t'],d['y']=curves[sid]
            elapsed=time.monotonic()-start
            logging.info('%s done (%d/7), elapsed %.1fs ETA %.1fs',sid,len(results),elapsed,elapsed*(7-len(results))/len(results))
    all_stats=pd.concat([results[s]['stats'] for s in SUBSTRATES],ignore_index=True)
    all_stats.to_csv(args.out/'model_comparison.csv',index=False)
    pd.concat([results[s]['rates'] for s in SUBSTRATES],ignore_index=True).to_csv(args.out/'extracted_rates_and_residuals.csv',index=False)
    pd.concat([results[s]['sensitivity'] for s in SUBSTRATES],ignore_index=True).to_csv(args.out/'sensitivity.csv',index=False)
    pd.DataFrame(source_rows).to_csv(args.out/'input_points_map.csv',index=False)
    boots=[];parameters=[];summary=[]
    for sid in SUBSTRATES:
        d=results[sid];b=pd.DataFrame(d['bootstrap']);b['substrate']=sid;boots.append(b)
        for model,popt in d['fits'].items():
            physical=physical_parameters(model,popt,d['metadata'])
            for name,value in zip(PARAM_NAMES[model],physical):
                if args.bootstrap:
                    values=b.loc[b.model==model,name].dropna()
                    low,high=np.quantile(values,[.025,.975])
                else:low=high=np.nan
                parameters.append(dict(substrate=sid,model=model,parameter=name,estimate=value,
                                       bootstrap_low=low,bootstrap_high=high,
                                       uncertainty='conditional block-wild percentile interval'))
        row=dict(substrate=sid)
        for model in MODELS:
            item=d['stats'].set_index('model').loc[model]
            row[f'{model}_R2_weighted']=item.R2_weighted
            row[f'{model}_delta_AICc']=item.delta_AICc
        row['lowest_working_AICc']=d['stats'].sort_values('AICc').iloc[0]['model']
        row['models_with_delta_AICc_lt2']=';'.join(d['stats'].loc[d['stats'].delta_AICc<2,'model'])
        row['n_rate_blocks']=d['metadata']['rate_blocks'];summary.append(row)
    pd.DataFrame(parameters).to_csv(args.out/'parameter_estimates.csv',index=False)
    pd.DataFrame(summary).to_csv(args.out/'Table_S13_reanalysis.csv',index=False)
    if args.bootstrap: pd.concat(boots,ignore_index=True).to_csv(args.out/'bootstrap_parameters.csv',index=False)
    draw_figures(results,args.out)
    manifest=dict(source=str(args.source.resolve()),source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                  input_status='current processed CSV; provenance and physical units require author confirmation',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  primary=asdict(CONFIGS[0]),sensitivity=[asdict(c) for c in CONFIGS],
                  bootstrap=args.bootstrap,seed=args.seed,
                  scipy=scipy.__version__,numpy=np.__version__,pandas=pd.__version__,
                  metadata={s:results[s]['metadata'] for s in SUBSTRATES},
                  bootstrap_config={s:results[s]['bootstrap_config'] for s in SUBSTRATES},
                  bootstrap_failures={s:results[s]['bootstrap_failed'] for s in SUBSTRATES},
                  warnings=['Reduced diagnostic rate families, not a redefinition of training labels.',
                            'Tail background is an assumption; zero/last-point alternatives are reported.',
                            'AICc is a working-likelihood diagnostic, not proof of a mechanism.',
                            'Conditional processed-curve bootstrap cannot restore instrument-level uncertainty.'])
    if args.source.read_bytes()!=source_bytes:raise RuntimeError('Input changed during analysis')
    (args.out/'summary.json').write_text(json.dumps(manifest,indent=2)+'\n')
    logging.info('Finished: %s',args.out)


if __name__=='__main__':
    main()

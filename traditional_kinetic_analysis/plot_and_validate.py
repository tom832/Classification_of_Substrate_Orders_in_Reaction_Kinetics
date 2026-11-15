#!/usr/bin/env python3
"""Validate the complete uniform reanalysis and draw every primary residual.

Methods references (also cited in SI Section 10):
Local-polynomial differentiation: Savitzky and Golay, Anal. Chem. 1964,
36, 1627-1639. DOI: 10.1021/ac60214a047.
Small-sample AIC correction: Hurvich and Tsai, Biometrika 1989,
76, 297-307. DOI: 10.1093/biomet/76.2.297.
Kinetic model discrimination: Taylor et al., React. Chem. Eng. 2021,
6, 1404-1411. DOI: 10.1039/D1RE00098E."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code'))
from kinetic_validation import MODELS,predict_rate
from preprocessing_sensitivity import extract,Method

COLORS=dict(S1='#777777',S2='#e49a50',S3='#6e9bcf',S4='#7ebc93',S5='#d96d79')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=HERE/'results')
    parser.add_argument('--output',type=Path,default=HERE/'all_substrates_fits_and_diagnostics.pdf')
    parser.add_argument('--full_diagnostics',action='store_true',help='include the simulation control and individual residual panels')
    parser.add_argument('--individual_figures',action='store_true',help='also write separate PNG/PDF figures')
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    plan=json.loads((HERE/'analysis_plan.json').read_text());cfg=Method(**plan['primary']).key
    p=args.results;out=args.output.parent/'figures'
    if args.individual_figures:out.mkdir(exist_ok=True)
    data=pd.read_csv(HERE/'input_data_current.csv',float_precision='round_trip')
    summaries=pd.read_csv(p/'configuration_summary.csv')
    allstats=pd.read_csv(p/'all_rate_model_comparisons.csv');stats=allstats[allstats.configuration==cfg]
    rates=pd.read_csv(p/'rates_predictions_residuals.csv');rates=rates[rates.configuration==cfg]
    conc=pd.read_csv(p/'concentration_model_comparison.csv')
    cpoints=pd.read_csv(p/'concentration_predictions_residuals.csv')
    primary=summaries[summaries.primary].set_index('substrate')
    source=HERE/'input_data_current.csv'
    digest=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
    assert digest(HERE/'input_data_current.csv')==plan['source_sha256']
    source_available=source.exists()
    if source_available:assert digest(source)==plan['source_sha256']
    for _,f in allstats.groupby(['substrate','configuration']):
        f=f.set_index('model');assert f.loc['S5','Q']<=f.loc['S2','Q']+1e-8
        assert abs(f.loc['S1','R2_GLS'])<1e-10
    for _,f in conc.groupby(['substrate','error_model']):
        f=f.set_index('model')
        assert f.loc['S5','n2loglik']<=min(f.loc['S2','n2loglik'],f.loc['S3','n2loglik'])+1e-6
    assert len(stats)==35 and len(allstats)==210
    assert summaries.bootstrap_completed.sum()==5600 and summaries.bootstrap_failures.sum()==0
    rows=[];background=[]
    for sid in plan['substrates']:
        f=stats[stats.substrate==sid].set_index('model');s=primary.loc[sid]
        cs=conc[(conc.substrate==sid)&(conc.error_model=='AR1')].set_index('model')
        rows.append(dict(substrate=sid,n_rate=int(s.rate_points),best_rate=f.AICc.idxmin(),
                         S2_R2_GLS=f.loc['S2','R2_GLS'],S5_R2_GLS=f.loc['S5','R2_GLS'],
                         rate_AICc_S5_minus_S2=f.loc['S5','AICc']-f.loc['S2','AICc'],
                         S5_K_boundary=f.loc['S5','K_boundary'],best_concentration_AR1=cs.AICc.idxmin(),
                         AR1_AICc_S5_minus_S2=cs.loc['S5','AICc']-cs.loc['S2','AICc']))
        t,y=data[[f'time_{sid}',f'S_{sid}']].dropna().to_numpy().T
        n=max(5,int(np.ceil(.1*len(y))));slope=np.polyfit(t[-n:]-t[-n:].mean(),y[-n:],1)[0]
        background.append(dict(substrate=sid,n_points=len(y),tail_points=n,tail_median=s.baseline,
                               first_signal=y[0],tail_fraction_of_first=s.baseline/y[0],
                               tail_linear_slope=slope,tail_fitted_change=slope*(t[-1]-t[-n]),
                               note='Descriptive terminal trend only; not an independent background measurement or a hypothesis test.'))
    overview=pd.DataFrame(rows)
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':10})
    pdf=PdfPages(args.output)
    def save(fig,name):
        for ax in fig.axes:ax.tick_params(direction='out',color='black',length=4,width=1)
        if args.individual_figures:
            fig.savefig(out/(name+'.png'),dpi=190,bbox_inches='tight')
            fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
        pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    def grid():return plt.subplots(2,4,figsize=(14,7.5))

    fig,axes=grid()
    for ax,sid in zip(axes.flat,plan['substrates']):
        f=rates[rates.substrate==sid];ss=stats[stats.substrate==sid].set_index('model')
        ax.errorbar(f.x,f.rate,yerr=f.rate_se,fmt='.',color='.45',ms=4,lw=.6,label='Derived rates')
        x=np.linspace(f.x.min(),f.x.max(),250)
        for model in MODELS:
            ax.plot(x,predict_rate(model,json.loads(ss.loc[model,'parameters']),x),color=COLORS[model],
                    lw=1.4,ls='--' if model=='S5' else '-',label=model)
        ax.set(title=f'{sid}: lowest AICc {ss.AICc.idxmin()}',xlabel='Normalized signal x',ylabel='Rate -dx/dtau')
    axes.flat[-1].axis('off');axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',frameon=False)
    fig.suptitle('Uniform SG21/stride7, tail10 background: five S2 + two S5 rankings',fontsize=14)
    fig.text(.5,.015,'Candidate comparison, not independent mechanistic confirmation. Error bars are marginal SDs; full overlap covariance is used in fitting.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95));save(fig,'01_uniform_rate_fits')

    fig,axes=grid()
    for ax,sid in zip(axes.flat,plan['substrates']):
        f=rates[rates.substrate==sid];ax.axhline(0,color='.5',lw=.7)
        for model in MODELS:ax.plot(f.t,f[model+'_whitened_residual'],'o-',ms=3,lw=.8,color=COLORS[model],label=model)
        ax.set(title=sid,xlabel='Time (input units)',ylabel='Whitened rate residual')
    axes.flat[-1].axis('off');axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',frameon=False)
    fig.suptitle('All five candidates: covariance-whitened rate residuals',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95));save(fig,'02_whitened_rate_residuals')

    keys=[Method(**a).key for a in [plan['primary']]+plan['sensitivity']]
    short=['SG21 / tail10','SG21 / tail05','SG21 / tail15','SG21 / zero','SG17 / tail10','SG25 / tail10']
    fig,ax=plt.subplots(figsize=(10,5.5));arr=np.empty((7,6));annotations=[]
    for i,sid in enumerate(plan['substrates']):
        for j,key in enumerate(keys):
            f=allstats[(allstats.substrate==sid)&(allstats.configuration==key)].set_index('model')
            arr[i,j]=f.loc['S5','AICc']-f.loc['S2','AICc']
            annotations.append((i,j,f'{f.AICc.idxmin()}\n{arr[i,j]:+.2f}'))
    ax.imshow(arr,cmap='RdBu_r',vmin=-10,vmax=10,aspect='auto')
    for i,j,s in annotations:ax.text(j,i,s,ha='center',va='center',color='white' if abs(arr[i,j])>7 else 'black')
    ax.set_xticks(range(6));ax.set_xticklabels(short,rotation=15,ha='right');ax.set_yticks(range(7));ax.set_yticklabels(plan['substrates'])
    ax.set_title('All-substrate sensitivity: lowest-AICc model and AICc(S5) - AICc(S2)')
    fig.text(.5,.01,'Every configuration is retained. Background zero changes 1d/1e and 1g/1h rankings.\nCompare models within cells, not absolute AICc across processing methods.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1));save(fig,'03_all_sensitivity_results')

    fig,axes=grid()
    for ax,sid in zip(axes.flat,plan['substrates']):
        t,y=data[[f'time_{sid}',f'S_{sid}']].dropna().to_numpy().T
        ax.plot(t,y,'.',ms=3,color='.6',label='All supplied points')
        ax.axhline(primary.loc[sid,'baseline'],color='.3',lw=.8,ls=':',label='Assumed background')
        for model in ('S2','S3','S5'):
            f=cpoints[(cpoints.substrate==sid)&(cpoints.error_model=='AR1')&(cpoints.model==model)]
            ax.plot(f.t,f.predicted,color=COLORS[model],lw=1.6,ls='--' if model=='S5' else '-',label=model)
        best=overview.set_index('substrate').loc[sid,'best_concentration_AR1']
        ax.set(title=f'{sid}: AR1 minimum {best}',xlabel='Time (input units)',ylabel='Supplied signal')
    axes.flat[-1].axis('off');axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',frameon=False)
    fig.suptitle('Direct concentration check: matched SG window and background, no extra smoothing',fontsize=13)
    fig.text(.5,.015,'AR1 is a working error model. 1h: S2/S5 nearly tied; 1f: modest preference for S5. Curves outside the fitted window are not extrapolated.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95));save(fig,'04_concentration_AR1_check')

    if args.full_diagnostics:
        # Illustrative mathematical control: no noise and no side reaction. Never
        # use AICc on nearly exact noiseless fits as an inferential test.
        t=np.arange(257,dtype=float);true_k=.014;y=.18*np.exp(-true_k*t)
        fig,axes=plt.subplots(1,3,figsize=(13,4.5))
        assumed_B=float(np.median(y[-26:]))
        axes[0].plot(t,y,color=COLORS['S3']);axes[0].axhline(assumed_B,color='.4',ls=':')
        axes[0].set(title='Pure first-order simulation; no side reaction',xlabel='Time (arbitrary units)',ylabel='Simulated signal')
        for ax,baseline in zip(axes[1:],['zero','tail10']):
            settings=dict(plan['primary']);settings['baseline']=baseline
            f,_,m=extract(t,y,Method(**settings));xx=np.linspace(f.x.min(),f.x.max(),200)
            slope=true_k*m['time_span'];offset=slope*m['baseline']/m['signal_span']
            ax.plot(f.x,f.rate,'.',color='.35',label='Same local derivative')
            ax.plot(xx,slope*xx,color=COLORS['S3'],label='First order through origin')
            ax.plot(xx,slope*xx+offset,color=COLORS['S5'],ls='--',label='Offset + first order')
            ax.set(title=f'{baseline}: normalized rate offset {offset:.3f}',xlabel='Normalized signal x',ylabel='Normalized rate')
            ax.legend(frameon=False,fontsize=8)
        fig.suptitle('Known background confounding: residual substrate is not automatically an instrumental blank',fontsize=12)
        fig.text(.5,.01,'Diagnostic simulation only, not experimental data. No model probabilities or AICc inference are assigned to this noiseless control.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,.94));save(fig,'05_background_negative_control')

        # Individual figures avoid hiding differences through overlapping curves.
        for sid in plan['substrates']:
            fig,axes=plt.subplots(2,3,figsize=(12,7))
            f=rates[rates.substrate==sid];ss=stats[stats.substrate==sid].set_index('model')
            for ax,model in zip(axes.flat,MODELS):
                ax.axhline(0,color='.5',lw=.7)
                ax.plot(f.t,f[model+'_residual'],'o-',ms=4,color=COLORS[model],lw=1)
                ax.set(title=f'{model}: delta AICc {ss.loc[model,"delta_AICc"]:.2f}',xlabel='Time (input units)',ylabel='Rate residual')
            ax=axes.flat[-1]
            for model in ('S2','S3','S5'):
                cc=cpoints[(cpoints.substrate==sid)&(cpoints.error_model=='AR1')&(cpoints.model==model)]
                ax.plot(cc.t,cc.innovation,'.-',ms=2,lw=.6,color=COLORS[model],label=model)
            ax.axhline(0,color='.5',lw=.7);ax.set(title='AR1 concentration innovations',xlabel='Time (input units)',ylabel='Signal innovation');ax.legend(frameon=False,fontsize=8)
            fig.suptitle(f'{sid}: complete rate residuals and concentration-error check',fontsize=14)
            fig.tight_layout(rect=(0,0,1,.95));save(fig,'residuals_'+sid)
    pdf.close()
    print(f'PASS: input integrity, nested fits, centered S1 R2; report: {args.output}',flush=True)


if __name__=='__main__':main()

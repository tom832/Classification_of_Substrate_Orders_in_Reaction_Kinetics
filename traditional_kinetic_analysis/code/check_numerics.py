#!/usr/bin/env python3
"""Numerical checks for the point-2 analysis (no data/model files changed).

Methods references (also cited in SI Section 10):
Local-polynomial differentiation: Savitzky and Golay, Anal. Chem. 1964,
36, 1627-1639. DOI: 10.1021/ac60214a047.
Small-sample AIC correction: Hurvich and Tsai, Biometrika 1989,
76, 297-307. DOI: 10.1093/biomet/76.2.297.
Kinetic model discrimination: Taylor et al., React. Chem. Eng. 2021,
6, 1404-1411. DOI: 10.1039/D1RE00098E."""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from kinetic_validation import Config,extract_rates,predict_rate,fit_all
from concentration_check import concentration


def main():
    t=np.linspace(0,1,90);y=.2-.08*t
    rates,meta=extract_rates(t,y,Config(baseline='zero'))
    np.testing.assert_allclose(rates.rate,.4,atol=1e-12)
    np.testing.assert_allclose(rates.rate_supplied_units,.08,atol=1e-12)
    assert set(rates.start_row).isdisjoint(set(rates.end_row))

    x=np.geomspace(.05,1,60);se=np.linspace(.02,.05,len(x))
    for truth,p in [('S1',[1]),('S2',[1.3,.2]),('S3',[.8]),
                    ('S4',[1.2,.3]),('S5',[.7,.18,.9])]:
        y=predict_rate(truth,p,x)
        frame=pd.DataFrame(dict(x=x,rate=y,rate_se=se))
        fits,stats=fit_all(frame,Config())
        np.testing.assert_allclose(predict_rate(truth,fits[truth],x),y,atol=5e-7)
        q=stats.set_index('model').weighted_SSE
        assert q['S5']<=q['S2']+1e-8
        if truth!='S1':
            assert abs(stats.set_index('model').loc['S1','R2_weighted'])<1e-12

    # S4 order increases as concentration decreases: n=1+K/(K+S).
    K=.3
    x=np.array([1.,.1,.01]);n=1+K/(K+x)
    assert np.all(np.diff(n)>0) and np.all((n>1)&(n<2))
    h=1e-5
    finite=(np.log(predict_rate('S4',[1,K],x*np.exp(h)))-
            np.log(predict_rate('S4',[1,K],x*np.exp(-h))))/(2*h)
    np.testing.assert_allclose(finite,n,rtol=1e-8)

    # Full reversible-cycle stationary flux and exact substrate bookkeeping.
    for s in (.01,.1,1):
        k1,km1,k2,km2,k3,C=2.,.3,.5,.4,3.,.01
        a,b,c,d,e=k1*s,km1,k2,km2,k3*s
        weights=np.array([b*d+b*e+c*e,a*d+a*e,a*c]);cat=C*weights/weights.sum()
        net1=a*cat[0]-b*cat[1];net2=c*cat[1]-d*cat[2];J=e*cat[2]
        A=km1*km2;B=(km1+k2)*k3+k1*(km2+k2);D=k1*k3
        J_formula=C*k1*k2*k3*s*s/(A+B*s+D*s*s)
        np.testing.assert_allclose([net1,net2,J],[J_formula]*3)
        np.testing.assert_allclose(net1+J,2*J_formula)

    # Independent integration checks of the implicit closed-form solutions.
    ts=np.linspace(0,.5,60)
    for model,p in [('S1',[1,.3]),('S2',[1,1.2,.2]),('S3',[1,.8]),
                    ('S4',[1,1.1,.4]),('S5',[1,.7,.2,1.2]),
                    ('S5',[1,.7,.2,1e-8]),('S5',[1,.7,.2,0]),
                    ('S5',[1,0,.2,1.])]:
        sol=solve_ivp(lambda t,y:-predict_rate(model,p[1:],y),(0,ts[-1]),[p[0]],
                      t_eval=ts,rtol=1e-11,atol=1e-13)
        np.testing.assert_allclose(concentration(model,np.array(p),ts),sol.y[0],atol=2e-8,rtol=2e-8)
    print('PASS: local slopes, model recovery, S1 baseline, nested S2/S5, S4 direction, stationary flux, integrated laws')


if __name__=='__main__':main()

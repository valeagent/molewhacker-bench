"""Independent deterministic checks; no sampling and no archived-file writes.
Reimplements formulas in truth_mridges.jl, truth_funnel.jl, truth_shell.jl
at bench commit460f594. Python math CDF is independent of Julia's implementation.

Audit copy of 2026-09-26 (experiments/tools/audit_2026-09-26/): identical to
the supplied review script docs/FINAL-DAY-2026-09-26/handoff-package/checks/
v5_grid_checks.py of the thesis repository except for the two path lines below
(repository root derived from the script location; JSON output written to
experiments/out/audit_2026-09-26/ instead of next to the script) and the
recorded Python/NumPy versions. Inputs (read only): experiments/out/tables/
cells.csv and experiments/out/revision_2026-09-25/cells_fresh.csv.
Run from the bench repository root:  python experiments/tools/audit_2026-09-26/v5_grid_checks.py
"""
import csv, json, math, time, sys, platform
from pathlib import Path
import numpy as np

START=time.perf_counter()
ROOT=Path(__file__).resolve().parents[3]          # audit copy: bench repository root
OUT=ROOT/'experiments'/'out'/'audit_2026-09-26'/'v5_grid_checks.json'
OUT.parent.mkdir(parents=True, exist_ok=True)
SQRT2=math.sqrt(2)
def phi_cdf(x):
    return np.fromiter((.5*math.erfc(-float(z)/SQRT2) for z in np.ravel(x)),float).reshape(np.shape(x))
def erf_arr(x):
    return np.fromiter((math.erf(float(z)) for z in np.ravel(x)),float).reshape(np.shape(x))
def erf_source(x):
    t=1/(1+.3275911*x)
    return 1-(((((1.061405429*t-1.453152027)*t+1.421413741)*t-.284496736)*t+.254829592)*t)*np.exp(-x*x)
def integrate_grid(target,d,n,source_erf=True):
    x=np.linspace(0,10,n) if target=='shell' else np.linspace(-10,10,n)
    if target=='mridges':
        p1=(np.exp(-.5*((x-2)/.5)**2)+np.exp(-.5*((x+2)/.5)**2))/(2*.5*math.sqrt(2*math.pi))
        s=.5*np.exp(.2*x); m=1+.3*x*x
        cond=.5*(phi_cdf((10-m)/s)-phi_cdf((-10-m)/s)+phi_cdf((10+m)/s)-phi_cdf((-10+m)/s))
        p=p1*cond**(d-1)
    elif target=='funnel':
        arg=10/np.sqrt(2*np.exp(x))
        f=erf_source(arg) if source_erf else erf_arr(arg)
        p=np.exp(-.5*(x/3)**2)*(2*math.pi)**((d-1)/2)*f**(d-1)
    else:
        p=x**(d-1)*np.exp(-.5*((x-4)/.5)**2)
    cumulative=np.r_[0,np.cumsum(.5*(p[1:]+p[:-1])*np.diff(x))]
    z=cumulative[-1]; c=cumulative/z
    logz=math.log(z)-d*math.log(20)
    if target=='shell': logz+=math.log(2)+d/2*math.log(math.pi)-math.lgamma(d/2)
    return x,c,logz

rows=[]
for target,base in [('mridges',8192),('funnel',4096),('shell',8192)]:
    for d in [2,5,10]:
        results=[integrate_grid(target,d,n) for n in [base,base*2,base*4]]
        entry={'target':target,'d':d,'base_grid':base,'logZ_grid':[r[2] for r in results], 'comparisons':[]}
        for (n1,r1),(n2,r2) in zip(zip([base,base*2],results),zip([base*2,base*4],results[1:])):
            # Sup norm between two piecewise-linear CDFs occurs at their union of knots.
            knots=np.union1d(r1[0],r2[0])
            sup=float(np.max(np.abs(np.interp(knots,*r1[:2])-np.interp(knots,*r2[:2]))))
            ps=np.array([.025,.16,.5,.84,.975])
            dq=np.interp(ps,r2[1],r2[0])-np.interp(ps,r1[1],r1[0])
            entry['comparisons'].append({'n1':n1,'n2':n2,'max_CDF_difference':sup,'delta_logZ':r2[2]-r1[2],'quantile_probs':ps.tolist(),'delta_quantiles':dq.tolist()})
        if target=='funnel':
            exact=integrate_grid(target,d,base*4,False); approx=results[-1]
            entry['source_erf_vs_math_erf']={'delta_logZ_math_minus_source':exact[2]-approx[2],'max_CDF_difference':float(np.max(np.abs(exact[1]-approx[1])))}
        rows.append(entry)

banana=[]
for n in [128,256,512]:
    x,w=np.polynomial.legendre.leggauss(n); x=x*10
    integral=float(np.sum(w*10*np.exp(-x*x/2)/math.sqrt(2*math.pi)*(phi_cdf(10-x*x)-phi_cdf(-10-x*x))))
    banana.append({'nodes':n,'cube_mass_2d':integral})
fresh=list(csv.DictReader(open(ROOT/'experiments/out/revision_2026-09-25/cells_fresh.csv',newline='',encoding='utf-8')))
banana_summary=[]
for d in [2,5,10]:
    exact=d/2*math.log(2*math.pi)+math.log(banana[-1]['cube_mass_2d'])-d*math.log(20)
    rr=[r for r in fresh if r['problem']=='banana' and int(r['d'])==d and r['logZ_fresh']]
    stored=float(rr[0]['truth_logZ'])
    for b in [50000,500000]:
        sub=[r for r in rr if float(r['B'])==b]
        banana_summary.append({'d':d,'B':b,'n':len(sub),'deterministic_logZ':exact,'stored_minus_deterministic_logZ':stored-exact,'median_abs_error_stored':float(np.median([abs(float(r['logZ_fresh'])-stored) for r in sub])),'median_abs_error_deterministic':float(np.median([abs(float(r['logZ_fresh'])-exact) for r in sub]))})
archive=list(csv.DictReader(open(ROOT/'experiments/out/tables/cells.csv',newline='',encoding='utf-8')))
ranking=[]
for s in banana_summary:
    delta=s['stored_minus_deterministic_logZ']
    comparisons={}
    for alg in ['is','ns']:
        rr=[r for r in archive if r['problem']=='banana' and int(r['d'])==s['d'] and float(r['B'])==s['B'] and r['algorithm']==alg and not any(t in r['notes'] for t in ['RHAT-FAIL','BUDGET-VIOLATION','budget-infeasible'])]
        if rr:
            med=float(np.median([float(r['dlogZ']) for r in rr]))
            # cells.csv stores absolute errors, not signed estimates. Triangle
            # inequality bounds each error's change by abs(delta), hence the
            # same bound applies to its median. Do not invent missing signs.
            comparisons[alg]={'n':len(rr),'old_median_abs_error':med,'deterministic_median_lower_bound':max(0,med-abs(delta)),'deterministic_median_upper_bound':med+abs(delta)}
    ranking.append({'d':s['d'],'B':s['B'],'fresh_old':s['median_abs_error_stored'],'fresh_deterministic':s['median_abs_error_deterministic'],'comparators':comparisons,'fresh_wins_old':all(s['median_abs_error_stored']<v['old_median_abs_error'] for v in comparisons.values()),'fresh_win_proved_under_deterministic_reference':all(s['median_abs_error_deterministic']<v['deterministic_median_lower_bound'] for v in comparisons.values())})
result={'scope':'Independent formula/grid calculations, no sampling. Grid-only funnel/shell logZ is a sensitivity diagnostic; production evidence uses adaptive quadrature. SupCDF is exact for the two piecewise-linear interpolants in floating-point arithmetic, not true-CDF error.','grid_checks':rows,'banana_quadrature':banana,'banana_reference_sensitivity':banana_summary,'banana_evidence_rank_sensitivity':ranking,'elapsed_seconds':time.perf_counter()-START,
        'audit_provenance':{'run_date':time.strftime('%Y-%m-%d'),'python':sys.version.split()[0],'numpy':np.__version__,'platform':platform.platform(),'script':str(Path(__file__).resolve().relative_to(ROOT)).replace('\\','/')}}
OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'elapsed_seconds':result['elapsed_seconds'],'banana_evidence_rank_sensitivity':ranking},indent=2))

"""Independent CSV-to-LaTeX verification of the thesis master results table.

Audit copy of 2026-09-26 (experiments/tools/audit_2026-09-26/): identical to
the supplied review script docs/FINAL-DAY-2026-09-26/handoff-package/checks/
v5_table_checks.py of the thesis repository except for the path lines below
(bench root derived from the script location; thesis root overridable by the
first command-line argument; JSON output written to
experiments/out/audit_2026-09-26/) and the recorded Python version and runtime.
Inputs (read only): experiments/out/tables/cells.csv,
experiments/out/revision_2026-09-25/partition_mass.csv and the thesis file
appendices/B-master-table.tex.
Run from the bench repository root:
  python experiments/tools/audit_2026-09-26/v5_table_checks.py [C:/path/to/thesis]
"""
import csv,math,re,json,statistics,sys,time
from pathlib import Path
_START=time.perf_counter()
B=Path(__file__).resolve().parents[3]              # audit copy: bench repository root
T=Path(sys.argv[1]) if len(sys.argv)>1 else Path('C:/MyFiles/master/masterarbeit/masterarbeit_v1')
_OUT=B/'experiments'/'out'/'audit_2026-09-26'/'v5_table_checks.json'
_OUT.parent.mkdir(parents=True, exist_ok=True)
rows=list(csv.DictReader(open(B/'experiments/out/tables/cells.csv',encoding='utf-8',newline='')))
rows=[r for r in rows if not(r['problem']=='eggbox' and int(r['d'])==5 and int(r['seed'])==11)]
parts=list(csv.DictReader(open(B/'experiments/out/revision_2026-09-25/partition_mass.csv',encoding='utf-8',newline='')))
pn={'MVN':'mvn','Banana':'banana','Funnel':'funnel','M-ridges':'mridges','Shell':'shell','Spiky M-ridges':'mridges_spiky','Eggbox':'eggbox'}
an={'IS':'is','MH':'mh','NUTS':'nuts','NS':'ns','MoleWhacker':'mw'}
def adm(r):return not any(x in r['notes'] for x in ['RHAT-FAIL','BUDGET-VIOLATION','budget-infeasible'])
def median(v):
    vv=[float(x) for x in v if x and not math.isnan(float(x))]
    return statistics.median(vv) if vv else math.nan
def fmt(x):
    if math.isnan(x):return '--'
    if x==0:return '0'
    if .01<=abs(x)<1000:return r'\('+format(x,'.3g')+r'\)'
    m,e=format(x,'.2e').split('e');m=m.rstrip('0').rstrip('.')
    return r'\('+m+r'\times 10^{'+str(int(e))+r'}\)'
count=0;values=0;errors=[]
for lineno,line in enumerate((T/'appendices/B-master-table.tex').read_text(encoding='utf-8').splitlines(),1):
    if line.startswith(r'\multicolumn{9}{@{}l}{\textbf{'):
        prob=pn[re.search(r'\\textbf\{([^}]+)\}',line)[1]]
        d=int(re.search(r'd = (\d+)',line)[1]);b=5*10**int(re.search(r'10\^\{(\d+)\}',line)[1])
    elif line.split(' & ')[0] in an:
        tokens=line.removesuffix(r' \\').split(' & ');alg=an[tokens[0]]
        rr=[r for r in rows if r['problem']==prob and int(r['d'])==d and float(r['B'])==b and r['algorithm']==alg]
        aa=[r for r in rr if adm(r)]
        seedtoken=f'{len(aa)}/{len(rr)}'
        if not aa and rr:
            causes=[name for name,tag in [('r','RHAT-FAIL'),('b','BUDGET-VIOLATION'),('i','budget-infeasible')] if any(tag in r['notes'] for r in rr)]
            seedtoken+=r'\textsuperscript{'+','.join(causes or ['?'])+'}'
        vv=[median(r[col] for r in aa) for col in ['eta_Nlike','W1_marginal_avg','SWD']]
        vv+=[median(abs(float(r['dlogZ'])) for r in aa) if alg not in ('mh','nuts') else math.nan]
        vv += [median(r[col] for r in aa) for col in ['QE_p500','QE_p975']]
        vv += [median(r['TV_partition'] for r in parts if r['problem']==prob and int(r['d'])==d and float(r['B'])==b and r['algorithm']==alg and r['admissible']=='true' and r['estimator']!='fresh')]
        expected=[tokens[0],seedtoken]+[fmt(x) for x in vv]
        if tokens!=expected:errors.append({'line':lineno,'cell':[prob,d,b,alg],'actual':tokens,'expected':expected})
        count+=1;values+=len(vv)
result={'master_table_rows_checked':count,'metric_entries_checked':values,'seed_counts_checked':count,'mismatches':errors,'nuts_over_1point2B':[{'problem':r['problem'],'d':r['d'],'B':r['B'],'seed':r['seed'],'cost':r['Nlike_used'],'notes':r['notes']} for r in rows if r['algorithm']=='nuts' and float(r['Nlike_used'])>1.2*float(r['B'])]}
result['audit_provenance']={'run_date':time.strftime('%Y-%m-%d'),'python':sys.version.split()[0],'thesis_table':str(T/'appendices/B-master-table.tex'),'elapsed_seconds':time.perf_counter()-_START}
_OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='nuts_over_1point2B'},indent=2))
print('nuts_over_1point2B entries:',len(result['nuts_over_1point2B']))

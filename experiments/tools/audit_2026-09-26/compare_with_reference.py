"""Compare the reproduced check outputs of 2026-09-26 with the supplied
reference JSONs of the review handoff package.

Inputs (read only):
  experiments/out/audit_2026-09-26/v5_grid_checks.json   (reproduced here)
  experiments/out/audit_2026-09-26/v5_table_checks.json  (reproduced here)
  <thesis>/docs/FINAL-DAY-2026-09-26/handoff-package/checks/v5_grid_checks.json  (reference)
  <thesis>/docs/FINAL-DAY-2026-09-26/handoff-package/checks/v5_table_checks.json (reference)
Output:
  experiments/out/audit_2026-09-26/comparison_report.json

Numbers are compared with a relative tolerance of 1e-9 (absolute 1e-15);
keys that describe the run itself (elapsed_seconds, audit_provenance) are
excluded from the comparison and reported separately. The headline values
named in the final-day brief are additionally checked one by one.
Run from the bench repository root:
  python experiments/tools/audit_2026-09-26/compare_with_reference.py [C:/path/to/thesis]
"""
import json, math, sys, time, platform
from pathlib import Path
import numpy as np

B = Path(__file__).resolve().parents[3]
T = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('C:/MyFiles/master/masterarbeit/masterarbeit_v1')
AUD = B / 'experiments' / 'out' / 'audit_2026-09-26'
REF = T / 'docs' / 'FINAL-DAY-2026-09-26' / 'handoff-package' / 'checks'
SKIP = {'elapsed_seconds', 'audit_provenance'}
RTOL, ATOL = 1e-9, 1e-15


def walk(a, b, path, out):
    """Collect every leaf difference between the two JSON trees."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in SKIP:
                continue
            if k not in a or k not in b:
                out.append({'path': path + '/' + k, 'issue': 'missing in ' + ('reproduction' if k not in a else 'reference')})
                continue
            walk(a[k], b[k], path + '/' + k, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append({'path': path, 'issue': f'length {len(a)} vs {len(b)}'})
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f'{path}[{i}]', out)
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
        if a != b:
            out.append({'path': path, 'issue': f'{a!r} vs {b!r}'})
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL):
            out.append({'path': path, 'reproduced': a, 'reference': b, 'rel_diff': abs(a - b) / max(abs(b), 1e-300)})
    else:
        if a != b:
            out.append({'path': path, 'issue': f'{a!r} vs {b!r}'})


def compare(name):
    rep = json.loads((AUD / name).read_text(encoding='utf-8'))
    ref = json.loads((REF / name).read_text(encoding='utf-8'))
    diffs = []
    walk(rep, ref, '', diffs)
    return rep, ref, diffs


grid_rep, grid_ref, grid_diffs = compare('v5_grid_checks.json')
tab_rep, tab_ref, tab_diffs = compare('v5_table_checks.json')

# Headline values named in the final-day brief (reproduced values, printed
# for the record next to the brief's rounded targets).
first = {r['target']: r['comparisons'][0]['max_CDF_difference'] for r in grid_rep['grid_checks'] if r['d'] == 10}
firstmax = {}
for r in grid_rep['grid_checks']:
    firstmax[r['target']] = max(firstmax.get(r['target'], 0.0), r['comparisons'][0]['max_CDF_difference'])
secondmax = {}
for r in grid_rep['grid_checks']:
    secondmax[r['target']] = max(secondmax.get(r['target'], 0.0), r['comparisons'][1]['max_CDF_difference'])
qmax = max(abs(q) for r in grid_rep['grid_checks'] for c in r['comparisons'][:1] for q in c['delta_quantiles'])
det = {s['d']: s['deterministic_logZ'] for s in grid_rep['banana_reference_sensitivity']}
off = {s['d']: s['stored_minus_deterministic_logZ'] for s in grid_rep['banana_reference_sensitivity']}
med = {(s['d'], s['B']): (s['median_abs_error_stored'], s['median_abs_error_deterministic']) for s in grid_rep['banana_reference_sensitivity']}
headline = {
    'first_doubling_max_CDF_difference_over_d': firstmax,
    'second_doubling_max_CDF_difference_over_d': secondmax,
    'largest_first_doubling_quantile_shift': qmax,
    'banana_deterministic_logZ': det,
    'banana_stored_minus_deterministic': off,
    'banana_fresh_median_abs_error_stored_vs_deterministic': {f'd{d}_B{B}': v for (d, B), v in med.items()},
    'banana_fresh_wins_under_deterministic_reference': [r['fresh_win_proved_under_deterministic_reference'] for r in grid_rep['banana_evidence_rank_sensitivity']],
    'gauss_legendre_cube_mass_2d': {b['nodes']: b['cube_mass_2d'] for b in grid_rep['banana_quadrature']},
    'gauss_legendre_256_vs_512': abs(grid_rep['banana_quadrature'][1]['cube_mass_2d'] - grid_rep['banana_quadrature'][2]['cube_mass_2d']),
    'table_rows_checked': tab_rep['master_table_rows_checked'],
    'table_metric_entries_checked': tab_rep['metric_entries_checked'],
    'table_seed_counts_checked': tab_rep['seed_counts_checked'],
    'table_mismatches': len(tab_rep['mismatches']),
    'table_nuts_over_1point2B_entries': len(tab_rep['nuts_over_1point2B']),
}
expected = {
    'first_doubling_max_CDF_difference_over_d': {'mridges': 4.903e-7, 'funnel': 3.210e-7, 'shell': 2.705e-7},
    'banana_deterministic_logZ': {2: -4.155408138371801, 5: -10.385789359419755, 10: -20.769758061166346},
    'banana_stored_minus_deterministic': {2: -7.614e-5, 5: -4.608e-5, 10: 4.909e-5},
    'banana_fresh_median_d5_B500000': (0.0008029036, 0.0008418278),
    'banana_fresh_median_d10_B500000': (0.0010999288, 0.0010671229),
    'table': (225, 1575, 225, 0),
}
brief_checks = {
    'first_doubling_CDF_rounded_ok': all(abs(firstmax[k] - v) / v < 5e-4 for k, v in expected['first_doubling_max_CDF_difference_over_d'].items()),
    'deterministic_logZ_exact_ok': all(math.isclose(det[k], v, rel_tol=1e-12) for k, v in expected['banana_deterministic_logZ'].items()),
    'offsets_rounded_ok': all(abs(off[k] - v) / abs(v) < 5e-4 for k, v in expected['banana_stored_minus_deterministic'].items()),
    'fresh_medians_rounded_ok': all(abs(med[(5, 500000)][i] - expected['banana_fresh_median_d5_B500000'][i]) < 5e-11 for i in (0, 1))
        and all(abs(med[(10, 500000)][i] - expected['banana_fresh_median_d10_B500000'][i]) < 5e-11 for i in (0, 1)),
    'all_six_fresh_wins_survive': all(headline['banana_fresh_wins_under_deterministic_reference']) and len(headline['banana_fresh_wins_under_deterministic_reference']) == 6,
    'table_counts_ok': (headline['table_rows_checked'], headline['table_metric_entries_checked'], headline['table_seed_counts_checked'], headline['table_mismatches']) == expected['table'],
}

report = {
    'run_date': time.strftime('%Y-%m-%d %H:%M:%S'),
    'python': sys.version.split()[0], 'numpy': np.__version__, 'platform': platform.platform(),
    'tolerance': {'rel': RTOL, 'abs': ATOL},
    'grid_checks': {'differences': grid_diffs, 'n_differences': len(grid_diffs),
                    'elapsed_seconds_reproduced': grid_rep.get('elapsed_seconds'), 'elapsed_seconds_reference': grid_ref.get('elapsed_seconds')},
    'table_checks': {'differences': tab_diffs, 'n_differences': len(tab_diffs),
                     'elapsed_seconds_reproduced': tab_rep.get('audit_provenance', {}).get('elapsed_seconds')},
    'headline_values_reproduced': headline,
    'brief_targets_met': brief_checks,
    'all_brief_targets_met': all(brief_checks.values()),
    'byte_identical_apart_from_runtime_keys': len(grid_diffs) == 0 and len(tab_diffs) == 0,
}
(AUD / 'comparison_report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k not in ('headline_values_reproduced',)}, indent=2, default=str))
print(json.dumps(headline, indent=2, default=str))

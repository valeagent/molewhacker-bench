# Audit 2026-09-26 (final-day benchmark writer)

Read-only reproduction of the v5 annex checks and a caption-only regeneration
of the master results table. Nothing under `experiments/out/**` other than
`experiments/out/audit_2026-09-26/` was written. No sampling, campaign or
truth-regeneration code was executed.

Environment: Windows 11 (10.0.26200), Python 3.13.1, NumPy 2.1.3, Julia 1.11.6.
Repository state: `main`, HEAD `460f594` (tag `v5-frozen-2026-09-26`).

## Files

| File | Purpose |
|---|---|
| `v5_grid_checks.py` | Copy of `handoff-package/checks/v5_grid_checks.py`; only the paths were changed (repo-relative `ROOT`, output to `experiments/out/audit_2026-09-26/v5_grid_checks.json`) and an `audit_provenance` block was added. Grid-doubling of the M-ridges / funnel / shell inverse-CDF grids and a deterministic Gauss-Legendre banana cube fraction. |
| `v5_table_checks.py` | Copy of `handoff-package/checks/v5_table_checks.py`; paths only (thesis path as optional argv[1]) plus `audit_provenance`. Recomputes every master-table cell from `experiments/out/revision_2026-09-25/cells.csv` and `partition_mass.csv` and compares with `appendices/B-master-table.tex`. |
| `compare_with_reference.py` | Recursive JSON comparison (rel 1e-9, abs 1e-15) of the two reproduced outputs against the supplied reference JSONs; writes `experiments/out/audit_2026-09-26/comparison_report.json`. |
| `regenerate_master_table.jl` | Loads `experiments/tools/table_master_results.jl` via `include_string` with the output path redirected to `experiments/out/audit_2026-09-26/master_results_table.tex`, so the archived `experiments/out/tables/master_results_table.tex` is not touched. |

## Commands and runtimes (26 September 2026)

```powershell
cd C:\MyFiles\master\02_molewhacker\molewhacker-bench-repo
python experiments\tools\audit_2026-09-26\v5_grid_checks.py        # 0.48 s numeric, 1.26 s wall
python experiments\tools\audit_2026-09-26\v5_table_checks.py       # 0.20 s numeric, 0.38 s wall
python experiments\tools\audit_2026-09-26\compare_with_reference.py
julia --project=experiments experiments\tools\audit_2026-09-26\regenerate_master_table.jl
#   environment load 122.4 s, table generation 23.5 s, 156 s wall
```

## Results

* Table checks: 225 rows, 1575 metric entries, 225 seed labels, 0 mismatches;
  0 differences against the reference JSON.
* Grid checks: first-doubling max |dCDF| 4.903e-7 (M-ridges), 3.210e-7 (funnel),
  2.705e-7 (shell); banana deterministic log Z -4.155408138371801 (d=2),
  -10.385789359419755 (d=5), -20.769758061166346 (d=10); stored minus
  deterministic -7.614e-5, -4.608e-5, +4.909e-5; 0 differences against the
  reference JSON. Gauss-Legendre 128/256/512 cube masses agree with the
  reference to the last printed digit (256 vs 512 differ by 3.1e-15 here,
  3.4e-15 in the reference; the thesis quotes "within 4e-15").
* Master table regeneration: the regenerated file differs from the frozen
  `appendices/B-master-table.tex` only in the caption line (line 7); all 225
  data rows are byte-identical; `\setlength{\tabcolsep}{2.8pt}` preserved.

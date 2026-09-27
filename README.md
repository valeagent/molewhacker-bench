# molewhacker-bench

> **Public companion repository for the master's thesis
> "Importance Sampling Methods in the Bayesian Analysis Toolkit"**
> (Valentin Reindel, Technical University of Munich, Department of
> Physics, 2026).

This repository contains the complete benchmark suite, the exact
algorithm implementations, the canonical per-run results table of the
thesis, the LaTeX table fragments and every figure of the thesis
benchmark chapters, so that each numerical claim of the thesis can be
traced to code and data. It is a read-only archive: the code and the
result files are frozen at the state that produced the thesis numbers.

**Final state.** The last commit that changed code or result files is
`03d94d0f92682b062df6f3a3fd7546498025acc3` (27 September 2026, tagged
`v10-final-2026-09-27`). Every later commit is documentation only
(this file, `THESIS-FIGURES.md`); the thesis tag
`v11-final-2026-09-27` points at that documentation commit.
Reproduce from a checkout of a tagged commit with its pinned
`Manifest.toml` (Julia 1.11.6).

## At a glance

* **7 benchmark problems**: MVN, banana, funnel, M-ridges, shell,
  spiky M-ridges, eggbox — synthetic targets that isolate the
  geometric pathologies of physics posteriors (correlation, curvature,
  scale hierarchy, isolated modes, thin manifolds, narrow spikes,
  periodic mode proliferation). Every target has a semi-analytical
  ground truth.
* **5 algorithms**: plain importance sampling (IS), random-walk
  Metropolis–Hastings (MH), the No-U-Turn Sampler (NUTS, via
  BAT.jl/AdvancedHMC), ellipsoidal nested sampling (NS), and the
  thesis's adaptive importance sampler **MoleWhacker** (MW).
* **One cost convention**: every algorithm is charged per
  likelihood-equivalent evaluation by the wrapped counter of
  `experiments/src/counter.jl` — primal calls cost 1, a
  `d`-dimensional forward-mode gradient costs `d`, a Hessian ≈ `d²`
  (see `experiments/docs/NUTS-COST-ACCOUNTING.md` for the recorded
  NUTS leapfrog charge). The final MoleWhacker sample (below) is
  charged one unit per independent draw on top of the adaptation cost.
* **3,975 runs**: (algorithm, problem, dimension, budget, seed)
  quintuples — budgets `B ∈ {5e3, 5e4, 5e5}`, dimensions
  `d ∈ {2, 5, 10}` on the four scaling targets, up to 20 seeds per
  cell (10 for spiky M-ridges/eggbox, 5 for shell).
* **The MoleWhacker estimator of the thesis** is the *independent
  final sample*: after adaptation the final mixture is frozen and
  `N_fresh = max(0, floor(B − C_adapt))` independent points are drawn
  from it and weighted by the exact target (thesis Secs. 5.7 and 8.3).
  489 of the 795 archived MW runs leave budget for such a sample
  (28 of 45 cells); the 306 runs whose adaptation alone reached the
  budget have *no* estimate. The accumulated adaptation population
  that the original campaign stored is used only for the labelled
  adaptation diagnostics.

## What is where

| Product | Location | State |
|---|---|---|
| **Canonical primary results (27 Sep 2026)** — `primary_cells.csv` (3,975 rows, `cells.csv` schema plus provenance columns; the 795 MW rows carry the final-sample values or are marked unavailable), `primary_cell_medians.csv`, `primary_manifest.csv`, `primary_checks.json` (35 assertions, all passed) | `experiments/out/primary_fresh_2026-09-27/` | in Git |
| **Primary figures** — 20 summary heatmaps, 2 dimension grids, 2 partition-mass figures, 9 MW triangle plots (PDF + PNG), `heatmap_cell_values.csv`, `triangle_manifest.csv` | `experiments/out/primary_fresh_2026-09-27/figs/` | in Git |
| **Primary LaTeX table fragments** — the 9 `\input` files of thesis Ch. 8 and App. A (`B-master-table.tex`, `B-fresh-headline.tex`, `B-partition-headline.tex`, `B-quantile-headline.tex`, `B-pass-fractions.tex`, `B-wall-times.tex`, `ch8-headline-*.tex`) | `experiments/out/primary_fresh_2026-09-27/tables/` | in Git |
| **Per-run numerical summaries of the final samples (25 Sep 2026)** — `cells_fresh.csv` (795 rows: status, `N_fresh`, `C_total`, `Neff`, efficiencies, Pareto `k`, W1, SWD, quantile errors, `dlogZ`, phase wall times), `fresh_manifest.csv` (source-file SHA-256, proposal SHA-256, RNG seed, counts, costs), the partition-mass files `partition_reference.csv`, `partition_calibration.csv`, `partition_mass.csv`, `partition_tests.csv`, the `fresh_vs_population.tex` diagnostic table (thesis Tab. A.1) | `experiments/out/revision_2026-09-25/` | in Git |
| **Derived final-sample payloads** — one `.jld2` per MW run (795 files, 228,081,633 bytes): summary values, a 10,000-point equal-weight display resample, top weights and weight quantiles. Not a dump of the 119,859,172 native draws. | `experiments/out/revision_2026-09-25/fresh_runs/` | **not in Git, not on Zenodo** — reconstructible, see route B |
| **Historical products of the original campaign** — `experiments/out/tables/cells.csv` (one row per run; the MW rows are the accumulated adaptation population), `headline_tests.csv`, `headline_medians.csv`, `mw_iteration_summary.csv` (thesis Tab. 8.3), `master_results_table.tex`, and the 420 PDF + 420 PNG figures of `experiments/out/figs/` | `experiments/out/tables/`, `experiments/out/figs/` | in Git; **historical** — the MW rows/panels show the adaptation population, not the thesis estimator. The thesis still embeds the adaptation *diagnostics* from `out/figs/` (Figs. 5.1–5.4, 6.1–6.4, 7.1–7.7, A.25–A.45); its summary heatmaps, dimension grids, partition figures and MW triangles come from `primary_fresh_2026-09-27/figs/`. |
| **Raw output of the original benchmark grid** — 3,975 run directories with `result.h5` (samples, weights, diagnostics, MW iteration logs and the stored frozen mixture in `extras`), the truth reference sets, the instrumented storyboard runs; ≈ 23 GB | Zenodo, [**DOI 10.5281/zenodo.22228405**](https://doi.org/10.5281/zenodo.22228405) (published 1 September 2026) | The record holds the *original* grid only. It contains no product of the 25/27 September analyses; those live in this repository (CSV, tables, figures) or are reconstructible from the archived mixtures (payloads). |

`out/figs/` and `primary_fresh_2026-09-27/figs/` contain files with
**identical names and different content** (for example
`summary__all__d5__B5e5__heatmap-eta.pdf`: adaptation population in
the former, final sample in the latter). Copy thesis assets only from
the directory named in `THESIS-FIGURES.md`.

## Repository layout

```
molewhacker-bench/
├── README.md                  <- this file
├── LICENSE                    <- MIT
├── CITATION.cff               <- citation metadata
├── THESIS-FIGURES.md          <- thesis figure/table -> file -> generator -> data map
├── Project.toml               <- Julia environment (single env at the root)
├── Manifest.toml              <- exact pinned package versions (Julia 1.11.6)
├── src/
│   └── MoleWhacker.jl         <- the MoleWhacker algorithm module (exact
│                                 version used by every benchmark run)
└── experiments/
    ├── src/                   <- benchmark framework
    │   ├── ExperimentsBase.jl <- module entry point
    │   ├── counter.jl         <- the likelihood-evaluation cost counter
    │   ├── base.jl            <- grids, shared types, helpers (save/load of MethodResult incl. the mixture)
    │   ├── metrics.jl         <- W1, SWD, Neff, dlogZ, quantile errors, ...
    │   ├── plotting.jl        <- the figure factory (CairoMakie)
    │   ├── problems/          <- the 7 benchmark targets
    │   ├── truths/            <- semi-analytical ground-truth builders
    │   └── algorithms/        <- the 5 algorithm runners
    ├── scripts/               <- the numbered pipeline of the original campaign
    ├── tools/                 <- final-stage estimator, primary view, primary figures/tables, historical figure passes, QA gates
    ├── tests/runtests.jl      <- regression tests (truth, counter, sanity)
    ├── docs/                  <- NUTS cost-accounting methodology note
    └── out/
        ├── primary_fresh_2026-09-27/  <- canonical primary view, primary figures and table fragments
        ├── revision_2026-09-25/       <- per-run final-sample summaries, partition-mass files (payloads: not in Git)
        ├── tables/                    <- historical: cells.csv + statistical-test tables of the original campaign
        └── figs/                      <- historical: 420 PDF + 420 PNG figures of the original campaign
```

## Reproducing the thesis results

All commands run from the repository root of a checkout of a tagged
commit. Use a disposable checkout: the generators write into
`experiments/out/`. Nothing below re-runs the benchmark grid; the two
routes differ in whether new *target evaluations* take place.

### 0. Requirements

* Julia **1.11.x** (the Manifest pins 1.11.6); Python 3.9+ with
  `numpy` and `pandas` for the LaTeX table script.
* Route A needs no data download. Route B needs the runs and truth
  archives of the Zenodo record (≈ 23 GB unpacked, ≈ 16 GB RAM).

### 1. Install the exact environment

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -t auto experiments/tests/runtests.jl
```

### 2. Route A — primary tables and summary figures from the shipped CSVs (no payloads, no target evaluations)

The canonical primary view `primary_cells.csv` and every primary
figure and table fragment are already in the repository after
cloning; inspecting them needs no computation. To regenerate the
summary figures and the table fragments from the committed CSVs:

```sh
julia --project=. -t 8 experiments/tools/figs_primary.jl
julia --project=.      experiments/tools/table_master_primary.jl
python experiments/tools/revision_tables_primary.py
```

* `figs_primary.jl` rebuilds the 20 summary heatmaps, the 2 dimension
  grids and the 2 partition-mass figures (24 PDF + 24 PNG) into
  `primary_fresh_2026-09-27/figs/` from `primary_cells.csv` and the
  saved partition CSVs, and asserts every inscribed value against
  `primary_cell_medians.csv` (`--only-offdim` restricts it to the six
  `d = 2` / `d = 10` maps).
* `table_master_primary.jl` writes `tables/B-master-table.tex` (thesis
  Tab. A.4); `revision_tables_primary.py` writes the other eight
  fragments and checks its medians against `primary_cell_medians.csv`.

Do **not** put `primary_fresh_view.jl` in front of these commands:
it rebuilds the primary view from `cells.csv`, `cells_fresh.csv` and
`partition_mass.csv` but requires the presence of all 795 payload
files (an existence check, line 215), which a fresh clone does not
have. The shipped `primary_cells.csv` is its verified output
(`primary_checks.json`). The nine MW triangle plots are not
regenerated by route A; their PDFs are published as they are and
their provenance is in `figs/triangle_manifest.csv`.

### 3. Route B — reconstruct the final-sample payloads, the primary view and the triangle plots (target evaluations, no adaptation)

Download the Zenodo record, reassemble and verify the runs archive as
described in its `DATA-README.md`, and unpack the runs and truth
archives into `experiments/out/` (`runs/` and `truth/`; the storyboard
archive is not needed for these products). Use the **archived truth
files**: the spiky M-ridges reference sampler was revised after the
shipped reference was drawn, so a regenerated reference is
statistically equivalent but not bit-identical (see "Verification
status" below). Then:

```sh
julia --project=. -t 8 experiments/tools/fresh_draw_recompute.jl --subset all --aggregate
julia --project=. -t 8 experiments/tools/primary_fresh_view.jl
julia --project=. -t 8 experiments/tools/figs_triangles_primary.jl
```

followed by the route A commands if a complete regeneration is wanted.

* `fresh_draw_recompute.jl` loads the frozen final mixture of every
  archived MW run (`extras[:mixture]` of `result.h5`), seeds
  `Xoshiro(rng_seed_for(problem, d, B, seed))`, draws the full
  remaining budget and evaluates the exact target on the draws. This
  **does perform target evaluations** — 119,859,172 in total over the
  489 runs with remaining budget — but does not repeat the adaptation
  or any comparator run. `--subset` accepts `headline`, `top`, `mid`,
  `low`, `all`; `--aggregate` writes `cells_fresh.csv` and
  `fresh_manifest.csv`; existing payloads are resumed. Do not use
  `--nfresh-cap` for the final full-remaining-budget results.
* Recorded phase timings in `cells_fresh.csv` (sum over the 489 runs:
  756.1 s draw + evaluation, 173.1 s metrics; 8 threads, 25 September
  2026) are provenance of the original execution, not a promised
  runtime. Regenerated CSVs carry new date/host/wall-time metadata and
  are therefore not byte-identical; the estimator values are
  reproducible from the recorded seeds and mixtures.
* `figs_triangles_primary.jl` needs the payloads, `runs/` (SHA-256
  check of the source file) and `truth/`; it draws nothing.
* Optional validation: `partition_mass.jl --compute --with-fresh
  --stats` recomputes the native partition-mass errors, replaying the
  fresh draws. The saved partition CSVs are sufficient for every
  published figure and table.

### 4. Historical pipeline of the original campaign

The numbered scripts (`00_generate_truth.jl`, `04_run_all.jl`,
`02_aggregate.jl` → `cells.csv`, `02b_tests.jl`, `03_plots.jl`) and the
figure passes `figs_headline.jl`, `figs_calibration.jl`,
`figs_heatmap_angles.jl`, `figs_triangles.jl`, `figs_recovery.jl`
regenerate the **adaptation-population presentation** of MoleWhacker
into `experiments/out/figs/` and `experiments/out/tables/`. They are
retained as engineering history and for the adaptation diagnostics the
thesis embeds (`THESIS-FIGURES.md` lists which generators those are).
Running them does not affect `primary_fresh_2026-09-27/`, but their
outputs in `out/figs/` share file names with the primary figures (see
above). `04_run_all.jl` re-runs the whole grid (~24 h) and is not
needed for any product of this repository. Quality gates:
`experiments/tools/sanity_check.jl` and `experiments/tools/check_figs.jl`
(historical figure set).

## Verification status

* **Final analyses (25/27 September 2026).** `primary_checks.json`
  records the 35 assertions of the primary view (row counts, 489/306
  eligibility split, `C_total = B` for every available run, headline
  medians and win counts, equality of the 6,846 eligible values with
  `cells_fresh.csv`), all passed at commit `03d94d0`.
  `figs/heatmap_cell_values.csv` lists every inscribed heatmap value
  next to the primary medians (asserted equal by `figs_primary.jl`);
  `figs/triangle_manifest.csv` records the payload, source-file
  SHA-256 and representative seed of each triangle. An external review
  (27 September 2026) re-checked the 225 master-table rows and the nine
  triangle provenance chains against these files.
* **Original campaign (initial release, historical).** The initial
  release tree was verified end to end before publication:
  `Pkg.instantiate()` from the pinned Manifest plus the full
  regression suite pass (truth self-consistency 42/42, truth vs.
  analytic density 10/10, algorithm and counter sanity sets); ground
  truth regenerated from scratch matches the archived truth sets in
  every data field bit-for-bit for all deterministic constructions;
  the population-based headline heatmaps, dimension grid, recovery
  curves and triangle plots regenerated inside the tree from the
  shipped `cells.csv`, run data and truth were byte-identical in 22 of
  24 PNG outputs (PDF twins differ only in creation metadata). The two
  exceptions are the spiky M-ridges triangles: the shipped reference
  files were drawn by an earlier revision of the seeded truth sampler
  with a different consumption order of the same stream, so
  regeneration yields a **statistically equivalent, not bit-identical**
  reference for this one target (the analytic log-evidence is
  bit-identical; sample moments agree at the Monte-Carlo noise level
  of the 5×10⁴-draw reference). For bit-exact reproduction use the
  archived truth files from the Zenodo record.

## Relation to the thesis

The benchmark contract — problem specifications, ground-truth
derivations, metric definitions, protocol, and all conclusions — is
specified in the thesis (Ch. 7 "Benchmark Design and Test Problems",
Ch. 8 "Results", App. A "Supplementary Benchmark Results", App. C
implementation listings). Source-code listings of every target,
algorithm invocation, metric helper, the grid runner and the
final-stage estimator are reproduced in the thesis's implementation
appendix and correspond line-for-line to the files in this repository.
Comments of the form `Protocol §n` refer to the benchmark protocol as
specified in thesis Ch. 7; comments tagged `V2-FIX-*` … `V8-FIX-*`
record internal revision milestones of the benchmark engine and are
kept as engineering history.

## Revision of 25 September 2026 — final-sample estimator

Following an external review of the thesis, two post-campaign
analyses were added. Neither modifies the archived runs, truth sets,
`cells.csv` or the historical figures; all outputs live under
`experiments/out/revision_2026-09-25/`.

* **Frozen-mixture final-sample estimator**
  (`experiments/tools/fresh_draw_recompute.jl`). The archived
  MoleWhacker output is the accumulated population of all whacking
  iterations weighted against the final mixture, which is not a valid
  importance sampling estimator in general (thesis Sec. 5.7.1). For
  every one of the 795 archived MoleWhacker runs the tool loads the
  stored final mixture, draws `N_fresh = max(0, floor(B − C_adapt))`
  independent points (the budget the archived run left unspent),
  weights them by the exact target, and evaluates the archived metric
  pipeline on the weighted final sample. `cells_fresh.csv` holds one
  row per run; `fresh_manifest.csv` records provenance (source-file
  hash, proposal hash, RNG seed, counts, costs). `--test` runs the
  estimator checks (constant ratio, Gaussian target, the
  two-component counterexample, transform and full-mixture identities
  against BAT and Distributions, thread-count independence). The
  per-run `.jld2` payloads are not committed and not part of the
  Zenodo record; route B reconstructs them.
* **Partition-mass error** (`experiments/tools/partition_mass.jl`),
  which replaces the retired centroid-clustering mode-recovery rate
  (still present in `cells.csv` as `mode_recovery`; on independent
  draws from the targets that metric returns zero in five and ten
  dimensions). `TV_partition` is the total-variation distance between
  the weighted output's and the true probabilities of a fixed
  partition: the 2^d coordinate-sign orthants of M-ridges (reference
  probabilities by quadrature of the truncated theta_1 marginal) and
  the seven theta_1 intervals of spiky M-ridges bounded by the six
  density minima (Gaussian-CDF reference probabilities). Outputs:
  `partition_reference.csv`, `partition_calibration.csv`,
  `partition_mass.csv` (every archived multimodal cell with
  population/output weights, plus the MoleWhacker final samples),
  `partition_tests.csv` (two-sided seed-paired Wilcoxon and Cliff's
  delta at the headline setting).
* `experiments/tools/revision_tables.py` emits the diagnostic table
  `fresh_vs_population.tex` (thesis Tab. A.1, archived population
  against final sample).

## Revision of 27 September 2026 — canonical primary view

The thesis reports MoleWhacker exclusively through the final sample.
`experiments/tools/primary_fresh_view.jl` joins `cells.csv` (the
comparators and the MW adaptation records), `cells_fresh.csv` and
`partition_mass.csv` into `primary_fresh_2026-09-27/primary_cells.csv`
(comparator rows verbatim with estimator `output`; MW rows with the
final-sample values, `Nlike_used = C_total`, `terminated_by =
final_sample`, or marked `NO-FINAL-SAMPLE` with no metric value) and
`primary_cell_medians.csv`, and writes `primary_checks.json`.
`figs_primary.jl`, `figs_triangles_primary.jl`,
`table_master_primary.jl` and `revision_tables_primary.py` generate
every primary figure and table fragment from that view with the
existing thesis file names. No sampling and no target evaluation take
place in these four tools. The six `d = 2` / `d = 10` heatmaps were
regenerated at `03d94d0` with rows restricted to the targets present
in the grid at that dimension (`--only-offdim`; inscribed values
unchanged).

## License

MIT — see `LICENSE`. Every Julia source file carries an SPDX header.
The Zenodo data record is CC-BY-4.0.

## Citation

```bibtex
@mastersthesis{reindel2026molewhacker,
  author = {Valentin Reindel},
  title  = {Importance Sampling Methods in the Bayesian Analysis Toolkit},
  school = {Technical University of Munich, Department of Physics},
  year   = {2026},
  note   = {Companion code: \url{https://github.com/valeagent/molewhacker-bench}
            (final analysis commit 03d94d0);
            data archive of the original benchmark grid:
            \url{https://doi.org/10.5281/zenodo.22228405}}
}
```

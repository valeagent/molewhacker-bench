# molewhacker-bench

> **Public companion repository for the master's thesis
> "Importance Sampling Methods in the Bayesian Analysis Toolkit"**
> (Valentin Reindel, Technical University of Munich, Department of
> Physics, 2026).

This repository contains the complete benchmark suite, the exact
algorithm implementations, the full per-cell results table, and every
figure of the thesis benchmark chapters, so that each numerical claim
of the thesis can be traced to code and data and reproduced from
scratch. It is a read-only archive: the code is frozen at the state
that produced the thesis numbers.

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
* **One cost axis**: every algorithm is charged per
  likelihood-equivalent evaluation through a single wrapped counter —
  primal calls cost 1, a `d`-dimensional forward-mode gradient costs
  `d`, a Hessian ≈ `d²` (see
  `experiments/docs/NUTS-COST-ACCOUNTING.md` for the exact NUTS
  leapfrog charge).
* **3,975 runs**: (algorithm, problem, dimension, budget, seed)
  quintuples — budgets `B ∈ {5e3, 5e4, 5e5}`, dimensions
  `d ∈ {2, 5, 10}` on the four scaling targets, up to 20 seeds per
  cell (10 for spiky M-ridges/eggbox, 5 for shell).
* **Shipped results**: the complete per-cell metric table
  (`experiments/out/tables/cells.csv`, one row per run), the
  statistical-test tables, and 420 publication-quality figures
  (PDF + PNG) including every figure embedded in the thesis.

## Repository layout

```
molewhacker-bench/
├── README.md                  <- this file
├── LICENSE                    <- MIT
├── CITATION.cff               <- citation metadata
├── THESIS-FIGURES.md          <- thesis figure -> file -> script audit map
├── Project.toml               <- Julia environment (single env at the root)
├── Manifest.toml              <- exact pinned package versions (Julia 1.11.6)
├── src/
│   └── MoleWhacker.jl         <- the MoleWhacker algorithm module (exact
│                                 version used by every benchmark run)
└── experiments/
    ├── src/                   <- benchmark framework
    │   ├── ExperimentsBase.jl <- module entry point
    │   ├── counter.jl         <- the likelihood-evaluation cost counter
    │   ├── base.jl            <- grids, shared types, helpers
    │   ├── metrics.jl         <- W1, SWD, Neff, dlogZ, recovery, ...
    │   ├── plotting.jl        <- the figure factory (CairoMakie)
    │   ├── problems/          <- the 7 benchmark targets
    │   ├── truths/            <- semi-analytical ground-truth builders
    │   └── algorithms/        <- the 5 algorithm runners
    ├── scripts/               <- the numbered reproduction pipeline
    ├── tools/                 <- thesis-figure regeneration + QA gates
    ├── tests/runtests.jl      <- regression tests (truth, counter, sanity)
    ├── docs/                  <- NUTS cost-accounting methodology note
    └── out/
        ├── tables/            <- cells.csv + statistical-test tables
        └── figs/              <- 420 PDF + 420 PNG figures (+ LaTeX tables)
```

Raw per-run outputs (`experiments/out/runs/`, ~23 GB of per-cell HDF5
sample files, iteration logs, and truth reference sets) exceed GitHub
limits and are archived on Zenodo:
[**DOI 10.5281/zenodo.22228405**](https://doi.org/10.5281/zenodo.22228405).
Unpack the archive into
`experiments/out/` to enable the run-dependent figure scripts without
re-running the grid; alternatively, everything regenerates from
scratch (see tier 3 below).

## Reproducing the thesis results

The pipeline is idempotent: every script reads from
`experiments/out/` and writes back into it. All commands run from the
repository root.

### 0. Requirements

* Julia **1.11.x** (the Manifest pins 1.11.6).
* ~16 GB RAM; ~25 GB free disk if you re-run the full grid.

### 1. Install the exact environment

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This installs every package at the pinned version from
`Manifest.toml`. Then run the regression tests:

```sh
julia --project=. -t auto experiments/tests/runtests.jl
```

### Verification status

The initial release tree was verified end to end before publication
(the checks below). Later commits added figure generators and figure
outputs (angle/calibration heatmaps, iteration- and
accuracy-convergence figures, the master results table) that were
each produced *by the shipped scripts inside this tree* at the time
of their commit; the raw data (`cells.csv`, runs, truth) is unchanged
since the verification.

* `Pkg.instantiate()` from the pinned Manifest plus the full
  regression suite pass (truth self-consistency 42/42, truth vs.
  analytic density 10/10, algorithm and counter sanity sets).
* Ground truth regenerated from scratch matches the archived truth
  sets in **every data field bit-for-bit** for all deterministic
  constructions. The one exception is documented below.
* The headline heatmaps, test heatmap, dimension grid, recovery
  curves, and all fifteen thesis triangle plots were regenerated
  inside this tree from the shipped `cells.csv`, run data, and truth:
  **22 of 24 PNG outputs are byte-identical** to the shipped files
  (PDF twins differ only in embedded creation metadata). The two
  exceptions are the spiky M-ridges triangles, whose truth-sample
  overlay came from the regenerated reference (see below); with the
  archived truth they are byte-identical as well.

One reproducibility caveat, stated plainly: the spiky M-ridges truth
sampler is seeded, but the shipped reference files were drawn by an
earlier revision of the sampler with a different consumption order of
the same stream. Regeneration therefore yields a **statistically
equivalent, not bit-identical** reference for this one target (the
analytic log-evidence is bit-identical; sample moments agree at the
Monte-Carlo noise level of the 5×10⁴-draw reference). For bit-exact
reproduction of the shipped spiky figures and metrics, use the
archived truth files from the Zenodo record.

### 2. Tier 1 — inspect the shipped artifacts (no compute)

Every thesis number is already in this repository:

* `experiments/out/tables/cells.csv` — one row per run: every metric,
  the cost-counter value at termination, stop reason, `R̂`, wall time.
* `experiments/out/tables/headline_tests.csv` — seed-paired one-sided
  Wilcoxon tests with Holm correction and Cliff's δ effect sizes.
* `experiments/out/tables/headline_medians.csv` — per-cell medians
  behind the thesis tables.
* `experiments/out/figs/` — every figure of the thesis (see
  `THESIS-FIGURES.md` for the figure-by-figure audit map).

### 3. Tier 2 — regenerate truth, tables, and figures (minutes)

```sh
# Ground-truth sets (~5 min; writes experiments/out/truth/)
julia --project=. -t auto experiments/scripts/00_generate_truth.jl

# Statistical tests + headline medians from the shipped cells.csv
julia --project=. -t auto experiments/scripts/02b_tests.jl
julia --project=.          experiments/scripts/extract_headlines.jl

# Base figure catalogue from cells.csv + truth (~15 min)
julia --project=. -t 1 experiments/scripts/03_plots.jl

# Thesis-final figure passes (see THESIS-FIGURES.md for which figures
# each script owns; the run-dependent ones need out/runs/ from the
# Zenodo archive or a fresh grid run)
julia --project=. -t 1 experiments/tools/figs_headline.jl
julia --project=. -t 1 experiments/tools/figs_viz_recovery_scaling.jl
julia --project=. -t 1 experiments/tools/figs_recovery.jl
julia --project=. -t 1 experiments/tools/figs_triangles.jl
julia --project=. -t 1 experiments/tools/figs_iterconv.jl
julia --project=. -t 1 experiments/tools/figs_storyboard.jl
julia --project=. -t 1 experiments/tools/figs_runconv_singles.jl   # ~1-2 h
julia --project=. -t 1 experiments/tools/figs_runconv_grids.jl     # ~1-2 h
julia --project=. -t 1 experiments/tools/figs_runconv_overlays.jl
julia --project=. -t 1 experiments/tools/figs_accconv.jl
julia --project=. -t 1 experiments/tools/figs_runconv_shell.jl
```

### 4. Tier 3 — re-run the full benchmark grid (~24 h)

```sh
julia --project=. -t auto experiments/scripts/04_run_all.jl
```

`04_run_all.jl` is the master orchestrator: it generates truth where
missing, runs every (problem, algorithm, d, B, seed) cell into
`experiments/out/runs/`, then chains aggregation
(`02_aggregate.jl` → `cells.csv`), statistical tests, headline
extraction, and the base figure catalogue. A single cell can be run
in isolation with `experiments/scripts/01_run_cell.jl`; the
per-cell/per-seed RNG streams are fixed, so the grid is exactly
repeatable up to floating-point scheduling effects of multithreaded
reductions. After a fresh grid run, apply the figure passes of tier 2.

Quality gates: `experiments/tools/sanity_check.jl` (budget band,
`Neff ≤ #gradients`, `R̂` flags over `cells.csv`) and
`experiments/tools/check_figs.jl` (figure set completeness and
geometry).

## Relation to the thesis

The benchmark contract — problem specifications, ground-truth
derivations, metric definitions, protocol, and all conclusions — is
specified in the thesis (Ch. 7 "Benchmark Design and Test Problems",
Ch. 8 "Results", Appendices A–C). Source-code listings of every
target, algorithm invocation, metric helper, and the grid runner are
reproduced in the thesis's implementation appendix and correspond line-for-line to the
files in this repository. Comments of the form `Protocol §n` refer to
the benchmark protocol as specified in thesis Ch. 7; comments tagged
`V2-FIX-*` … `V8-FIX-*` record internal revision milestones of the
benchmark engine and are kept as engineering history.

## Revision of 25 September 2026

Following an external review of the thesis, two post-campaign
analyses were added. Neither modifies the archived runs, truth sets,
`cells.csv` or figures; all outputs live under
`experiments/out/revision_2026-09-25/`.

* **Frozen-mixture fresh-draw estimator**
  (`experiments/tools/fresh_draw_recompute.jl`). The archived
  MoleWhacker output is the accumulated population of all whacking
  iterations weighted against the final mixture, which is not a valid
  importance sampling estimator in general (thesis Sec. 5.7.1). For
  every one of the 795 archived MoleWhacker runs the tool loads the
  stored final mixture, draws `N_fresh = floor(B - N_L)` independent
  points (the budget the archived run left unspent), weights them by
  the exact target, and evaluates the archived metric pipeline on the
  fresh weighted sample. `cells_fresh.csv` holds one row per run
  (status, the two efficiencies `eta_cost_fresh = ESS / (N_L + N_fresh)`
  and `eta_proposal_fresh = ESS / N_fresh`, Pareto shape, W1, SWD,
  quantile errors, log-evidence error, wall time); `fresh_manifest.csv`
  records provenance (source-file hash, proposal hash, RNG seed,
  counts, costs). `--test` runs the estimator checks (constant ratio,
  Gaussian target, the two-component counterexample, transform and
  full-mixture identities against BAT and Distributions, thread-count
  independence). The per-run `.jld2` payloads (10k-point resample, top
  weights) are not committed; they are regenerable from the archived
  mixtures with the recorded seeds.
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
  `partition_reference.csv`, `partition_calibration.csv` (checks
  against the 10^6-draw truth sets and multinomial finite-sample
  floors), `partition_mass.csv` (every archived multimodal cell with
  population/output weights, plus the fresh MoleWhacker draws),
  `partition_tests.csv` (two-sided seed-paired Wilcoxon and Cliff's
  delta at the headline setting), and the two `figs/partition__*`
  figures.
* `experiments/tools/revision_tables.py` (Python, pandas) emits the
  LaTeX tables of thesis Appendix A from these CSVs, and
  `table_master_results.jl` now prints the exclusion reason of a `0/n`
  cell per cause (r: R-hat, b: budget band, i: infeasible) and the
  partition-mass column in place of the retired recovery rate.

## License

MIT — see `LICENSE`. Every Julia source file carries an SPDX header.

## Citation

```bibtex
@mastersthesis{reindel2026molewhacker,
  author = {Valentin Reindel},
  title  = {Importance Sampling Methods in the Bayesian Analysis Toolkit},
  school = {Technical University of Munich, Department of Physics},
  year   = {2026},
  note   = {Companion code: \url{https://github.com/valeagent/molewhacker-bench};
            data archive: \url{https://doi.org/10.5281/zenodo.22228405}}
}
```

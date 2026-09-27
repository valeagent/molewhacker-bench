<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
-->

# Thesis figures and tables: audit map

Every benchmark figure and table embedded in the submitted thesis
(numbering of the final PDF, September 2026), the file that is
embedded, the directory it is taken from, the script that
(re)generates it, and the data it reads. The plot functions live in
`experiments/src/plotting.jl`; the scripts listed here are drivers
around them. PNG twins of every PDF are in a `png/` subdirectory.

**Two figure directories.** The thesis's summary heatmaps, dimension
grids, partition-mass figures and MoleWhacker triangle plots are the
**primary** assets in `experiments/out/primary_fresh_2026-09-27/figs/`
(MoleWhacker = independent final sample from the frozen mixture). The
adaptation diagnostics, the comparator-only panels and the visual
identities are taken from the historical catalogue
`experiments/out/figs/`. Both directories contain files of the same
name (for example `summary__all__d5__B5e5__heatmap-eta.pdf`) whose
MoleWhacker content differs (final sample vs. adaptation population);
take each asset from the directory named in the tables below.

Data inputs:

* **primary** — `primary_fresh_2026-09-27/primary_cells.csv` and
  `primary_cell_medians.csv` (in Git; built once by
  `tools/primary_fresh_view.jl`, see README route A/B).
* **partition** — `revision_2026-09-25/partition_{reference,calibration,mass}.csv`
  (in Git).
* **cells_fresh** — `revision_2026-09-25/cells_fresh.csv` and
  `fresh_manifest.csv` (in Git): per-run numerical summaries of the
  final samples.
* **payload** — `revision_2026-09-25/fresh_runs/<cell>.jld2`: **not in
  Git and not in the Zenodo record**; reconstructed by
  `tools/fresh_draw_recompute.jl` from the archived mixtures (README
  route B; target evaluations, no adaptation).
* **cells** — `experiments/out/tables/cells.csv` (in Git; historical,
  MW rows = adaptation population).
* **truth** — `experiments/out/truth/*.h5`: truth archive of the Zenodo
  record (use the archived files; `scripts/00_generate_truth.jl`
  regenerates a statistically equivalent but not bit-identical spiky
  M-ridges reference).
* **runs** — per-run `result.h5` + MoleWhacker iteration logs under
  `experiments/out/runs/`: runs archive of the Zenodo record (original
  grid) or a full re-run (`scripts/04_run_all.jl`, ~24 h).
* **story** — instrumented per-iteration snapshot runs under
  `experiments/out/runs_storyboard/`: storyboard archive of the Zenodo
  record (`mridges_d2_seed11`, `mridges_d2_seed11_ns4`,
  `mridges_d5_seed11`, `mridges_d5_seed11_acc`, `funnel_d5_seed11_acc`).
* **instrumented reruns** — snapshot caches that are *not* in the
  Zenodo record (`experiments/out/runs_klconv/`, the remaining
  `runs_storyboard/*_acc/` directories). When the cache is absent the
  generator performs seeded instrumented MoleWhacker runs, i.e.
  sampling. These figures are adaptation diagnostics; regenerating
  them is not needed for any primary result.

Zenodo record: [10.5281/zenodo.22228405](https://doi.org/10.5281/zenodo.22228405)
(original benchmark grid: runs, truth, storyboard; published
1 September 2026; no product of the 25/27 September analyses).

## Chapter 5 — The MoleWhacker Algorithm (adaptation diagnostics)

| Thesis figure | File | Directory | Generator | Data |
|---|---|---|---|---|
| Fig. 5.1 (whacking-loop storyboard) | `story__mridges__d2__B5e5__mw__loop-weakinit` | `out/figs/` | `tools/figs_storyboard.jl` | story |
| Fig. 5.2 (measured iteration behavior, 2×2) | `iterconv__quad__d5__B5e5__mw__eta` | `out/figs/` | `tools/figs_storyboard.jl` (re-render of the `tools/figs_iterconv.jl` panels) | runs |
| Fig. 5.3 (proposal divergence, KL over iterations) | `klconv__quad__d5__B5e5__mw__klcube` | `out/figs/` | `tools/figs_klconv.jl` | instrumented reruns (`out/runs_klconv/`) + truth |
| Fig. 5.4 (accuracy over adaptive iterations) | `acciter__quad__d5__B5e5__mw__w1-qe975` | `out/figs/` | `tools/figs_acciter.jl` | instrumented reruns (`runs_storyboard/*_acc/`; the record holds the funnel and M-ridges caches) + truth |

## Chapter 6 — Comparison Algorithms

| Thesis figure | File | Directory | Generator | Data |
|---|---|---|---|---|
| Fig. 6.1 (MH running efficiency, funnel) | `runconv__funnel__d5__B5e5__mh__eta` | `out/figs/` | `tools/figs_runconv_singles.jl` | runs |
| Fig. 6.2 (NUTS running efficiency, funnel) | `runconv__funnel__d5__B5e5__nuts__eta` | `out/figs/` | ″ | runs |
| Fig. 6.3 (NS running efficiency, funnel) | `runconv__funnel__d5__B5e5__ns__eta` | `out/figs/` | ″ | runs |
| Fig. 6.4 (IS running efficiency, funnel) | `runconv__funnel__d5__B5e5__is__eta` | `out/figs/` | ″ | runs |

## Chapter 7 — Benchmark Design (visual identities)

Each of Figs. 7.1–7.7 is a pair: the 3-D density surface at `d = 2`
and the contour + marginals overview at `d = 5`.

| Thesis figure | Files | Directory | Generator | Data |
|---|---|---|---|---|
| Fig. 7.1 (MVN) | `viz__mvn__d2__surface`, `viz__mvn__d5__overview` | `out/figs/` | `tools/figs_viz_recovery_scaling.jl` | truth |
| Fig. 7.2 (banana) | `viz__banana__d2__surface`, `viz__banana__d5__overview` | `out/figs/` | ″ | truth |
| Fig. 7.3 (funnel) | `viz__funnel__d2__surface`, `viz__funnel__d5__overview` | `out/figs/` | ″ | truth |
| Fig. 7.4 (M-ridges) | `viz__mridges__d2__surface`, `viz__mridges__d5__overview` | `out/figs/` | ″ | truth |
| Fig. 7.5 (shell) | `viz__shell__d2__surface`, `viz__shell__d5__overview` | `out/figs/` | ″ | truth |
| Fig. 7.6 (spiky M-ridges) | `viz__mspiky__d2__surface`, `viz__mridges_spiky__d5__overview` | `out/figs/` | ″ | truth |
| Fig. 7.7 (eggbox) | `viz__eggbox__d2__surface`, `viz__eggbox__d5__overview` | `out/figs/` | ″ | truth |

## Chapter 8 — Results (primary: MoleWhacker = final sample)

| Thesis figure | File(s) | Directory | Generator | Data |
|---|---|---|---|---|
| Fig. 8.1 (cost-efficiency heatmap) | `summary__all__d5__B5e5__heatmap-eta` | `primary_fresh_2026-09-27/figs/` | `tools/figs_primary.jl` | primary |
| Fig. 8.2 (log-evidence error heatmap) | `summary__all__d5__B5e5__heatmap-dlogz` | ″ | ″ | primary |
| Fig. 8.3 (partition-mass error, two panels) | `partition__mridges__d5__Ball__all__all`, `partition__mridges_spiky__d5__Ball__all__all` | ″ | ″ | partition |
| Fig. 8.4 (marginal-W1 heatmap) | `summary__all__d5__B5e5__heatmap-W1` | ″ | ″ | primary |
| Fig. 8.5 (median quantile-error heatmap) | `summary__all__d5__B5e5__heatmap-qe500` | ″ | ″ | primary |
| Fig. 8.6 (0.975-quantile-error heatmap) | `summary__all__d5__B5e5__heatmap-qe975` | ″ | ″ | primary |
| Fig. 8.7 (dimensional small multiples, B = 5e5 and 5e4) | `dim__all__B5e5__all__all`, `dim__all__B5e4__all__all` | ″ | ″ | primary |

## Appendix A — Supplementary Benchmark Results

| Thesis figure | File(s) | Directory | Generator | Data |
|---|---|---|---|---|
| Figs. A.1–A.4 (efficiency heatmaps: B = 5e3, B = 5e4, d = 2, d = 10) | `summary__all__d5__B5e3__heatmap-eta`, `summary__all__d5__B5e4__heatmap-eta`, `summary__all__d2__B5e5__heatmap-eta`, `summary__all__d10__B5e5__heatmap-eta` | `primary_fresh_2026-09-27/figs/` | `tools/figs_primary.jl` (`--only-offdim` for the d = 2 / d = 10 maps) | primary |
| Figs. A.5–A.8 (log-evidence error heatmaps, same angles) | `summary__all__{d5__B5e3,d5__B5e4,d2__B5e5,d10__B5e5}__heatmap-dlogz` | ″ | ″ | primary |
| Figs. A.9–A.12 (marginal-W1 heatmaps, same angles) | `summary__all__{d5__B5e3,d5__B5e4,d2__B5e5,d10__B5e5}__heatmap-W1` | ″ | ″ | primary |
| Figs. A.13–A.15 (quantile-error heatmaps at 0.025, 0.16, 0.84) | `summary__all__d5__B5e5__heatmap-qe{025,160,840}` | ″ | ″ | primary |
| Fig. A.16 (MVN triangles) | `tri__mvn__d5__B5e5__mw` (a) | `primary_fresh_2026-09-27/figs/` | `tools/figs_triangles_primary.jl` | payload + runs + truth |
| | `tri__mvn__d5__B5e5__nuts` (b) | `out/figs/` | `tools/figs_triangles.jl` | runs + truth |
| Fig. A.17 (banana triangles) | `tri__banana__d5__B5e5__mw` (a) / `tri__banana__d5__B5e5__nuts` (b) | primary / `out/figs/` | as A.16 | as A.16 |
| Fig. A.18 (funnel triangles) | `tri__funnel__d5__B5e5__mw` (a) / `tri__funnel__d5__B5e5__nuts` (b) | primary / `out/figs/` | as A.16 | as A.16 |
| Fig. A.19 (M-ridges, NUTS and NS) | `tri__mridges__d5__B5e5__nuts`, `tri__mridges__d5__B5e5__ns` | `out/figs/` | `tools/figs_triangles.jl` | runs + truth |
| Fig. A.20 (MW on M-ridges, d = 5) | `tri__mridges__d5__B5e5__mw` | `primary_fresh_2026-09-27/figs/` | `tools/figs_triangles_primary.jl` | payload + runs + truth |
| Fig. A.21 (MW on shell) | `tri__shell__d5__B5e5__mw` | ″ | ″ | payload + runs + truth |
| Fig. A.22 (spiky M-ridges, MW and MH) | `tri__mridges_spiky__d5__B5e5__mw` (a) / `tri__mridges_spiky__d5__B5e5__mh` (b) | primary / `out/figs/` | as A.16 | as A.16 |
| Fig. A.23 (MW on M-ridges, d = 2 and d = 10) | `tri__mridges__d2__B5e5__mw`, `tri__mridges__d10__B5e5__mw` | `primary_fresh_2026-09-27/figs/` | `tools/figs_triangles_primary.jl` | payload + runs + truth |
| Fig. A.24 (eggbox, MW and MH, d = 2) | `tri__eggbox__d2__B5e5__mw` (a) / `tri__eggbox__d2__B5e5__mh` (b) | primary / `out/figs/` | as A.16 | as A.16 |
| Fig. A.25 (iteration convergence, scaling targets) | `iterconv__{mvn,banana,funnel,mridges}__d5__B5e5__mw__eta` | `out/figs/` | `tools/figs_storyboard.jl` / `tools/figs_iterconv.jl` | runs |
| Fig. A.26 (iteration convergence, specialists) | `iterconv__{shell,mridges_spiky}__d5__B5e5__mw__eta`, `iterconv__eggbox__d2__B5e5__mw__eta` | `out/figs/` | ″ | runs |
| Fig. A.27 (accuracy over iterations, per target) | `acciter__pertarget__d5__B5e5__mw__all` | `out/figs/` | `tools/figs_acciter.jl` | instrumented reruns + truth |
| Fig. A.28 (running marginal accuracy, funnel) | `accconv__funnel__d5__B5e5__all__W1` | `out/figs/` | `tools/figs_accconv.jl` | runs + story + truth |
| Figs. A.29–A.35 (running-efficiency atlas, d = 5; eggbox d = 2) | `runconv__{mvn,banana,funnel,mridges,shell,mridges_spiky}__d5__B5e5__all__grid`, `runconv__eggbox__d2__B5e5__all__grid` | `out/figs/` | `tools/figs_runconv_grids.jl` | runs |
| Figs. A.36–A.39 (running-efficiency atlas, d = 10) | `runconv__{mvn,banana,funnel,mridges}__d10__B5e5__all__grid` | `out/figs/` | `tools/figs_runconv_d10.jl` | runs |
| Fig. A.40 (running marginal accuracy, M-ridges) | `accconv__mridges__d5__B5e5__all__W1` | `out/figs/` | `tools/figs_accconv.jl` | runs + story + truth |
| Figs. A.41–A.45 (running marginal accuracy, remaining targets) | `accconv__{mvn,banana,shell,mridges_spiky}__d5__B5e5__all__W1`, `accconv__eggbox__d2__B5e5__all__W1` | `out/figs/` | `tools/figs_accconv_atlas.jl` | runs + instrumented reruns + truth |

The MoleWhacker panels of Figs. A.16–A.24 show the equal-weight
display resample of the final sample stored in the payload, with the
native `N_eff` and `eta` of the weighted final sample in the footer
(`figs/triangle_manifest.csv`: payload, source `result.h5` SHA-256,
representative seed). The comparator panels are the archived
admissible output of the original grid.

## Thesis tables

| Thesis table | Fragment / source | Directory | Generator | Data |
|---|---|---|---|---|
| Tab. 8.1 (headline final-sample metrics) | `B-fresh-headline.tex` | `primary_fresh_2026-09-27/tables/` | `tools/revision_tables_primary.py` | primary + cells_fresh + partition |
| Tab. 8.2 (efficiency, headline) | `ch8-headline-eta.tex` | ″ | ″ | primary |
| Tab. 8.3 (MW stopping behaviour; adaptation diagnostic) | transcribed from `experiments/out/tables/mw_iteration_summary.csv` | `out/tables/` | `tools/figs_iterconv.jl` | runs |
| Tab. 8.4 (efficiency by budget) | `ch8-headline-eta-by-budget.tex` | `primary_fresh_2026-09-27/tables/` | `tools/revision_tables_primary.py` | primary |
| Tab. 8.5 (log-evidence error, headline) | `ch8-headline-dlogz.tex` | ″ | ″ | primary |
| Tab. 8.6 (partition-mass error, headline) | `B-partition-headline.tex` | ″ | ″ | partition + primary |
| Tab. 8.7 (quantile errors at five levels) | `B-quantile-headline.tex` | ″ | ″ | primary |
| Tab. A.1 (archived population vs. final sample; labelled diagnostic) | `fresh_vs_population.tex` | `revision_2026-09-25/tables/` | `tools/revision_tables.py` | cells + cells_fresh |
| Tab. A.2 (available seeds by reason) | `B-pass-fractions.tex` | `primary_fresh_2026-09-27/tables/` | `tools/revision_tables_primary.py` | primary + cells_fresh |
| Tab. A.3 (measured wall time per phase) | `B-wall-times.tex` | ″ | ″ | cells (adaptation wall time) + cells_fresh (final-sample phases) |
| Tab. A.4 (master results table) | `B-master-table.tex` | ″ | `tools/table_master_primary.jl` | primary + partition |

The thesis wraps the bare tabular bodies in its own `table`
environments (caption and label in the thesis source); `B-master-table`,
`B-pass-fractions` and `B-wall-times` are complete longtables.

## Historical map

Before 25 September 2026 this file described the population-based
figure set of the original campaign (MoleWhacker panels and heatmap
cells from the accumulated adaptation population, the retired
centroid mode-recovery figure, the Wilcoxon test heatmap) with the
figure numbering of the thesis drafts of that time. That map is in the
Git history of this file (commits up to `03d94d0`); those figures
remain in `experiments/out/figs/` as engineering history and are not
embedded in the submitted thesis unless listed above.

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# ExperimentsBase.jl — module entrypoint, re-exports everything
# =============================================================================
#
# Implements the benchmark protocol specified in the thesis (Ch. 7,
# with the source listings reproduced in the implementation appendix).
# Loading order:
#
#     1.  counter.jl                -- LikelihoodCounter
#     2.  shared utilities          -- MethodResult, logsumexp, weighted helpers
#     3.  problems/Problem*.jl      -- 7 ProblemConfig + log_f's
#     4.  truths/truth_*.jl         -- compute_truth_* / sample_truth_*
#     5.  metrics.jl                -- W1, SWD, Neff, dlogZ, QE, KL_cube
#     6.  algorithms/algo_*.jl      -- 5 run_<alg> entry points
#     7.  plotting.jl               -- figure factory
#
# A user script `using .ExperimentsBase` after `include`-ing
# this file gets every public symbol it needs.

module ExperimentsBase

using Random
using Random: AbstractRNG, MersenneTwister, Xoshiro
using LinearAlgebra
using Statistics
using StatsBase
using Distributions
using DataFrames
using CSV
using JLD2
using ForwardDiff
using DensityInterface
using BAT
using ProgressMeter
using Printf
using SHA
using Logging

# -- 1. counter --------------------------------------------------------------
include("counter.jl")
export LikelihoodCounter, cost, is_over_budget, reset!, freeze!, snapshot

# -- 2. shared utilities -----------------------------------------------------
include("base.jl")
export MethodResult,
       logsumexp,
       weighted_quantile,
       resample_to_equal_weight,
       provenance_string,
       BUDGET_GRID,
       SEED_GRID,
       SEED_GRID_V2,
       PROBLEM_NAMES,
       PROBLEM_NAMES_V2,
       ALG_NAMES,
       SCALING_PROBLEMS,
       SCALING_DIMENSIONS,
       SCALING_PROBLEMS_V2,
       SCALING_DIMENSIONS_V2,
       HEADLINE_DIMENSION

# -- 3. problems -------------------------------------------------------------
include("problems/ProblemMVN.jl")
include("problems/ProblemBanana.jl")
include("problems/ProblemFunnel.jl")
include("problems/ProblemMRidges.jl")
include("problems/ProblemShell.jl")
include("problems/ProblemMRidgesSpiky.jl")
include("problems/ProblemEggbox.jl")
export ProblemConfig,
       ConfigMVN, ConfigBanana, ConfigFunnel, ConfigMRidges, ConfigShell,
       ConfigMRidgesSpiky, ConfigEggbox,
       make_config_mvn, make_config_banana, make_config_funnel,
       make_config_mridges, make_config_shell,
       make_config_mridges_spiky, make_config_eggbox,
       mridges_spiky_outer_means, mridges_spiky_jitters,
       mridges_spiky_modes_theta1, mridges_spiky_modes,
       eggbox_modes,
       build_log_f,
       prior_box,
       posterior_measure

# -- 4. truths ---------------------------------------------------------------
include("truths/truth_common.jl")
include("truths/truth_mvn.jl")
include("truths/truth_banana.jl")
include("truths/truth_funnel.jl")
include("truths/truth_mridges.jl")
include("truths/truth_shell.jl")
include("truths/truth_mridges_spiky.jl")
include("truths/truth_eggbox.jl")
export TruthSet, compute_truth, sample_truth,
       save_truth, load_truth,
       compute_truth_mridges_spiky, sample_truth_mridges_spiky,
       compute_truth_eggbox, sample_truth_eggbox

# -- 5. metrics --------------------------------------------------------------
include("metrics.jl")
export w1_marginal_avg,
       sliced_wasserstein,
       neff,
       neff_kish,
       neff_chain,
       rhat_per_coord,
       rhat_max,
       dlogZ,
       quantile_errors,
       kl_cube_mw,
       compute_metrics_for_cell,
       W1_EVAL_N,
       _is_budget_exempt,
       _BUDGET_EXEMPT_TERMINATORS,
       mode_recovery,
       mmd_rbf,
       median_pairwise_distance,
       bootstrap_median_ci,
       bootstrap_paired_median_ci,
       wilcoxon_signed_rank,
       cliffs_delta,
       holm_bonferroni,
       friedman_chi2,
       nemenyi_critical_difference,
       _eps_radius_for_problem

# -- 6. algorithms -----------------------------------------------------------
include("algorithms/algo_is.jl")
include("algorithms/algo_mh.jl")
include("algorithms/algo_nuts.jl")
include("algorithms/algo_ns.jl")
include("algorithms/algo_mw.jl")
export run_algorithm,
       run_is, run_mh, run_nuts, run_ns, run_mw,
       save_method_result, load_method_result,
       cell_dir, cell_metadata,
       write_metadata_json,
       read_metadata_json

# -- 7. plotting -------------------------------------------------------------
include("plotting.jl")
export ALG_COLOR, PROBLEM_COLOR,
       ALG_LINESTYLE, ALG_MARKER, ALG_LINEWIDTH,
       ALG_LABEL, PROBLEM_LABEL, PROBLEM_MARKER,
       ALG_ORDER, PROBLEM_ORDER,
       METRIC_TITLE, SCORE_LABEL,
       figure_size, tex_label,
       set_pub_theme!,
       save_pdf, save_png,
       fig_filename,
       standard_axis!,
       tight_limits,
       # Catalogue A
       fig_viz_density,
       fig_viz_samples_truth,
       fig_viz_marginals,
       fig_viz_overview,
       fig_viz_surface_3d,
       fig_viz_gallery,
       # V5 cleanup helpers
       drop_v5_excluded_rows!,
       HEATMAP_PROBLEM_ORDER,
       # Catalogue B
       fig_conv,
       fig_eff,
       fig_logz,
       fig_qe,
       fig_tri,
       fig_pareto_per_problem,
       # Catalogue C
       fig_iter_mw,
       fig_mw_itercurves,
       fig_mw_itercurves_quad,
       fig_mw_storyboard,
       fig_convergence_cell,
       fig_runconv_panel,
       convergence_curve,
       fig_accuracy_cell,
       accuracy_curve,
       fig_runconv_grid,
       fig_diag_mw_moles,
       fig_diag_nuts_div,
       fig_diag_mh_trace,
       fig_diag_ns_live,
       fig_diag_is_weights,
       # Catalogue D
       fig_summary_heatmap,
       fig_scaling,
       fig_pareto_grand,
       # V3 (Protocol §10.17)
       fig_hardness_mridges_spiky,
       fig_recovery,
       fig_dim_grid,
       fig_tests_heatmap,
       fig_cd_diagram,
       # LaTeX tables
       write_summary_table

end # module

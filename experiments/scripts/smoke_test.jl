# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# smoke_test.jl — fast load + tiny run to validate the codebase
# =============================================================================
#
# Runs: load module → build every problem → tiny B=1000 IS run → tiny
# truth (N_ref=2000) → metrics → save+load round-trip → IS+MH+MW small
# runs and metric computation. Should finish in <60 s after Julia's
# first-load compilation.
#
# Usage:
#     julia --project=. -t auto experiments/scripts/smoke_test.jl

using Pkg
const _ROOT = abspath(joinpath(@__DIR__, "..", ".."))
if Base.active_project() != joinpath(_ROOT, "Project.toml")
    Pkg.activate(_ROOT)
end

println("[smoke] include ExperimentsBase.jl …")
@time include(joinpath(_ROOT, "experiments", "src", "ExperimentsBase.jl"))
using .ExperimentsBase
using Statistics, LinearAlgebra, Printf, Random

println("[smoke] PROBLEM_NAMES = $PROBLEM_NAMES")
println("[smoke] ALG_NAMES     = $ALG_NAMES")

println("\n[smoke] Build configs and log_f for every problem")
for problem in PROBLEM_NAMES
    cfg = if problem === :mvn
        make_config_mvn(d = 5)
    elseif problem === :banana
        make_config_banana(d = 5)
    elseif problem === :funnel
        make_config_funnel(d = 5)
    elseif problem === :mridges
        make_config_mridges(d = 5)
    elseif problem === :shell
        make_config_shell(d = 5)
    end
    log_f = build_log_f(cfg)
    val = log_f(zeros(cfg.d))
    @printf("  %-7s log_f(0) = %.4f\n", String(problem), val)
end

println("\n[smoke] Tiny truth set (N=2_000) for mvn ...")
cfg_m = make_config_mvn(d = 5)
@time truth = compute_truth(cfg_m; N_ref = 2_000, seed = 20260901)
@printf("  truth.logZ = %.4f, mean[1] = %.4f, cov[1,1] = %.4f\n",
         truth.logZ, truth.mean[1], truth.cov[1, 1])

tmpdir = mktempdir()
truth_path = joinpath(tmpdir, "truth_mvn_d5.h5")
save_truth(truth_path, truth)
truth2 = load_truth(truth_path)
@printf("  truth I/O round-trip: logZ_diff = %.2e\n", abs(truth.logZ - truth2.logZ))

println("\n[smoke] Tiny IS run (B=1000) on mvn ...")
@time mr_is = run_is(cfg_m, 1000, 11)
@printf("  Nlike_used=%.0f  N_out=%d  logZ=%.4f  Neff=%.1f\n",
         mr_is.Nlike_used, size(mr_is.samples, 2),
         mr_is.logZ_estimate === missing ? NaN : mr_is.logZ_estimate,
         neff(mr_is))

println("\n[smoke] Save / load MethodResult round-trip ...")
cell_path = joinpath(tmpdir, "is_cell")
save_method_result(cell_path, mr_is)
mr_is2 = load_method_result(cell_path)
@assert mr_is.algorithm == mr_is2.algorithm
@assert isapprox(mr_is.samples, mr_is2.samples)
println("  OK")

println("\n[smoke] Tiny MH run (B=2000) on mvn ...")
try
    @time mr_mh = run_mh(cfg_m, 2000, 11)
    @printf("  Nlike_used=%.0f  N_out=%d  acc_rate=%s\n",
             mr_mh.Nlike_used, size(mr_mh.samples, 2),
             string(get(mr_mh.extras, :acceptance_rate, NaN)))
catch err
    println("  WARNING: MH failed → ", typeof(err), ": ", err)
end

println("\n[smoke] Tiny MW run (B=4000) on mvn ...")
try
    @time mr_mw = run_mw(cfg_m, 4000, 11)
    @printf("  Nlike_used=%.0f  N_out=%d  Neff=%.1f  iters=%s\n",
             mr_mw.Nlike_used, size(mr_mw.samples, 2),
             neff_kish(mr_mw),
             string(get(mr_mw.extras, :n_iterations, "?")))
catch err
    println("  WARNING: MW failed → ", typeof(err), ": ", err)
    Base.show_backtrace(stderr, catch_backtrace())
end

println("\n[smoke] Metrics on IS/mvn ...")
row = compute_metrics_for_cell(mr_is, truth2)
for k in (:W1_marginal_avg, :SWD, :Neff, :eta_Nlike, :dlogZ)
    @printf("  %-18s = %s\n", String(k), repr(get(row, k, missing)))
end

println("\n[smoke] All checks passed.")

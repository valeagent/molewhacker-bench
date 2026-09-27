# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# figs_triangles_primary.jl — PRIMARY MoleWhacker triangle (corner) plots
# (revision of 27 September 2026).
#
# Regenerates the nine MoleWhacker triangle plots of thesis Appendix B with
# their EXISTING file names (tri__<problem>__d<d>__B5e5__mw) from the saved
# independent final sample of the frozen mixture — the payload
# experiments/out/revision_2026-09-25/fresh_runs/<cell>.jld2 written by
# fresh_draw_recompute.jl on 25 September 2026 — instead of the archived
# adaptation population in experiments/out/runs/<cell>/result.h5:
#
#   * the algorithm cloud is the stored equal-weight display resample
#     "resample_10k_theta" (d x 10^4 points, resampled from the weighted
#     final sample); the truth reference, the tight axis limits and the
#     coordinate subset (all coordinates for d <= 6, theta_1..3 and theta_d
#     for d = 10) are unchanged (fig_tri defaults);
#   * the footer prints the NATIVE final-sample values N_eff = Neff_fresh
#     (Kish ESS of the weighted final sample) and eta = Neff_fresh / C_total,
#     via the new default-preserving fig_tri keywords neff_override /
#     cost_override — never the display size 10^4 or eta = 1;
#   * the 1-D marginal KDEs use h = 0.9 min(sd, IQR/1.34) n_eff^(-1/5) with
#     n_eff = min(10^4, Neff_fresh) (fig_tri keyword bandwidth_neff).
#
# Representative seed: the SAME rule as the archived plots
# (03_plots.jl `_median_seed_run`: the middle element, index ceil(n/2), of
# the existing seed grid of the cell) — 239 for 20 seeds, 41 for 5 seeds,
# 97 for 10 seeds. The fixed seeds below are asserted against both an
# isfile() replication of that rule and the actual `_median_seed_run`
# result. The result.h5 the payload was drawn from is verified by SHA-256
# against summary["source_sha256"]. Nothing is sampled and no target is
# evaluated; experiments/out/runs, truth and revision_2026-09-25 are only
# read.
#
# Outputs (experiments/out/primary_fresh_2026-09-27/figs/):
#   tri__<problem>__d<d>__B5e5__mw.pdf  (+ png/<same>.png, 4 px/pt)
#   triangle_manifest.csv
#
# Usage:  julia --project=. -t 8 experiments/tools/figs_triangles_primary.jl
# =============================================================================

include(joinpath(@__DIR__, "..", "scripts", "03_plots.jl"))   # helpers only; main() is guarded
using JLD2, SHA
using DataFrames, CSV, Statistics, Printf

const OUT         = joinpath(_ROOT, "experiments", "out")
const RUNS_DIR    = joinpath(OUT, "runs")
const TRUTH_DIR   = joinpath(OUT, "truth")
const REV         = joinpath(OUT, "revision_2026-09-25")
const FRESH_DIR   = joinpath(REV, "fresh_runs")
const PRIMARY_DIR = joinpath(OUT, "primary_fresh_2026-09-27")
const FIGS_DIR    = joinpath(PRIMARY_DIR, "figs")
const B_TRI       = 5e5
const N_DISPLAY   = 10_000
const FOOTER_NOTE = "final sample; 5000 of 10,000 display draws shown"

# (problem, d, fixed representative seed) — thesis Appendix B MW triangles.
const JOBS = [
    (:mvn,           5, 239), (:banana,  5, 239), (:funnel,  5, 239),
    (:mridges,       5, 239), (:shell,   5,  41), (:mridges_spiky, 5, 97),
    (:mridges,       2, 239), (:mridges, 10, 239), (:eggbox, 2, 97),
]

_cell(prob, d, seed) = basename(cell_dir(OUT, prob, :mw, d, B_TRI, seed))

# isfile() replication of _median_seed_run's choice (middle of the existing
# seed grid, index ceil(n/2), in SEED_GRID order) without loading any run.
function _middle_existing_seed(prob::Symbol, d::Int)
    existing = [s for s in SEED_GRID if isfile(joinpath(RUNS_DIR, _cell(prob, d, s), "result.h5"))]
    isempty(existing) && error("no archived run for $prob d=$d")
    return existing[ceil(Int, length(existing) / 2)], length(existing)
end

_sha256_file(path) = open(path, "r") do io
    bytes2hex(sha256(io))
end

function main()
    isdir(FIGS_DIR) || mkpath(FIGS_DIR)
    prim = CSV.read(joinpath(PRIMARY_DIR, "primary_cells.csv"), DataFrame)
    part = CSV.read(joinpath(REV, "partition_mass.csv"), DataFrame)
    rows = NamedTuple[]
    for (prob, d, seed_fixed) in JOBS
        # ---- representative seed: fixed == middle of existing grid == _median_seed_run
        seed_mid, n_exist = _middle_existing_seed(prob, d)
        mr_ref = _median_seed_run(RUNS_DIR, prob, :mw, d, B_TRI)
        mr_ref === nothing && error("_median_seed_run returned nothing for $prob d=$d")
        @assert seed_fixed == seed_mid == mr_ref.seed "seed selection mismatch for $prob d=$d: fixed $seed_fixed, middle-of-grid $seed_mid, _median_seed_run $(mr_ref.seed)"
        @info "representative seed" prob d seed = seed_fixed n_seeds_existing = n_exist rule = "middle of existing seed grid (index ceil(n/2)); equals _median_seed_run"

        cell = _cell(prob, d, seed_fixed)
        payload = joinpath(FRESH_DIR, cell * ".jld2")
        isfile(payload) || error("payload missing: $payload")
        s, theta = JLD2.jldopen(payload, "r") do f
            haskey(f, "resample_10k_theta") || error("payload $cell has no final sample (status $(f["summary"]["status"]))")
            f["summary"], Matrix{Float64}(f["resample_10k_theta"])
        end
        @assert s["status"] == "full_remaining_budget" && s["weight_status"] == "ok" "cell $cell is not eligible"
        @assert size(theta) == (d, N_DISPLAY) "display resample of $cell has size $(size(theta))"
        @assert all(isfinite, theta)
        C_old = Float64(s["C_old"]); N_fresh = Float64(s["N_fresh"]); C_total = Float64(s["C_total"])
        Neff_fresh = Float64(s["Neff_fresh"]); eta_fresh = Float64(s["eta_cost_fresh"])
        @assert C_old + N_fresh == C_total == B_TRI "budget accounting of $cell: C_old=$C_old N_fresh=$N_fresh C_total=$C_total"
        @assert isapprox(eta_fresh, Neff_fresh / C_total; rtol = 1e-12)

        # ---- provenance: payload source == archived result.h5 (SHA-256)
        src_sha = String(s["source_sha256"])
        h5 = joinpath(RUNS_DIR, cell, "result.h5")
        @assert _sha256_file(h5) == src_sha "SHA-256 of $h5 differs from summary source_sha256"

        # ---- consistency with the primary view (primary_cells.csv)
        pr = prim[(prim.problem .== String(prob)) .& (prim.algorithm .== "mw") .& (prim.d .== d) .&
                  (prim.B .== B_TRI) .& (prim.seed .== seed_fixed), :]
        @assert nrow(pr) == 1
        pr = pr[1, :]
        @assert String(pr.eligibility_status) == "eligible"
        @assert pr.Neff == Neff_fresh && pr.Nlike_used == C_total && pr.eta_Nlike == eta_fresh
        @assert Float64(pr.N_fresh) == N_fresh && Float64(pr.C_total) == C_total
        @assert String(pr.source_sha256) == src_sha

        # ---- equal-weight display MethodResult; native values go through the kwargs
        logZ = Float64(s["logZ_fresh"]); logZ_se = Float64(s["logZ_fresh_se"])
        extras = Dict{Symbol,Any}(
            :estimator => String(s["estimator_version"]), :display_sample => "resample_10k_theta",
            :Neff_fresh => Neff_fresh, :C_total => C_total, :N_fresh => N_fresh, :C_old => C_old,
            :payload => payload, :source_sha256 => src_sha)
        mr = MethodResult(:mw, prob, d, seed_fixed, B_TRI, C_total, Int(pr.n_primal), Int(pr.n_grad_partials),
                          Float64(pr.wall_time_s), theta, ones(N_DISPLAY), fill(NaN, N_DISPLAY), logZ, logZ_se, extras)
        # The trap this script avoids: the equal-weight display sample has Kish ESS 10^4.
        @assert isapprox(neff(mr), N_DISPLAY; rtol = 1e-9)

        truth = _load_truth(TRUTH_DIR, prob, d)
        truth === nothing && error("truth missing for $prob d=$d")
        bw_neff = min(Float64(N_DISPLAY), Neff_fresh)
        # Marginal KDE bandwidths actually used (same rule as fig_tri's
        # `_marginal_bandwidth`), recorded per shown coordinate.
        h_coords = map(d <= 6 ? (1:d) : (1, 2, 3, d)) do j
            xs = vec(theta[j, :]); sd = std(xs); iqr = quantile(xs, 0.75) - quantile(xs, 0.25)
            spread = min(sd, iqr / 1.34); (isfinite(spread) && spread > 0) || (spread = sd)
            0.9 * spread * bw_neff^(-1 / 5)
        end
        fig = fig_tri(mr, truth; neff_override = Neff_fresh, cost_override = C_total,
                      footer_note = FOOTER_NOTE, bandwidth_neff = bw_neff)
        name = fig_filename(family = :tri, problem = prob, d = d, B = B_TRI, alg = :mw)
        save_pdf(fig, name; dir = FIGS_DIR)

        # ---- TV_partition of THIS final sample, if the partition study covers the cell
        tv = part[(part.problem .== String(prob)) .& (part.d .== d) .& (part.B .== B_TRI) .&
                  (part.seed .== seed_fixed) .& (part.algorithm .== "mw") .& (part.estimator .== "fresh"), :]
        @assert nrow(tv) <= 1
        tv_fresh = nrow(tv) == 1 ? Float64(tv.TV_partition[1]) : NaN

        push!(rows, (panel = name, problem = String(prob), d = d, B = B_TRI, seed = seed_fixed,
                     payload_path = relpath(payload, _ROOT), source_sha256 = src_sha,
                     Neff_fresh = Neff_fresh, eta_cost_fresh = eta_fresh, C_total = C_total, N_fresh = N_fresh,
                     C_adaptation = C_old,
                     W1_marginal_avg_fresh = Float64(s["W1_marginal_avg_fresh"]), dlogZ_fresh = Float64(s["dlogZ_fresh"]),
                     pareto_k_fresh = Float64(s["pareto_k_fresh"]), TV_partition_fresh = tv_fresh,
                     bandwidth_neff_used = bw_neff,
                     bandwidth_h_shown_coords = join((@sprintf("%.4g", h) for h in h_coords), ";"),
                     N_display = N_DISPLAY, N_shown = 5_000,
                     footer_Neff = Neff_fresh, footer_eta = eta_fresh,
                     seed_rule = "middle of existing seed grid (n=$n_exist) == _median_seed_run",
                     coords_shown = d <= 6 ? "1:$d" : "1,2,3,$d"))
        @info "triangle written" name Neff_fresh eta_fresh C_total N_fresh tv_fresh
    end
    man = DataFrame(rows)
    CSV.write(joinpath(FIGS_DIR, "triangle_manifest.csv"), man)
    @info "triangle manifest" path = joinpath(FIGS_DIR, "triangle_manifest.csv") n = nrow(man)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end

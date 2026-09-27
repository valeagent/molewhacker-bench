# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# figs_primary.jl — PRIMARY summary figures (revision of 27 September 2026).
#
# Regenerates, from the canonical primary view
# experiments/out/primary_fresh_2026-09-27/primary_cells.csv (built by
# primary_fresh_view.jl), the thesis summary assets with their EXISTING file
# names, so that MoleWhacker is shown through the independent final sample
# drawn from the frozen mixture with the remaining budget:
#
#   summary__all__d5__B5e5__heatmap-{eta,dlogz,W1,qe025,qe160,qe500,qe840,qe975}
#   summary__all__d5__B5e3__heatmap-{eta,dlogz,W1}
#   summary__all__d5__B5e4__heatmap-{eta,dlogz,W1}
#   summary__all__d2__B5e5__heatmap-{eta,dlogz,W1}
#   summary__all__d10__B5e5__heatmap-{eta,dlogz,W1}
#   dim__all__B5e5__all__all, dim__all__B5e4__all__all
#   partition__mridges__d5__Ball__all__all, partition__mridges_spiky__d5__Ball__all__all
#
# Transforms are those of 03_plots.jl / figs_headline.jl / figs_calibration.jl
# (eta: log10; dlogZ, W1, QE: negative_log10). Differences to the archived
# generators, all via new default-preserving keyword arguments:
#   * unavailable MW cells (no within-budget final sample) render as grey
#     gaps with the legend "gray: no admissible run or no within-budget final
#     sample" (fig_summary_heatmap kwarg gap_label);
#   * MW rows carry terminated_by = "final_sample", so no early-stop star is
#     drawn (asserted on the data before plotting);
#   * colour scales are recomputed from the data (cap_dlogZ = Inf; the
#     inscribed number is always the exact median);
#   * fig_dim_grid(connect_across_missing = false): no line is drawn across
#     a missing dimension.
# The partition figures are produced by a wrapper around the figure function
# of partition_mass.jl (fig_partition) on the SAVED CSVs: MW = estimator
# "fresh" rows only (the population series is dropped), comparators =
# estimator "output" rows, i.i.d. reference from partition_calibration.csv,
# region count checked against partition_reference.csv. Nothing is
# recomputed; no sampling and no target evaluation take place.
#
# Every PDF is written together with a PNG preview (figs/png/, 4 px/pt) by
# save_pdf. figs/heatmap_cell_values.csv lists every inscribed heatmap value
# next to primary_cell_medians.csv (asserted equal).
#
# Usage:  julia --project=. -t 8 experiments/tools/figs_primary.jl
# =============================================================================

# partition_mass.jl (definitions only; its CLI main is guarded) pulls in
# fresh_draw_recompute.jl, which activates the project and loads
# ExperimentsBase. fig_partition is the figure function wrapped below.
include(joinpath(@__DIR__, "partition_mass.jl"))
using .ExperimentsBase
using DataFrames, CSV, Statistics, Printf
using CairoMakie

const PRIMARY_DIR = joinpath(OUT, "primary_fresh_2026-09-27")
const FIGS_DIR    = joinpath(PRIMARY_DIR, "figs")
const GAP_LABEL   = "gray: no admissible run or no within-budget final sample"

_str(x) = x === missing ? "" : String(string(x))

# Same admissibility filter as fig_summary_heatmap (RHAT-FAIL /
# BUDGET-VIOLATION / FAILED-SANITY), used to cross-check inscribed values.
function _heatmap_median(df::DataFrame, prob::Symbol, alg::Symbol, d::Int, B::Real, metric::Symbol)
    ss = df[(df.problem .== String(prob)) .& (df.algorithm .== String(alg)) .& (df.d .== d) .& (df.B .== Float64(B)), :]
    isempty(ss) && return NaN
    keep = [!(occursin("RHAT-FAIL", n) || occursin("BUDGET-VIOLATION", n) || occursin("FAILED-SANITY", n)) for n in ss.notes]
    v = filter(!isnan, Vector{Float64}(ss[keep, metric]))
    return isempty(v) ? NaN : median(v)
end

"""
    fig_partition_primary(part, calib, pref, problem; d = 5)

Wrapper around `fig_partition` (partition_mass.jl) for the primary figure:
drops the MW population series (keeps estimator == "fresh" for MW and
estimator == "output" for the comparators), checks the region count of the
saved rows against partition_reference.csv, and asserts that every plotted
series is contiguous in budget (no line across a missing budget).
"""
function fig_partition_primary(part::DataFrame, calib::DataFrame, pref::DataFrame, problem::Symbol; d::Int = 5)
    keep = .!((part.algorithm .== "mw") .& (part.estimator .!= "fresh"))
    sub = part[keep, :]
    @assert all((sub.estimator .== "fresh") .== (sub.algorithm .== "mw")) "MW rows must be fresh, comparators output"
    ref = pref[(pref.problem .== String(problem)) .& (pref.d .== d), :]
    @assert abs(sum(ref.p_true) - 1) < 1e-9 "reference masses of $problem d=$d do not sum to one"
    nreg = unique(sub[(sub.problem .== String(problem)) .& (sub.d .== d), :n_regions])
    @assert nreg == [nrow(ref)] "n_regions in partition_mass.csv ($nreg) differs from partition_reference.csv ($(nrow(ref)))"
    # Contiguity of every plotted series in budget.
    s_pd = sub[(sub.problem .== String(problem)) .& (sub.d .== d) .& sub.admissible, :]
    all_B = sort(unique(Float64.(s_pd.B)))
    for alg in ALG_ORDER
        s = s_pd[s_pd.algorithm .== String(alg), :]
        isempty(s) && continue
        g = combine(groupby(s, :B), :TV_partition => (v -> median(filter(isfinite, v))) => :med)
        bs = sort(Float64.(g.B[isfinite.(g.med)]))
        idx = [findfirst(==(b), all_B) for b in bs]
        @assert isempty(idx) || idx == collect(idx[1]:idx[end]) "series $alg on $problem skips a budget: $bs"
        @info "partition series" problem alg estimator = (alg === :mw ? "fresh" : "output") budgets = bs medians = round.(Float64.(g.med); digits = 4)
    end
    return fig_partition(sub, calib, problem; d = d)
end

function main()
    isdir(FIGS_DIR) || mkpath(FIGS_DIR)
    df = CSV.read(joinpath(PRIMARY_DIR, "primary_cells.csv"), DataFrame)
    df.notes = [_str(x) for x in df.notes]
    df.terminated_by = [_str(x) for x in df.terminated_by]
    df.eligibility_status = [_str(x) for x in df.eligibility_status]
    meds = CSV.read(joinpath(PRIMARY_DIR, "primary_cell_medians.csv"), DataFrame)

    # No early-stop star can appear on MW: the primary rows are the final
    # sample ("final_sample") or unavailable ("").
    mw = df[df.algorithm .== "mw", :]
    @assert all(tb -> tb in ("final_sample", ""), mw.terminated_by) "MW terminated_by must be final_sample or empty"
    @assert all((mw.terminated_by .== "final_sample") .== (mw.eligibility_status .== "eligible"))
    @assert !any(occursin.("RHAT-FAIL", mw.notes) .| occursin.("BUDGET-VIOLATION", mw.notes))
    @info "primary view loaded" rows = nrow(df) mw_rows = nrow(mw) mw_eligible = count(mw.eligibility_status .== "eligible")

    # ---- heatmaps -------------------------------------------------------------
    spec = Dict(:eta_Nlike => (:log10, "eta"), :dlogZ => (:negative_log10, "dlogz"),
                :W1_marginal_avg => (:negative_log10, "W1"),
                :QE_p025 => (:negative_log10, "qe025"), :QE_p160 => (:negative_log10, "qe160"),
                :QE_p500 => (:negative_log10, "qe500"), :QE_p840 => (:negative_log10, "qe840"),
                :QE_p975 => (:negative_log10, "qe975"))
    jobs = [(5, 5e5, [:eta_Nlike, :dlogZ, :W1_marginal_avg, :QE_p025, :QE_p160, :QE_p500, :QE_p840, :QE_p975]),
            (5, 5e3, [:eta_Nlike, :dlogZ, :W1_marginal_avg]),
            (5, 5e4, [:eta_Nlike, :dlogZ, :W1_marginal_avg]),
            (2, 5e5, [:eta_Nlike, :dlogZ, :W1_marginal_avg]),
            (10, 5e5, [:eta_Nlike, :dlogZ, :W1_marginal_avg])]
    med_col = Dict(:eta_Nlike => :eta_Nlike, :dlogZ => :abs_dlogZ, :W1_marginal_avg => :W1_marginal_avg,
                   :QE_p025 => :QE_p025, :QE_p160 => :QE_p160, :QE_p500 => :QE_p500, :QE_p840 => :QE_p840,
                   :QE_p975 => :QE_p975)
    check_rows = NamedTuple[]
    n_pdf = 0
    for (d, B, metrics) in jobs, metric in metrics
        transform, tag = spec[metric]
        fig = fig_summary_heatmap(df, metric; transform = transform, d = d, B = B, cap_dlogZ = Inf,
                                  gap_label = GAP_LABEL, gap_label_latex = false)
        name = fig_filename(family = :summary, problem = :all, d = d, B = B, extra = "heatmap-" * tag)
        save_pdf(fig, name; dir = FIGS_DIR); n_pdf += 1
        # Cross-check every inscribed cell against primary_cell_medians.csv.
        algs = metric === :dlogZ ? [:is, :ns, :mw] : collect(ALG_ORDER)
        for prob in HEATMAP_PROBLEM_ORDER, alg in algs
            d_i = ExperimentsBase._heatmap_problem_dim(prob, d)
            # dlogZ is stored as |Delta log Z| in every input (asserted in
            # primary_fresh_view.jl), so the heatmap median equals abs_dlogZ.
            v = _heatmap_median(df, prob, alg, d_i, B, metric)
            m = meds[(meds.problem .== String(prob)) .& (meds.algorithm .== String(alg)) .& (meds.d .== d_i) .& (meds.B .== Float64(B)), :]
            ref = isempty(m) ? NaN : Float64(m[1, med_col[metric]])
            grey = !isfinite(v)
            push!(check_rows, (d = d, B = B, metric = String(metric), problem = String(prob), algorithm = String(alg),
                               row_d = d_i, inscribed_median = v, primary_cell_medians = ref, grey = grey,
                               n_eligible = isempty(m) ? 0 : Int(m[1, :n_eligible])))
            if grey
                @assert !isfinite(ref) "grey cell but medians file has a value: $prob $alg d=$d_i B=$B $metric"
            else
                @assert isfinite(ref) && abs(v - ref) <= 1e-12 * max(1.0, abs(ref)) "inscribed value differs from medians file: $prob $alg d=$d_i B=$B $metric ($v vs $ref)"
            end
        end
    end
    CSV.write(joinpath(FIGS_DIR, "heatmap_cell_values.csv"), DataFrame(check_rows))
    n_grey_mw = count(r -> r.algorithm == "mw" && r.grey, check_rows)
    n_grey_mw_B5e3 = count(r -> r.algorithm == "mw" && r.grey && r.B == 5e3, check_rows)
    @info "heatmap cells cross-checked" n_cells = length(check_rows) n_grey_mw_cells = n_grey_mw n_grey_mw_cells_at_B5e3 = n_grey_mw_B5e3
    # At B = 5e3 every MW cell must be grey (no within-budget final sample).
    @assert all(r.grey for r in check_rows if r.algorithm == "mw" && r.B == 5e3)

    # ---- dimension grids (W1 vs d), no line across a missing dimension ----------
    for Bdim in (5e4, 5e5)
        fig = fig_dim_grid(df; B = Bdim, connect_across_missing = false)
        save_pdf(fig, fig_filename(family = :dim, problem = :all, d = nothing, B = Bdim, alg = :all, extra = "all");
                 dir = FIGS_DIR); n_pdf += 1
    end

    # ---- partition figures from the saved CSVs -----------------------------------
    part  = DataFrame(CSV.File(joinpath(REV, "partition_mass.csv")))
    part.admissible = [x === missing ? false : (x isa Bool ? x : lowercase(String(x)) == "true") for x in part.admissible]
    calib = DataFrame(CSV.File(joinpath(REV, "partition_calibration.csv")))
    pref  = DataFrame(CSV.File(joinpath(REV, "partition_reference.csv")))
    for prob in PART_PROBLEMS
        fig = fig_partition_primary(part, calib, pref, prob; d = 5)
        save_pdf(fig, fig_filename(family = :partition, problem = prob, d = 5, B = :all, alg = :all, extra = "all");
                 dir = FIGS_DIR); n_pdf += 1
    end

    @info "primary figures written" FIGS_DIR n_pdf
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end

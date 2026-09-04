# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# figs_acciter.jl — MoleWhacker accuracy over adaptive iterations: the
# marginal Wasserstein-1 error and the quantile errors of the pooled,
# globally reweighted population after every whacking iteration, against
# the truth reference, for the four exemplar cells of the thesis
# (mvn, funnel, mridges, shell at d = 5, B = 5e5, seed 11).
#
# Reuses the storyboard snapshot machinery: instrumented runs cache the
# per-iteration sample batches; the pooled population at iteration t is
# reconstructed with weights recomputed against the iteration-t mixture,
# exactly as the algorithm's global reweight does. Produces
#   (a) acciter__quad__d5__B5e5__mw__w1-qe975      (thesis body, Ch. 5)
#   (b) acciter__pertarget__d5__B5e5__mw__all      (results appendix)
# Per-curve values are cached as CSV inside each snapshot directory.
#
# Usage:  julia --project=. -t auto experiments/tools/figs_acciter.jl
# =============================================================================

include(joinpath(@__DIR__, "..", "scripts", "03_plots.jl"))   # helpers; main() guarded
using CairoMakie
using JLD2
using Statistics, Distributions
using Random: MersenneTwister
using DataFrames, CSV

const FIGS  = joinpath(_ROOT, "experiments", "out", "figs")
const OUT   = joinpath(_ROOT, "experiments", "out")
const STORY = joinpath(OUT, "runs_storyboard")
const B     = 5e5
const SEED  = 11

const JOBS = [
    (:mvn,     "MVN"),
    (:funnel,  "funnel"),
    (:mridges, "M-ridges"),
    (:shell,   "shell"),
]
# Wong-2011 problem colors, matching the KL-divergence figure.
const PROB_COLORS = Dict(
    :mvn     => RGBf(0.0, 0.0, 0.0),        # black
    :funnel  => RGBf(0.90, 0.62, 0.0),      # orange
    :mridges => RGBf(0.34, 0.71, 0.91),     # sky blue
    :shell   => RGBf(0.84, 0.37, 0.0),      # vermillion
)
# Metric colors for the per-target panels.
const METRIC_STYLE = [
    (:w1,    "marginal W1",  RGBf(0.0, 0.0, 0.0),      :solid),
    (:qe50,  "QE(0.5)",      RGBf(0.90, 0.62, 0.0),    :dash),
    (:qe975, "QE(0.975)",    RGBf(0.84, 0.37, 0.0),    :dot),
]

_dsv_to_matrix(dsv) = begin
    n = length(dsv.v)
    d = length(dsv.v[1])
    M = Matrix{Float64}(undef, d, n)
    @inbounds for i in 1:n, j in 1:d
        M[j, i] = dsv.v[i][j]
    end
    M
end

# Pooled population per iteration, weights recomputed against the
# iteration-t mixture (identical to the global reweight of the loop).
function _mw_history(cache_dir::AbstractString, cfg)
    files = sort(filter(f -> occursin(r"^iter_\d+\.jld2$", f), readdir(cache_dir)))
    snaps = Dict{Int,Any}()
    for f in files
        t = parse(Int, match(r"iter_(\d+)\.jld2", f).captures[1])
        jldopen(joinpath(cache_dir, f), "r") do io
            mix = io["approx_dist"]
            Z = U = nothing
            if haskey(io, "samples_mix")
                Z = _dsv_to_matrix(io["samples_mix"])
                U = _dsv_to_matrix(io["samples_user"])
            elseif haskey(io, "new_samples_mix")
                Z = _dsv_to_matrix(io["new_samples_mix"])
                U = _dsv_to_matrix(io["new_samples_user"])
            end
            snaps[t] = (mix = mix, Z = Z, U = U)
        end
    end
    log_f = build_log_f(cfg)
    Lcube = cfg.L
    stdn = Normal()
    hist = NamedTuple[]
    for t in sort(collect(keys(snaps)))
        mix_t = snaps[t].mix
        Zs = Matrix{Float64}[]
        Us = Matrix{Float64}[]
        for s in 0:t
            haskey(snaps, s) || continue
            snaps[s].Z === nothing && continue
            push!(Zs, snaps[s].Z)
            push!(Us, snaps[s].U)
        end
        isempty(Zs) && continue
        Z = hcat(Zs...)
        U = hcat(Us...)
        n = size(Z, 2)
        logw = Vector{Float64}(undef, n)
        for i in 1:n
            zi = view(Z, :, i)
            log_jac = sum(log(2 * Lcube) + logpdf(stdn, zj) for zj in zi)
            logw[i] = log_f(view(U, :, i)) + log_jac -
                      logpdf(mix_t, collect(zi))
        end
        m = maximum(logw)
        push!(hist, (t = t, U = U, w = exp.(logw .- m)))
    end
    return hist
end

# Coordinate-averaged absolute quantile error of an equal-weight
# multinomial resample of (U, w) against truth.quantiles — the same
# construction as quantile_errors in metrics.jl.
const _QLEVELS = [(0.025, 1), (0.16, 2), (0.5, 3), (0.84, 4), (0.975, 5)]
function _qe_of(U::AbstractMatrix, w::AbstractVector, truth;
                N::Int = 10_000, rng = MersenneTwister(20260904))
    d, n = size(U)
    cp = cumsum(w ./ sum(w))
    cp[end] = 1.0
    idx = [clamp(searchsortedfirst(cp, r), 1, n) for r in rand(rng, N)]
    out = Dict{Float64,Float64}()
    for (alpha, col) in _QLEVELS
        acc = 0.0
        for j in 1:d
            xj = sort(Float64[U[j, i] for i in idx])
            q_emp = xj[clamp(round(Int, alpha * N), 1, N)]
            acc += abs(q_emp - truth.quantiles[j, col])
        end
        out[alpha] = acc / d
    end
    return out
end

function _acc_curves(prob::Symbol)
    cache = joinpath(STORY, "$(prob)_d5_seed$(SEED)_acc")
    csv = joinpath(cache, "mw_acciter_curve.csv")
    if isfile(csv)
        df = CSV.read(csv, DataFrame)
        return df
    end
    cfg = _make_cfg(prob, 5)
    truth = _load_truth(joinpath(OUT, "truth"), prob, 5)
    truth === nothing && error("truth missing for $prob d=5")
    if !isdir(cache) || isempty(filter(f -> endswith(f, ".jld2"), readdir(cache)))
        mkpath(cache)
        @info "instrumented accuracy run starting" prob
        run_mw(cfg, B, SEED; cache_dir = cache, save_iteration_samples = true)
        @info "instrumented accuracy run finished" prob
    end
    hist = _mw_history(cache, cfg)
    rng = MersenneTwister(20260830)
    rows = NamedTuple[]
    for h in hist
        w1 = ExperimentsBase._w1_matrix(h.U, h.w, truth, 10_000, rng)
        qe = _qe_of(h.U, h.w, truth)
        push!(rows, (t = h.t, w1 = w1, qe025 = qe[0.025], qe160 = qe[0.16],
                     qe500 = qe[0.5], qe840 = qe[0.84], qe975 = qe[0.975]))
    end
    df = DataFrame(rows)
    CSV.write(csv, df)
    return df
end

function main()
    curves = Dict{Symbol,DataFrame}()
    for (p, _) in JOBS
        curves[p] = _acc_curves(p)
        @info "accuracy curve" p n = nrow(curves[p]) w1_first = curves[p].w1[1] w1_last = curves[p].w1[end]
    end

    # (a) Body figure: W1(t) and QE975(t), four problem curves each.
    set_pub_theme!(class = :narrow)
    fig = Figure(size = (760, 280))
    ax1 = Axis(fig[1, 1];
        xlabel = "adaptive iteration t",
        ylabel = L"\overline{W}_1(t)",
        yscale = log10)
    ax2 = Axis(fig[1, 2];
        xlabel = "adaptive iteration t",
        ylabel = L"\mathrm{QE}(0.975)(t)",
        yscale = log10)
    for (p, lab) in JOBS
        df = curves[p]
        lines!(ax1, df.t, max.(df.w1, 1e-4); color = PROB_COLORS[p],
               linewidth = 1.8, label = lab)
        scatter!(ax1, df.t, max.(df.w1, 1e-4); color = PROB_COLORS[p], markersize = 5)
        lines!(ax2, df.t, max.(df.qe975, 1e-4); color = PROB_COLORS[p],
               linewidth = 1.8, label = lab)
        scatter!(ax2, df.t, max.(df.qe975, 1e-4); color = PROB_COLORS[p], markersize = 5)
    end
    fig[1, 3] = Legend(fig, ax1; framevisible = false)
    save_pdf(fig, "acciter__quad__d5__B5e5__mw__w1-qe975"; dir = FIGS)

    # (b) Appendix figure: per-target panels, three accuracy metrics each.
    fig2 = Figure(size = (760, 540))
    axs = Dict{Symbol,Axis}()
    positions = Dict(:mvn => (1, 1), :funnel => (1, 2),
                     :mridges => (2, 1), :shell => (2, 2))
    for (p, lab) in JOBS
        r, c = positions[p]
        ax = Axis(fig2[r, c];
            title = lab,
            xlabel = r == 2 ? "adaptive iteration t" : "",
            ylabel = c == 1 ? "absolute error" : "",
            yscale = log10)
        df = curves[p]
        for (col, mlab, mcol, mstyle) in METRIC_STYLE
            y = max.(df[!, col === :w1 ? :w1 : (col === :qe50 ? :qe500 : :qe975)], 1e-4)
            lines!(ax, df.t, y; color = mcol, linewidth = 1.6,
                   linestyle = mstyle, label = mlab)
        end
        axs[p] = ax
    end
    fig2[1:2, 3] = Legend(fig2, axs[:mvn]; framevisible = false)
    save_pdf(fig2, "acciter__pertarget__d5__B5e5__mw__all"; dir = FIGS)

    @info "accuracy-over-iteration figures done" FIGS
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end

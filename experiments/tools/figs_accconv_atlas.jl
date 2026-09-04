# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# figs_accconv_atlas.jl — the complete per-problem accuracy-over-budget
# atlas: running marginal Wasserstein-1 error against consumed
# likelihood-equivalents, all five algorithms overlaid (one representative
# run each), for every benchmark problem at its headline setting. The
# funnel and M-ridges overlays of the thesis body/appendix are produced by
# figs_accconv.jl; this script completes the set (mvn, banana, shell,
# spiky M-ridges, eggbox). MoleWhacker's curve is reconstructed from
# instrumented snapshot runs (cached under out/runs_storyboard/*_acc/,
# created here when missing) and cached as CSV.
#
# Usage:  julia --project=. -t auto experiments/tools/figs_accconv_atlas.jl
# =============================================================================

include(joinpath(@__DIR__, "..", "scripts", "03_plots.jl"))   # helpers; main() guarded
using JLD2
using Statistics, Distributions
using Random: MersenneTwister
using DataFrames, CSV

const FIGS  = joinpath(_ROOT, "experiments", "out", "figs")
const OUT   = joinpath(_ROOT, "experiments", "out")
const STORY = joinpath(OUT, "runs_storyboard")
const B     = 5e5
const SEED  = 11

# (problem, dimension) cells to complete the atlas.
const JOBS = [(:mvn, 5), (:banana, 5), (:shell, 5),
              (:mridges_spiky, 5), (:eggbox, 2)]

function _cell_runs(prob::Symbol, alg::Symbol, d::Int, Bb::Real)
    runs = MethodResult[]
    for seed in SEED_GRID
        dir = cell_dir(OUT, prob, alg, d, Bb, seed)
        isdir(dir) || continue
        isfile(joinpath(dir, "result.h5")) || continue
        mr = try
            load_method_result(dir)
        catch
            continue
        end
        size(mr.samples, 2) > 1 || continue
        push!(runs, mr)
    end
    return runs
end

_final_eta(mr) = begin
    nf = neff(mr)
    (isfinite(nf) && mr.Nlike_used > 0) ? nf / mr.Nlike_used : NaN
end

function _representative(runs::Vector{MethodResult})
    etas = [_final_eta(mr) for mr in runs]
    ok = findall(isfinite, etas)
    isempty(ok) && return nothing
    med = median(etas[ok])
    return runs[ok[argmin(abs.(etas[ok] .- med))]]
end

_dsv_to_matrix(dsv) = begin
    n = length(dsv.v)
    d = length(dsv.v[1])
    M = Matrix{Float64}(undef, d, n)
    @inbounds for i in 1:n, j in 1:d
        M[j, i] = dsv.v[i][j]
    end
    M
end

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

# MoleWhacker accuracy-over-cost curve with CSV caching in the snapshot dir.
function _mw_w1_curve(prob::Symbol, d::Int)
    cache = joinpath(STORY, "$(prob)_d$(d)_seed$(SEED)_acc")
    csv = joinpath(cache, "mw_w1_curve.csv")
    if isfile(csv)
        df = CSV.read(csv, DataFrame)
        return Vector{Float64}(df.cost), Vector{Float64}(df.w1)
    end
    cfg = _make_cfg(prob, d)
    truth = _load_truth(joinpath(OUT, "truth"), prob, d)
    truth === nothing && error("truth missing for $prob d=$d")
    local mr_mw
    if !isdir(cache) || isempty(filter(f -> endswith(f, ".jld2"), readdir(cache)))
        mkpath(cache)
        @info "instrumented accuracy run starting" prob d
        mr_mw = run_mw(cfg, B, SEED; cache_dir = cache,
                       save_iteration_samples = true)
    else
        mr_mw = run_mw(cfg, B, SEED)   # snapshots exist; rerun for iter_log
    end
    il = get(mr_mw.extras, :iter_log, NamedTuple[])
    cost_by_t = Dict(Int(e.iter) => Float64(e.cum_cost) for e in il)
    hist = _mw_history(cache, cfg)
    rng = MersenneTwister(20260830)
    xs = Float64[]
    ys = Float64[]
    for h in hist
        haskey(cost_by_t, h.t) || continue
        v = ExperimentsBase._w1_matrix(h.U, h.w, truth, 10_000, rng)
        isfinite(v) || continue
        push!(xs, cost_by_t[h.t])
        push!(ys, v)
    end
    CSV.write(csv, DataFrame(cost = xs, w1 = ys))
    return xs, ys
end

function main()
    for (prob, d) in JOBS
        name = fig_filename(family = :accconv, problem = prob, d = d,
                            B = B, alg = :all, extra = "W1")
        truth = _load_truth(joinpath(OUT, "truth"), prob, d)
        truth === nothing && (@warn "truth missing — skipped" prob; continue)
        mwx, mwy = _mw_w1_curve(prob, d)
        runs_by_alg = Pair{Symbol,MethodResult}[]
        for alg in (:is, :mh, :nuts, :ns)
            runs = _cell_runs(prob, alg, d, B)
            isempty(runs) && continue
            mr = _representative(runs)
            mr === nothing && continue
            push!(runs_by_alg, alg => mr)
        end
        fig = fig_accuracy_cell(runs_by_alg; problem = prob, d = d, B = B,
                                   truth = truth, mw_curve = (mwx, mwy))
        save_pdf(fig, name; dir = FIGS)
        @info "accuracy overlay done" prob
    end
    @info "accconv atlas done" FIGS
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end

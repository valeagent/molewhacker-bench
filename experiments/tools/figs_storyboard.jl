# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# =============================================================================
# figs_storyboard.jl — thesis Chapter 5 iteration figures:
#
#   (1) Regenerate the seven HEADLINE-setting iterconv figures with the
#       decade-tick fix (shell's y-axis previously showed 10^-4.72-style
#       fractional ticks).
#   (2) Render the 2x2 chapter-5 figure (MVN, funnel, M-ridges, shell at
#       d = 5, B = 5e5): iterconv__quad__d5__B5e5__mw__eta.
#   (3) Instrumented storyboard runs on M-ridges at d = 2 and d = 5
#       (seed 11, B = 5e5) with per-iteration snapshots, then render the
#       whacking-loop storyboards story__mridges__d{2,5}__B5e5__mw__loop.
#       These runs are ILLUSTRATIVE ONLY and write nothing into the
#       benchmark's runs/ tree or cells.csv.
# =============================================================================

include(joinpath(@__DIR__, "..", "scripts", "03_plots.jl"))   # helpers; main() guarded
using JLD2
using Statistics, Distributions

const FIGS = joinpath(_ROOT, "experiments", "out", "figs")
const RUNS = joinpath(_ROOT, "experiments", "out", "runs")
const STORY = joinpath(_ROOT, "experiments", "out", "runs_storyboard")

function _load_cell_runs(prob::Symbol, d::Int, B::Real)
    runs = MethodResult[]
    for seed in SEED_GRID
        dir = cell_dir(joinpath(_ROOT, "experiments", "out"), prob, :mw, d, B, seed)
        isdir(dir) || continue
        isfile(joinpath(dir, "result.h5")) || continue
        try
            push!(runs, load_method_result(dir))
        catch err
            @warn "load failed" prob d B seed err
        end
    end
    return runs
end

# --- (1) headline iterconv figures with tick fix ---------------------------
function part1_headline_iterconv()
    headline = [(:mvn, 5), (:banana, 5), (:funnel, 5), (:mridges, 5),
                (:shell, 5), (:mridges_spiky, 5), (:eggbox, 2)]
    B = 5e5
    for (prob, d) in headline
        runs = _load_cell_runs(prob, d, B)
        isempty(runs) && (@warn "no runs" prob d; continue)
        fig = fig_mw_itercurves(runs; problem = prob, d = d, B = B,
                                class = :narrow)
        save_pdf(fig, fig_filename(family = :iterconv, problem = prob,
                   d = d, B = B, alg = :mw, extra = "eta"); dir = FIGS)
    end
end

# --- (2) chapter-5 quad --------------------------------------------------
function part2_quad()
    B = 5e5
    panels = NamedTuple[]
    for (prob, d) in [(:mvn, 5), (:funnel, 5), (:mridges, 5), (:shell, 5)]
        runs = _load_cell_runs(prob, d, B)
        isempty(runs) && error("no runs for quad panel $prob")
        push!(panels, (problem = prob, d = d, runs = runs))
    end
    fig = fig_mw_itercurves_quad(panels; B = B)
    save_pdf(fig, fig_filename(family = :iterconv, problem = :quad,
               d = 5, B = B, alg = :mw, extra = "eta"); dir = FIGS)
end

# --- (3) storyboard -------------------------------------------------------

_dsv_to_matrix(dsv) = begin
    n = length(dsv.v)
    d = length(dsv.v[1])
    M = Matrix{Float64}(undef, d, n)
    @inbounds for i in 1:n, j in 1:d
        M[j, i] = dsv.v[i][j]
    end
    M
end

# Reconstruct the accumulated population at each requested iteration.
# iter_000 stores the full initial population; iter_t (t > 0) stores only
# the NEW batch drawn at t. Snapshot weights are stale for the pooled
# view (each batch was weighted against the mixture current at its own
# iteration), so the pooled weights are recomputed against the
# iteration-t mixture exactly as the algorithm's global reweight does:
# w ∝ p_z(z) / q_t(z) in the transformed space, with
# log p_z(z) = log f(θ(z)) + Σ_j [log(2L) + log φ(z_j)].
function _storyboard_frames(cache_dir::AbstractString, want_ts::Vector{Int},
                              cfg)
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
    ts_all = sort(collect(keys(snaps)))
    @info "storyboard snapshots available" cache_dir ts_all

    log_f = build_log_f(cfg)               # raw closure, no counter
    Lcube = cfg.L
    stdn = Normal()

    frames = NamedTuple[]
    for t in want_ts
        haskey(snaps, t) || (@warn "snapshot missing" t; continue)
        mix_t = snaps[t].mix
        centers = hcat([Vector{Float64}(mean(c)) for c in mix_t.components]...)
        cw = Vector{Float64}(probs(mix_t))

        # Pooled population: batches 0..t with stored positions.
        Zs = Matrix{Float64}[]
        Us = Matrix{Float64}[]
        for s in 0:t
            haskey(snaps, s) || continue
            snaps[s].Z === nothing && continue
            push!(Zs, snaps[s].Z)
            push!(Us, snaps[s].U)
        end
        if isempty(Zs)
            push!(frames, (t = t, K = size(centers, 2), centers_z = centers,
                center_weights = cw,
                samples_user = zeros(size(centers, 1), 0),
                sample_weights = Float64[]))
            continue
        end
        Z = hcat(Zs...)
        U = hcat(Us...)
        n = size(Z, 2)
        logw = Vector{Float64}(undef, n)
        for i in 1:n
            zi = view(Z, :, i)
            log_jac = sum(log(2 * Lcube) + logpdf(stdn, zj) for zj in zi)
            log_pz = log_f(view(U, :, i)) + log_jac
            log_qz = logpdf(mix_t, collect(zi))
            logw[i] = log_pz - log_qz
        end
        m = maximum(logw)
        w = exp.(logw .- m)
        push!(frames, (t = t, K = size(centers, 2), centers_z = centers,
            center_weights = cw, samples_user = U, sample_weights = w))
    end
    return frames
end

function part3_storyboard(d::Int; seed::Int = 11, B::Real = 5e5,
                            want_ts::Vector{Int} = [0, 1, 3, 20],
                            n_seed_init::Union{Int,Nothing} = nothing,
                            tag::AbstractString = "loop")
    variant = n_seed_init === nothing ? "" : "_ns$(n_seed_init)"
    cache = joinpath(STORY, "mridges_d$(d)_seed$(seed)$(variant)")
    if !isdir(cache) || isempty(filter(f -> endswith(f, ".jld2"), readdir(cache)))
        mkpath(cache)
        @info "storyboard run starting" d seed B cache n_seed_init
        cfg = make_config_mridges(d = d)
        params = n_seed_init === nothing ? ExperimentsBase.MWParams() :
            ExperimentsBase.MWParams(n_seed = n_seed_init)
        run_mw(cfg, B, seed; params = params, cache_dir = cache,
               save_iteration_samples = true)
        @info "storyboard run finished" d
    else
        @info "storyboard snapshots already present — reusing" cache
    end
    frames = _storyboard_frames(cache, want_ts, make_config_mridges(d = d))
    isempty(frames) && (@warn "no frames for d=$d"; return)
    truth = _load_truth(joinpath(_ROOT, "experiments", "out", "truth"), :mridges, d)
    truth === nothing && error("truth missing for mridges d=$d")
    fig = fig_mw_storyboard(frames, truth.samples; L = 10.0, coords = (1, 2))
    save_pdf(fig, fig_filename(family = :story, problem = :mridges,
               d = d, B = B, alg = :mw, extra = tag); dir = FIGS)
end

_quad_pdf = joinpath(FIGS, fig_filename(family = :iterconv, problem = :quad,
    d = 5, B = 5e5, alg = :mw, extra = "eta") * ".pdf")
if isfile(_quad_pdf)
    @info "parts 1-2 already rendered — skipping" _quad_pdf
else
    part1_headline_iterconv()
    part2_quad()
end
part3_storyboard(2)
part3_storyboard(5)
# Weak-initialization variant (4 Sobol seeds instead of 64): the default
# initialization already covers all four d = 2 arms, so the loop's mode
# DISCOVERY is invisible; the weakened init makes the whack-repair
# mechanic visible. Illustrative configuration only — clearly labeled.
part3_storyboard(2; n_seed_init = 4, tag = "loop-weakinit")
@info "chapter-5 figures done"

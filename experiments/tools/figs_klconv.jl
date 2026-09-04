# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# figs_klconv.jl — per-iteration reverse KL of the cube-restricted mixture
# against the normalized target (thesis Fig. "Measured proposal divergence
# over iterations", Ch. 5). For the four exemplar cells (mvn, funnel,
# mridges, shell at d = 5, B = 5e5, seed 11) the script reruns MoleWhacker
# with per-iteration mixture snapshots enabled (cached under
# experiments/out/runs_klconv/, ~a few MB), then evaluates the same
# estimator as `kl_cube_mw` in metrics.jl on every snapshot.
#
# Usage:  julia --project=. -t auto experiments/tools/figs_klconv.jl
# =============================================================================

include(joinpath(@__DIR__, "..", "scripts", "03_plots.jl"))   # helpers; main() guarded
using CairoMakie
using JLD2
using Statistics, Distributions
using Random: Xoshiro

const OUT  = joinpath(_ROOT, "experiments", "out")
const FIGS = joinpath(OUT, "figs")

# Reverse KL of the cube-restricted mixture at iteration t against the
# normalized cube posterior — identical mathematics to kl_cube_mw in
# metrics.jl, but taking the mixture snapshot directly.
function kl_of_mix(mix, truth, cfg; N_eval::Integer = 10_000, seed::Integer = 54321)
    rng = Xoshiro(seed)
    d = cfg.d
    Lcube = cfg.L
    z = rand(rng, mix, N_eval)
    log_norm_p = truth.logZ + d * log(2 * Lcube)
    log_f = build_log_f(cfg)
    stdn = Normal()
    kl_terms = Float64[]
    sizehint!(kl_terms, N_eval)
    n_bad = 0
    θ = Vector{Float64}(undef, d)
    for i in 1:N_eval
        log_jac = 0.0
        for j in 1:d
            zj = z[j, i]
            θ[j] = 2 * Lcube * cdf(stdn, zj) - Lcube
            log_jac += log(2 * Lcube) + logpdf(stdn, zj)
        end
        log_q_z = try
            logpdf(mix, collect(view(z, :, i)))
        catch
            NaN
        end
        log_q_user = log_q_z - log_jac
        log_p_user = log_f(θ) - log_norm_p
        if !isfinite(log_q_user) || !isfinite(log_p_user)
            n_bad += 1
        else
            push!(kl_terms, log_q_user - log_p_user)
        end
    end
    (isempty(kl_terms) || n_bad > 0.01 * N_eval) && return NaN
    return mean(kl_terms)
end

const KL_JOBS = [
    (:mvn,     () -> make_config_mvn(d = 5),     "MVN"),
    (:funnel,  () -> make_config_funnel(d = 5),  "funnel"),
    (:mridges, () -> make_config_mridges(d = 5), "M-ridges"),
    (:shell,   () -> make_config_shell(d = 5),   "shell"),
]
# Wong-2011 colors keyed by problem (these are problem curves, not the
# algorithm identities used elsewhere).
const KL_COLORS = Dict(
    :mvn     => RGBf(0.0, 0.0, 0.0),        # black
    :funnel  => RGBf(0.90, 0.62, 0.0),      # orange
    :mridges => RGBf(0.34, 0.71, 0.91),     # sky blue
    :shell   => RGBf(0.84, 0.37, 0.0),      # vermillion
)

function main(; B::Real = 5e5, seed::Int = 11)
    curves = Dict{Symbol,Vector{Tuple{Int,Float64}}}()
    for (p, mk, _) in KL_JOBS
        cfg = mk()
        cache = joinpath(OUT, "runs_klconv", "$(p)_d5_seed$(seed)")
        if !isdir(cache) || isempty(filter(f -> endswith(f, ".jld2"), readdir(cache)))
            mkpath(cache)
            @info "instrumented KL run starting" p B seed
            run_mw(cfg, B, seed; cache_dir = cache, save_iteration_samples = false)
            @info "instrumented KL run finished" p
        else
            @info "KL snapshots already present — reusing" p
        end
        truth = _load_truth(joinpath(OUT, "truth"), p, 5)
        truth === nothing && error("truth missing for $p d=5")
        pts = Tuple{Int,Float64}[]
        for f in sort(filter(f -> occursin(r"^iter_\d+\.jld2$", f), readdir(cache)))
            it, mix = jldopen(joinpath(cache, f), "r") do io
                (io["iter"], io["approx_dist"])
            end
            kl = kl_of_mix(mix, truth, cfg)
            isfinite(kl) && push!(pts, (Int(it), kl))
        end
        sort!(pts, by = first)
        curves[p] = pts
        @info "KL curve" p n = length(pts) kl_first = (isempty(pts) ? NaN : pts[1][2]) kl_last = (isempty(pts) ? NaN : pts[end][2])
    end

    set_pub_theme!(class = :narrow)
    fig = Figure(size = (460, 280))
    ax = Axis(fig[1, 1];
        xlabel = "adaptive iteration t",
        ylabel = L"\mathrm{KL}\!\left(q_t^{\mathrm{cube}} \,\Vert\, p\right)\;\;[\mathrm{nats}]",
        yscale = log10)
    floor_kl = 1e-3
    for (p, _, lab) in KL_JOBS
        pts = get(curves, p, Tuple{Int,Float64}[])
        isempty(pts) && continue
        xs = first.(pts)
        ys = [max(kl, floor_kl) for kl in last.(pts)]
        lines!(ax, xs, ys; color = KL_COLORS[p], linewidth = 1.8, label = lab)
        scatter!(ax, xs, ys; color = KL_COLORS[p], markersize = 5)
    end
    fig[1, 2] = Legend(fig, ax; framevisible = false)
    save_pdf(fig, "klconv__quad__d5__B5e5__mw__klcube"; dir = FIGS)
    @info "KL(t) figure saved" FIGS
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end

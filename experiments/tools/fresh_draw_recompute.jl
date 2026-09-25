# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# fresh_draw_recompute.jl - frozen-proposal ("fresh-draw") estimator for the
# archived MoleWhacker runs (revision of 25 September 2026).
# =============================================================================
#
# Background. The archived MoleWhacker output of every benchmark cell is the
# accumulated population of all whacking iterations, weighted against the
# *final* mixture. Batches were drawn under different mixtures, so those
# weights are not a valid self-normalized importance sampling estimator in
# general (review finding TH-001). This tool computes, for every archived MW
# run, the estimator that *is* valid conditional on the frozen final
# mixture: N_fresh independent draws from the stored final mixture, weighted
# by the exact target density. The draws cost one likelihood-equivalent unit
# each and are charged on top of the archived cost.
#
#   N_fresh              = max(0, floor(B - C_old))      (full remaining budget)
#   C_total              = C_old + N_fresh
#   logw_i               = log f_z(z_i) - log q_T(z_i)     (absolute; no shift)
#   logZ_fresh           = logsumexp(logw) - log N_fresh
#   Kish ESS             = exp(2 logsumexp(logw) - logsumexp(2 logw))
#   eta_cost_fresh       = ESS / C_total       (effective samples per charged unit)
#   eta_proposal_fresh   = ESS / N_fresh       (SNIS efficiency of the frozen mixture)
#
# Coordinates. The stored mixture (`extras[:mixture]`) lives in BAT's
# PriorToGaussian space of the cube-prior posterior. f_z is the transformed
# posterior density `logdensityof(pstr, z)` (target * cube prior * Jacobian),
# whose integral is the cube-normalized evidence the truth files store. For
# the uniform cube prior the inverse transform is theta_j = 2 L Phi(z_j) - L
# and log f_z = log f(theta) + sum_j log phi(z_j); both identities were
# verified against BAT's own transform on real cells (max deviation 0.0) and
# are re-checked by `--test`.
#
# Archived files are never modified. Outputs go to
#   experiments/out/revision_2026-09-25/fresh_runs/<cell>.jld2  (per run)
#   experiments/out/revision_2026-09-25/cells_fresh.csv          (--aggregate)
#   experiments/out/revision_2026-09-25/fresh_manifest.csv       (--aggregate)
#
# Usage:
#   julia --project=. -t 8 experiments/tools/fresh_draw_recompute.jl --test
#   julia --project=. -t 8 experiments/tools/fresh_draw_recompute.jl --subset headline
#   julia --project=. -t 8 experiments/tools/fresh_draw_recompute.jl --subset top   # all d at B = 5e5
#   julia --project=. -t 8 experiments/tools/fresh_draw_recompute.jl --subset mid   # B = 5e4
#   julia --project=. -t 8 experiments/tools/fresh_draw_recompute.jl --subset low   # B = 5e3
#   julia --project=. -t 8 experiments/tools/fresh_draw_recompute.jl --aggregate
# Options: --nfresh-cap N  (fixed within-budget count min(N, remaining); status "capped")
#          --limit K       (process at most K runs; for timing)
#          --chunk M       (draws per chunk, default 8192)
# =============================================================================

using Pkg
const _ROOT = abspath(joinpath(@__DIR__, "..", ".."))
if Base.active_project() != joinpath(_ROOT, "Project.toml")
    Pkg.activate(_ROOT)
end
include(joinpath(_ROOT, "experiments", "src", "ExperimentsBase.jl"))
using .ExperimentsBase
using BAT, Distributions, LinearAlgebra, Random, Statistics, Printf, JLD2, SHA, Dates
using DataFrames, CSV
using BAT: bat_transform, PriorToGaussian
using DensityInterface: logdensityof

const OUT = joinpath(_ROOT, "experiments", "out")
const REV = joinpath(OUT, "revision_2026-09-25")
const FRESH_DIR = joinpath(REV, "fresh_runs")
const ESTIMATOR_VERSION = "fresh_v1_2026-09-25"
const PROBLEMS = (:mvn, :banana, :funnel, :mridges, :shell, :mridges_spiky, :eggbox)
const BUDGETS = (5e3, 5e4, 5e5)

_logsumexp(v::AbstractVector{<:Real}) = begin
    m = maximum(v)
    isfinite(m) || return m
    m + log(sum(exp.(v .- m)))
end
_normcdf(z) = cdf(Normal(), z)
_normlogpdf(z) = logpdf(Normal(), z)

function _make_cfg(problem::Symbol, d::Int)
    if problem === :mvn;     return make_config_mvn(d = d)
    elseif problem === :banana; return make_config_banana(d = d)
    elseif problem === :funnel; return make_config_funnel(d = d)
    elseif problem === :mridges; return make_config_mridges(d = d)
    elseif problem === :shell;   return make_config_shell(d = d)
    elseif problem === :mridges_spiky
        return make_config_mridges_spiky(d = d, M = 4, σ_spike = 0.1, kernel = :gaussian)
    elseif problem === :eggbox; return make_config_eggbox(d = d)
    end
    error("unknown problem $problem")
end

# Documented, scheduling-independent per-run RNG seed.
function rng_seed_for(problem::Symbol, d::Int, B::Real, seed::Int)
    pi = findfirst(==(problem), PROBLEMS)
    bi = findfirst(==(B), BUDGETS)
    return 20260925 * 10_000 + pi * 1_000_000 + d * 10_000 + bi * 1_000 + seed
end

# SHA-256 over the mixture parameters (weights, means, covariances) in a
# fixed order; identifies the frozen proposal.
function proposal_hash(mix::MixtureModel)
    io = IOBuffer()
    write(io, Int64(length(mix.components)))
    for (k, c) in enumerate(mix.components)
        write(io, Float64(mix.prior.p[k]))
        write(io, Vector{Float64}(mean(c)))
        write(io, Matrix{Float64}(cov(c)))
    end
    return bytes2hex(sha256(take!(io)))
end

# Generalized-Pareto shape of the largest weights (Zhang & Stephens 2009),
# as in the companion repository's 71_is_diagnostics.jl / 81_extension_fresh.jl.
function gpd_khat(x::AbstractVector{<:Real})
    x = sort(x); n = length(x)
    n < 5 && return NaN
    prior = 3.0
    m = 30 + floor(Int, sqrt(n))
    q = x[max(1, floor(Int, n / 4 + 0.5))]
    θs = [1 / x[end] + (1 - sqrt(m / (j - 0.5))) / prior / q for j in 1:m]
    lx(b) = (k = -mean(log1p.(-b .* x)); log(b / k) + k - 1)
    l = [n * lx(b) for b in θs]
    w = exp.(l .- maximum(l)); w ./= sum(w)
    b = sum(θs .* w)
    return mean(log1p.(-b .* x))
end
function pareto_k(weights::AbstractVector{<:Real})
    w = sort(weights; rev = true); n = length(w)
    M = min(ceil(Int, 0.2n), ceil(Int, 3sqrt(n)))
    M + 1 > n && return NaN
    return gpd_khat(w[1:M] .- w[M + 1])
end

"""
    FastMixture(mix)

Allocation-free evaluator of the *full* mixture log-density: per-component
lower Cholesky factors and log-normalizers are precomputed once;
`logpdf_fast(fm, z, buf)` forward-substitutes into the per-thread buffer
`buf` and log-sum-exps over all K components. Checked against
`Distributions.logpdf(mix, z)` on every run (spot check) and in `--test`.
"""
struct FastMixture
    d::Int
    K::Int
    mu::Vector{Vector{Float64}}
    Lf::Vector{LowerTriangular{Float64,Matrix{Float64}}}
    c::Vector{Float64}       # log w_k - d/2 log(2 pi) - sum log diag(L_k)
end
function FastMixture(mix::MixtureModel)
    d = length(mix); K = length(mix.components)
    mu = Vector{Vector{Float64}}(undef, K); Lf = Vector{LowerTriangular{Float64,Matrix{Float64}}}(undef, K); c = zeros(K)
    for k in 1:K
        comp = mix.components[k]
        mu[k] = Vector{Float64}(mean(comp))
        S = Symmetric(Matrix{Float64}(cov(comp)))
        F = cholesky(S)
        Lf[k] = LowerTriangular(Matrix(F.L))
        c[k] = log(mix.prior.p[k]) - 0.5d * log(2π) - sum(log, diag(Lf[k]))
    end
    return FastMixture(d, K, mu, Lf, c)
end
@inline function logpdf_fast(fm::FastMixture, z::AbstractVector{<:Real}, buf::Vector{Float64})
    m = -Inf; s = 0.0
    d = fm.d
    @inbounds for k in 1:fm.K
        mu = fm.mu[k]; L = fm.Lf[k].data
        # forward substitution: L y = z - mu
        q = 0.0
        for i in 1:d
            acc = z[i] - mu[i]
            for j in 1:(i - 1)
                acc -= L[i, j] * buf[j]
            end
            buf[i] = acc / L[i, i]
            q += buf[i] * buf[i]
        end
        v = fm.c[k] - 0.5q
        if v > m
            s = s * exp(m - v) + 1.0; m = v
        else
            s += exp(v - m)
        end
    end
    return m + log(s)
end

"""
    fresh_core(mix, logp_fun, nfresh, rng; chunk, fast_target=nothing) -> (Z, logq, logp)

Draw `nfresh` points from `mix` with the sequential `rng` (chunked, so the
draw sequence is independent of the thread count) and evaluate the full
mixture density and the target on every point (thread-parallel within a
chunk; order-independent). `logp_fun(z::Vector)` is the generic target
path (BAT's `logdensityof(pstr, z)` in production checks, closures in the
tests). If `fast_target = (counter, L)` is given, the target is evaluated
through the verified cube identity `log f_z = log f(theta) + sum log phi(z)`
with `theta = 2 L Phi(z) - L`, calling the counter-wrapped `log f` directly
(one charged unit per draw) and avoiding BAT's per-call allocations.
"""
function fresh_core(mix::MixtureModel, logp_fun, nfresh::Int, rng::AbstractRNG;
                    chunk::Int = 8192, fast_target = nothing)
    d = length(mix)
    fm = FastMixture(mix)
    Z = Matrix{Float64}(undef, d, nfresh)
    logq = Vector{Float64}(undef, nfresh)
    logp = Vector{Float64}(undef, nfresh)
    nt = Threads.nthreads()
    bufs = [zeros(d) for _ in 1:nt]
    thetas = [zeros(d) for _ in 1:nt]
    i0 = 1
    while i0 <= nfresh
        i1 = min(nfresh, i0 + chunk - 1)
        Zc = rand(rng, mix, i1 - i0 + 1)
        Z[:, i0:i1] .= Zc
        Threads.@threads :static for i in i0:i1
            tid = Threads.threadid()
            @inbounds begin
                z = view(Z, :, i)
                logq[i] = logpdf_fast(fm, z, bufs[tid])
                if fast_target === nothing
                    logp[i] = logp_fun(collect(z))
                else
                    counter, L = fast_target
                    th = thetas[tid]; lphi = 0.0
                    for j in 1:d
                        th[j] = 2L * _normcdf(z[j]) - L
                        lphi += _normlogpdf(z[j])
                    end
                    logp[i] = counter(th) + lphi
                end
            end
        end
        i0 = i1 + 1
    end
    return Z, logq, logp
end

"""
    weight_summary(logw) -> NamedTuple

Absolute-weight summaries: logZ, its delta-method standard error, Kish ESS,
maximum normalized weight, Pareto shape of the normalized weights, and
counts of zero / non-finite weights. Zero weights (logw = -Inf) stay in N.
"""
function weight_summary(logw::AbstractVector{<:Real})
    N = length(logw)
    n_nan = count(isnan, logw)
    n_zero = count(==(-Inf), logw)
    n_posinf = count(==(Inf), logw)
    (n_nan > 0 || n_posinf > 0) && return (; logZ = NaN, logZ_se = NaN, ess = NaN, max_w = NaN,
        pareto_k = NaN, n_zero, n_nan, n_posinf, status = "error_nonfinite")
    lse = _logsumexp(logw)
    logZ = lse - log(N)
    nw = exp.(logw .- lse)                 # normalized weights, sum 1
    ess = 1 / sum(nw .^ 2)
    lse2 = _logsumexp(2 .* logw)
    var_w = exp(lse2 - log(N)) - exp(2 * logZ)
    se = (var_w > 0 && N > 1) ? sqrt(var_w) / (sqrt(N) * exp(logZ)) : NaN
    return (; logZ, logZ_se = se, ess, max_w = maximum(nw), pareto_k = pareto_k(nw),
        n_zero, n_nan, n_posinf, status = "ok")
end

function _cell_name(mr::MethodResult)
    return basename(cell_dir(OUT, mr.problem, mr.algorithm, mr.d, mr.B, mr.seed))
end

"""
    process_run(dir; nfresh_cap, chunk) -> Dict

Full frozen-proposal estimate for one archived MW run. Writes
`<FRESH_DIR>/<cell>.jld2` and returns the summary row.
"""
function process_run(dir::AbstractString; nfresh_cap::Union{Nothing,Int} = nothing, chunk::Int = 8192)
    mr = load_method_result(dir)
    mr.algorithm === :mw || error("not an MW run: $dir")
    name = _cell_name(mr)
    src = joinpath(dir, "result.h5")
    src_sha = bytes2hex(sha256(read(src)))
    row = Dict{Symbol,Any}(
        :problem => String(mr.problem), :d => mr.d, :B => mr.B, :seed => mr.seed, :algorithm => "mw",
        :estimator_version => ESTIMATOR_VERSION, :source_file => relpath(src, _ROOT), :source_sha256 => src_sha,
        :C_old => mr.Nlike_used, :terminated_by_old => String(string(get(mr.extras, :stop_reason, :unknown))),
        :N_pop => size(mr.samples, 2), :Neff_pop => neff_kish(mr), :eta_pop_consumed => neff_kish(mr) / max(mr.Nlike_used, 1.0),
        :logZ_pop => mr.logZ_estimate === missing ? NaN : mr.logZ_estimate,
        :nthreads => Threads.nthreads(), :hostname => gethostname(), :julia => string(VERSION),
        :timestamp_utc => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"),
    )
    mix = get(mr.extras, :mixture, nothing)
    if !(mix isa MixtureModel)
        row[:status] = "no_mixture_stored"; row[:N_fresh] = 0
        return row
    end
    row[:K_components] = length(mix.components)
    row[:proposal_sha256] = proposal_hash(mix)
    remaining = floor(Int, mr.B - mr.Nlike_used)
    status = "full_remaining_budget"
    if mr.Nlike_used > mr.B
        status = "exceeds_strict_budget"; remaining = 0
    elseif remaining <= 0
        status = "no_fresh_estimate_within_budget"; remaining = 0
    end
    nfresh = remaining
    if nfresh_cap !== nothing && nfresh > nfresh_cap
        nfresh = nfresh_cap; status = "capped_within_budget"
    end
    row[:status] = status; row[:N_fresh] = nfresh
    row[:C_total] = mr.Nlike_used + nfresh
    row[:rng_seed] = rng_seed_for(mr.problem, mr.d, mr.B, mr.seed)
    row[:transformation] = "BAT PriorToGaussian (PriorSubstitution); theta_j = 2 L Phi(z_j) - L"
    nfresh == 0 && return row

    cfg = _make_cfg(mr.problem, mr.d)
    log_f = build_log_f(cfg)
    counter = LikelihoodCounter(log_f); reset!(counter)
    posterior = posterior_measure(cfg, counter)
    pstr, _ = bat_transform(PriorToGaussian(), posterior)
    L = cfg.L
    rng = Xoshiro(row[:rng_seed])
    t0 = time()
    Z, logq, logp = fresh_core(mix, nothing, nfresh, rng; chunk = chunk, fast_target = (counter, L))
    t_draw_eval = time() - t0
    charged = cost(counter)
    charged == nfresh || @warn "counter charge differs from N_fresh" charged nfresh
    row[:counter_charged] = charged
    # Spot check of the fast paths against the library paths on 200 of the
    # actual draws: BAT's transformed density and Distributions' mixture
    # density. Recorded per run; a deviation above 1e-8 aborts the run.
    nchk = min(200, nfresh)
    idx = round.(Int, range(1, nfresh; length = nchk))
    dev_p = maximum(abs(logdensityof(pstr, Z[:, i]) - logp[i]) for i in idx)
    dev_q = maximum(abs(logpdf(mix, Z[:, i]) - logq[i]) for i in idx)
    row[:spotcheck_max_dev_logp_vs_BAT] = dev_p
    row[:spotcheck_max_dev_logq_vs_Distributions] = dev_q
    if !(dev_p < 1e-8 && dev_q < 1e-8)
        row[:status] = "error_spotcheck"; row[:weight_status] = "error_spotcheck"
        @error "fast-path spot check failed" name dev_p dev_q
        return row
    end
    logw = logp .- logq
    ws = weight_summary(logw)
    row[:weight_status] = ws.status
    row[:logZ_fresh] = ws.logZ; row[:logZ_fresh_se] = ws.logZ_se
    row[:Neff_fresh] = ws.ess
    row[:eta_cost_fresh] = ws.ess / row[:C_total]
    row[:eta_proposal_fresh] = ws.ess / nfresh
    row[:eta_nominalB_fresh] = ws.ess / mr.B
    row[:max_w_fresh] = ws.max_w; row[:pareto_k_fresh] = ws.pareto_k
    row[:n_zero_w] = ws.n_zero; row[:n_nan_w] = ws.n_nan; row[:n_posinf_w] = ws.n_posinf
    row[:wall_fresh_draw_eval_s] = t_draw_eval
    ws.status == "ok" || return row

    # Back-transform to the cube (identity verified against BAT's inverse
    # transform) and evaluate the archived metric pipeline on the fresh
    # weighted sample with the fixed-10,000 resampling convention.
    theta = 2L .* _normcdf.(Z) .- L
    logf = similar(logp)
    @inbounds for i in eachindex(logp)
        logf[i] = logp[i] - sum(_normlogpdf, view(Z, :, i))
    end
    wshift = exp.(logw .- maximum(logw))
    mr_fresh = MethodResult(:mw, mr.problem, mr.d, mr.seed, mr.B, Float64(row[:C_total]),
                            mr.n_primal + nfresh, mr.n_grad_partials, mr.wall_time_s + t_draw_eval,
                            theta, wshift, logf, ws.logZ, ws.logZ_se,
                            Dict{Symbol,Any}(:stop_reason => :fresh_draw, :estimator => ESTIMATOR_VERSION))
    truth = load_truth(joinpath(OUT, "truth", @sprintf("%s_d%d.h5", String(mr.problem), mr.d)))
    t1 = time()
    row[:W1_marginal_avg_fresh] = w1_marginal_avg(mr_fresh, truth)
    row[:SWD_fresh] = sliced_wasserstein(mr_fresh, truth; seed = 12345 + mr.seed)
    qe = quantile_errors(mr_fresh, truth)
    for k in (:p025, :p160, :p500, :p840, :p975)
        row[Symbol("QE_", k, "_fresh")] = qe[k]
    end
    row[:dlogZ_fresh] = isfinite(truth.logZ) ? abs(ws.logZ - truth.logZ) : NaN
    row[:truth_logZ] = truth.logZ
    row[:wall_metrics_s] = time() - t1
    # Archived-population metrics recomputed here for a like-for-like row
    # (same code path, same resampling convention).
    row[:W1_marginal_avg_pop] = w1_marginal_avg(mr, truth)
    qp = quantile_errors(mr, truth)
    row[:QE_p500_pop] = qp[:p500]; row[:QE_p975_pop] = qp[:p975]
    row[:dlogZ_pop] = (mr.logZ_estimate === missing || !isfinite(truth.logZ)) ? NaN : abs(mr.logZ_estimate - truth.logZ)

    # Reproducibility payload: the equal-weight 10,000-point resample the
    # metrics used, the sorted top weights, and the summary.
    isdir(FRESH_DIR) || mkpath(FRESH_DIR)
    resample = ExperimentsBase._resample_to_eval_N(mr_fresh, ExperimentsBase.W1_EVAL_N,
                                                    MersenneTwister(20260925 + mr.seed))
    top = sort(logw; rev = true)[1:min(2000, nfresh)]
    tmp = joinpath(FRESH_DIR, name * ".jld2.tmp")
    JLD2.jldopen(tmp, "w"; iotype = IOStream) do f
        f["summary"] = Dict(String(k) => v for (k, v) in row)
        f["resample_10k_theta"] = resample
        f["logw_top2000_sorted"] = top
        f["logw_quantiles"] = quantile(logw, [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0])
    end
    mv(tmp, joinpath(FRESH_DIR, name * ".jld2"); force = true)
    row[:output_path] = relpath(joinpath(FRESH_DIR, name * ".jld2"), _ROOT)
    return row
end

# -----------------------------------------------------------------------------
# Subsets (complete predetermined groups)
# -----------------------------------------------------------------------------
function select_dirs(subset::String)
    dirs = String[]
    for prob in PROBLEMS, d in (2, 5, 10), B in BUDGETS, seed in SEED_GRID
        dir = cell_dir(OUT, prob, :mw, d, B, seed)
        isfile(joinpath(dir, "result.h5")) || continue
        keep = if subset == "headline"
            B == 5e5 && ((prob === :eggbox && d == 2) || (prob !== :eggbox && d == 5))
        elseif subset == "top"
            B == 5e5
        elseif subset == "mid"
            B == 5e4
        elseif subset == "low"
            B == 5e3
        elseif subset == "all"
            true
        else
            error("unknown subset $subset")
        end
        keep && push!(dirs, dir)
    end
    return dirs
end

_done(dir) = isfile(joinpath(FRESH_DIR, basename(dir) * ".jld2"))

function run_subset(subset::String; nfresh_cap = nothing, limit = nothing, chunk = 8192)
    dirs = select_dirs(subset)
    println("subset $subset: $(length(dirs)) MW run records; already done: $(count(_done, dirs))")
    todo = filter(!_done, dirs)
    limit !== nothing && (todo = todo[1:min(limit, length(todo))])
    tstart = time()
    for (i, dir) in enumerate(todo)
        t0 = time()
        row = process_run(dir; nfresh_cap = nfresh_cap, chunk = chunk)
        @printf("[%3d/%3d] %-40s status=%-30s N_fresh=%8d  eta_pop=%.4f eta_cost_fresh=%s  k=%s  %.1f s\n",
            i, length(todo), basename(dir), row[:status], row[:N_fresh],
            row[:eta_pop_consumed],
            haskey(row, :eta_cost_fresh) ? @sprintf("%.4f", row[:eta_cost_fresh]) : "n/a",
            haskey(row, :pareto_k_fresh) ? @sprintf("%.2f", row[:pareto_k_fresh]) : "n/a",
            time() - t0)
        if row[:N_fresh] == 0 && !haskey(row, :output_path)
            # Persist the no-draw record too, so the aggregate keeps the row.
            isdir(FRESH_DIR) || mkpath(FRESH_DIR)
            JLD2.jldopen(joinpath(FRESH_DIR, basename(dir) * ".jld2"), "w"; iotype = IOStream) do f
                f["summary"] = Dict(String(k) => v for (k, v) in row)
            end
        end
        flush(stdout)
    end
    @printf("subset %s finished: %d runs in %.1f min\n", subset, length(todo), (time() - tstart) / 60)
end

# -----------------------------------------------------------------------------
# Aggregate to cells_fresh.csv + manifest
# -----------------------------------------------------------------------------
function aggregate()
    rows = Dict{String,Any}[]
    for f in sort(readdir(FRESH_DIR))
        endswith(f, ".jld2") || continue
        s = JLD2.jldopen(joinpath(FRESH_DIR, f), "r") do h; h["summary"]; end
        push!(rows, s)
    end
    keys_all = sort(unique(vcat([collect(keys(r)) for r in rows]...)))
    front = ["problem", "algorithm", "d", "B", "seed", "estimator_version", "status", "weight_status",
             "C_old", "N_fresh", "C_total", "K_components", "N_pop", "Neff_pop", "eta_pop_consumed",
             "Neff_fresh", "eta_cost_fresh", "eta_proposal_fresh", "eta_nominalB_fresh",
             "logZ_pop", "logZ_fresh", "logZ_fresh_se", "truth_logZ", "dlogZ_pop", "dlogZ_fresh",
             "W1_marginal_avg_pop", "W1_marginal_avg_fresh", "SWD_fresh",
             "QE_p025_fresh", "QE_p160_fresh", "QE_p500_fresh", "QE_p840_fresh", "QE_p975_fresh",
             "QE_p500_pop", "QE_p975_pop", "max_w_fresh", "pareto_k_fresh", "n_zero_w", "n_nan_w", "n_posinf_w",
             "wall_fresh_draw_eval_s", "wall_metrics_s", "terminated_by_old", "counter_charged"]
    cols = vcat(front, [k for k in keys_all if !(k in front)])
    df = DataFrame([c => [get(r, c, missing) for r in rows] for c in cols]...)
    sort!(df, [:problem, :d, :B, :seed])
    CSV.write(joinpath(REV, "cells_fresh.csv"), df)
    man = df[:, intersect(["problem", "d", "B", "seed", "algorithm", "estimator_version", "source_file", "source_sha256",
                           "transformation", "proposal_sha256", "rng_seed", "N_fresh", "C_old", "C_total",
                           "wall_fresh_draw_eval_s", "status", "weight_status", "output_path", "nthreads", "hostname",
                           "julia", "timestamp_utc"], cols)]
    man.generating_commit .= _git_head()
    man.archived_code_commit .= "254e9852dede4aa6ee4547e114ff8f608824914d"
    CSV.write(joinpath(REV, "fresh_manifest.csv"), man)
    println("wrote $(nrow(df)) rows to $(joinpath(REV, "cells_fresh.csv")) and fresh_manifest.csv")
    return df
end

function _git_head()
    try
        return strip(read(Cmd(`git rev-parse HEAD`; dir = _ROOT), String))
    catch
        return "unknown"
    end
end

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
function run_tests()
    println("== test 1: constant ratio f_z = Z q_T  (every weight Z, logZ = log Z, ESS = N)")
    rng = Xoshiro(1)
    mix = MixtureModel([MvNormal([0.0, 0.0], [1.0 0.3; 0.3 2.0]), MvNormal([2.0, -1.0], [0.5 0.0; 0.0 0.5])], [0.3, 0.7])
    Ztrue = 7.5
    Z, logq, logp = fresh_core(mix, z -> log(Ztrue) + logpdf(mix, z), 20_000, rng; chunk = 4096)
    ws = weight_summary(logp .- logq)
    @assert abs(ws.logZ - log(Ztrue)) < 1e-12 "logZ mismatch: $(ws.logZ) vs $(log(Ztrue))"
    @assert abs(ws.ess - 20_000) < 1e-6 "ESS mismatch: $(ws.ess)"
    @assert maximum(abs.(exp.(logp .- logq) .- Ztrue)) < 1e-10
    println("   ok: logZ = $(ws.logZ) (log Z = $(log(Ztrue))), ESS = $(ws.ess)")

    println("== test 2: known Gaussian target under a wider mixture proposal (moments, logZ within MC tolerance)")
    rng = Xoshiro(2)
    target = MvNormal([1.0, -2.0, 0.5], Diagonal([1.0, 4.0, 0.25]))
    mixp = MixtureModel([MvNormal([0.5, -1.5, 0.3], Diagonal([2.0, 6.0, 0.6])),
                         MvNormal([1.5, -2.5, 0.7], Diagonal([2.0, 6.0, 0.6]))], [0.5, 0.5])
    n = 200_000
    Z, logq, logp = fresh_core(mixp, z -> log(3.0) + logpdf(target, z), n, rng)
    logw = logp .- logq
    ws = weight_summary(logw)
    nw = exp.(logw .- _logsumexp(logw))
    m = Z * nw
    @assert abs(ws.logZ - log(3.0)) < 4 * ws.logZ_se + 1e-3 "logZ off: $(ws.logZ) +- $(ws.logZ_se)"
    @assert maximum(abs.(m .- mean(target))) < 0.03 "mean off: $m"
    println(@sprintf("   ok: logZ = %.4f +- %.4f (log 3 = %.4f), mean = %s, ESS = %.0f of %d, k = %.2f",
        ws.logZ, ws.logZ_se, log(3.0), string(round.(m; digits = 3)), ws.ess, n, ws.pareto_k))

    println("== test 3: two-component counterexample (retained population biased, fresh draws unbiased)")
    a = 3.0; N = 100_000
    q0 = Normal(-a, 1.0); q1 = Normal(a, 1.0)
    qT = MixtureModel([q0, q1], [0.5, 0.5])          # final proposal equals the target
    rng = Xoshiro(3)
    x_hist = vcat(rand(rng, q0, N), rand(rng, q1, N ÷ 2))   # N from q0, N/2 from q1
    w_hist = [pdf(qT, x) / pdf(qT, x) for x in x_hist]     # all weights one
    mean_hist = sum(w_hist .* x_hist) / sum(w_hist)
    ess_hist = sum(w_hist)^2 / sum(w_hist .^ 2)
    x_fresh = rand(rng, qT, N)
    mean_fresh = mean(x_fresh)
    @assert abs(mean_hist + a / 3) < 0.05 "historical mean $(mean_hist), expected about $(-a/3)"
    @assert abs(mean_fresh) < 0.05 "fresh mean $(mean_fresh), expected about 0"
    @assert abs(ess_hist - 1.5N) < 1e-6
    println(@sprintf("   ok: retained-population mean = %.3f (limit -a/3 = %.3f) with Kish ESS = %.0f = 3N/2; fresh mean = %.3f",
        mean_hist, -a / 3, ess_hist, mean_fresh))

    println("== test 4: transform identities and full-mixture density on one archived cell (funnel d=5 seed 11)")
    dir = cell_dir(OUT, :funnel, :mw, 5, 5e5, 11)
    mr = load_method_result(dir)
    mix = mr.extras[:mixture]
    @assert mix isa MixtureModel && length(mix.components) > 1
    @assert abs(sum(mix.prior.p) - 1) < 1e-10
    cfg = _make_cfg(:funnel, 5); log_f = build_log_f(cfg)
    counter = LikelihoodCounter(log_f); reset!(counter)
    posterior = posterior_measure(cfg, counter)
    pstr, f_trafo = bat_transform(PriorToGaussian(), posterior)
    L = cfg.L
    Z, logq, logp = fresh_core(mix, z -> logdensityof(pstr, z), 3000, Xoshiro(4))
    @assert cost(counter) == 3000 "counter charged $(cost(counter)) for 3000 draws"
    theta_id = 2L .* _normcdf.(Z) .- L
    dsv = BAT.DensitySampleVector([Z[:, i] for i in 1:3000], logp)
    back = bat_transform(BAT.InverseFunctions.inverse(f_trafo), dsv).result
    theta_bat = hcat([collect(v) for v in back.v]...)
    @assert maximum(abs.(theta_bat .- theta_id)) < 1e-12 "inverse transform mismatch"
    rhs = [log_f(theta_id[:, i]) + sum(_normlogpdf, view(Z, :, i)) for i in 1:3000]
    @assert maximum(abs.(logp .- rhs)) < 1e-9 "density identity mismatch"
    # full mixture density equals the log-sum-exp over all components, and
    # the fast evaluator equals Distributions' logpdf
    lq_manual = [_logsumexp([log(mix.prior.p[k]) + logpdf(mix.components[k], Z[:, i]) for k in 1:length(mix.components)]) for i in 1:50]
    @assert maximum(abs.(lq_manual .- logq[1:50])) < 1e-8 "mixture density mismatch"
    lq_dist = [logpdf(mix, Z[:, i]) for i in 1:3000]
    @assert maximum(abs.(lq_dist .- logq)) < 1e-8 "fast mixture density differs from Distributions"
    # fast target path (counter + cube identity) equals BAT's logdensityof
    counter2 = LikelihoodCounter(log_f); reset!(counter2)
    Z2, logq2, logp2 = fresh_core(mix, nothing, 3000, Xoshiro(4); fast_target = (counter2, L))
    @assert Z2 == Z "draw sequence differs between paths"
    @assert cost(counter2) == 3000
    @assert maximum(abs.(logp2 .- logp)) < 1e-9 "fast target path differs from BAT: $(maximum(abs.(logp2 .- logp)))"
    @assert logq2 == logq
    ws = weight_summary(logp .- logq)
    truth = load_truth(joinpath(OUT, "truth", "funnel_d5.h5"))
    println(@sprintf("   ok: K = %d, counter = 3000, transform identities exact; fresh logZ (3000 draws) = %.4f, truth = %.4f, archived population = %.4f",
        length(mix.components), ws.logZ, truth.logZ, mr.logZ_estimate))

    println("== test 5: thread-count independence of the draw sequence (chunked sequential RNG)")
    Z1, _, _ = fresh_core(mix, z -> 0.0, 10_000, Xoshiro(5); chunk = 1000)
    Z2, _, _ = fresh_core(mix, z -> 0.0, 10_000, Xoshiro(5); chunk = 1000)
    @assert Z1 == Z2
    println("   ok (identical draws for identical seed; chunk boundaries fixed)")
    println("all tests passed")
end

# -----------------------------------------------------------------------------
function main(args)
    opts = Dict{String,Any}("subset" => nothing, "aggregate" => false, "test" => false,
                            "nfresh_cap" => nothing, "limit" => nothing, "chunk" => 8192)
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--subset"; opts["subset"] = args[i + 1]; i += 2
        elseif a == "--aggregate"; opts["aggregate"] = true; i += 1
        elseif a == "--test"; opts["test"] = true; i += 1
        elseif a == "--nfresh-cap"; opts["nfresh_cap"] = parse(Int, args[i + 1]); i += 2
        elseif a == "--limit"; opts["limit"] = parse(Int, args[i + 1]); i += 2
        elseif a == "--chunk"; opts["chunk"] = parse(Int, args[i + 1]); i += 2
        else; error("unknown argument $a")
        end
    end
    isdir(REV) || mkpath(REV)
    println("threads = $(Threads.nthreads()); estimator = $ESTIMATOR_VERSION; git HEAD = $(_git_head())")
    opts["test"] && run_tests()
    if opts["subset"] !== nothing
        run_subset(opts["subset"]; nfresh_cap = opts["nfresh_cap"], limit = opts["limit"], chunk = opts["chunk"])
    end
    opts["aggregate"] && aggregate()
end

# Run as a script; when `include`d by another tool (partition_mass.jl reuses
# FastMixture / fresh_core / rng_seed_for), only the definitions are loaded.
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

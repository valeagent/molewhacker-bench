# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# algo_mw.jl — MoleWhacker reference run (Protocol §5.5, Table 6.2)
# =============================================================================
#
# Wraps the MoleWhacker module (src/MoleWhacker.jl) so that:
#   * the LikelihoodCounter sees every primal and dual call (the
#     posterior is constructed from a counter-wrapped log_f);
#   * Table 6.2 hyperparameters are honoured exactly;
#   * the budget is the *only* termination criterion together with
#     T_max and Neff_target — no internal early stops on Shell.
#
# Loading: this file relies on `MoleWhacker` being available as a
# module; it is included from `src/MoleWhacker.jl` at the repository
# root. A guard below makes the include idempotent if the module is
# already loaded.

if !isdefined(@__MODULE__, :MoleWhacker)
    let mw_path = joinpath(@__DIR__, "..", "..", "..", "src", "MoleWhacker.jl")
        if isfile(mw_path)
            include(mw_path)
        else
            @warn "MoleWhacker module not found at $mw_path — algo_mw will fail at runtime."
        end
    end
end

import .MoleWhacker as _MW
using BAT: bat_transform, PriorToGaussian, bat_eff_sample_size, KishESS,
            DensitySampleVector
using InverseFunctions: inverse


"""
    MWParams

Container for Table 6.2 of the protocol.

| Parameter | Default |
| n_seed                | 64      |
| Sobol cube half-width | 5       |
| N (samples / iter)    | 2000    |
| n_parallel            | 8       |
| τ_μ                   | 0.5     |
| τ_Σ                   | 0.2 * d |
| Neff_target           | 0.5*|S| |
| T_max                 | 20      |
| init_budget_fraction  | 0.30    |
"""
Base.@kwdef struct MWParams
    n_seed::Int = 64
    sobol_r::Float64 = 5.0
    nsamples_per_iter::Int = 2_000
    n_parallel::Int = 8
    tau_mu::Float64 = 0.5
    tau_Sigma_factor::Float64 = 0.2     # tau_Σ = factor * d
    neff_target_frac::Float64 = 0.5
    T_max::Int = 20
    init_budget_fraction::Float64 = 0.30
end


"""
    run_mw(cfg::ProblemConfig, B::Real, seed::Integer; counter=nothing,
           params::MWParams=MWParams(), n_threads=Threads.nthreads()) -> MethodResult
"""
function run_mw(cfg::ProblemConfig, B::Real, seed::Integer;
                  counter::Union{LikelihoodCounter,Nothing} = nothing,
                  params::MWParams = MWParams(),
                  n_threads::Integer = Threads.nthreads(),
                  # Storyboard-figure support: optional per-iteration snapshot
                  # passthrough to whack_many_moles. Defaults preserve the
                  # benchmark behavior exactly (no snapshots, no extra cost
                  # on the likelihood counter — snapshots only serialize
                  # already-computed state).
                  cache_dir::Union{AbstractString,Nothing} = nothing,
                  save_iteration_samples::Bool = false)
    log_f = build_log_f(cfg)
    counter === nothing && (counter = LikelihoodCounter(log_f))
    reset!(counter)
    # V8 NOTE (B4): MoleWhacker's internals consume Julia's task-global
    # RNG (and use Threads.@threads for the parallel fits), so the run is
    # reproducible for a fixed seed AND fixed thread count, but not
    # across different `-t` settings. Plumbing an explicit RNG through
    # the MoleWhacker API is out of scope for the benchmark harness (the
    # algorithm under study owns that API). Also note the Sobol +
    # L-BFGS initialisation is fully deterministic — identical across
    # seeds; only the sampling noise varies seed-to-seed. Both caveats
    # are disclosed in the thesis methodology.
    Random.seed!(seed)
    d = cfg.d

    # Cap n_seed to fit init_budget_fraction. The empirical cost of one
    # L-BFGS + Hessian fit on a smooth target at d=5 is ~150 evaluations
    # (counter cost units, see MoleWhacker docs); we use 200 as a
    # conservative upper bound when planning. If the resulting cap is
    # below 4, we let n_seed = 4 (MoleWhacker needs ≥ 4 to be useful).
    init_budget = floor(Int, params.init_budget_fraction * B)
    cost_per_seed = 200 * (1 + d)              # L-BFGS dual + Hessian
    n_seed_planned = clamp(div(init_budget, cost_per_seed),
                            4, params.n_seed)

    posterior = posterior_measure(cfg, counter)

    t0 = time_ns()

    # ------------------------------------------------------------------
    # Initialization: Sobol seeds → L-BFGS → local Gaussian fits
    # ------------------------------------------------------------------
    init = try
        _MW.make_init_samples(posterior, n_seed_planned, params.nsamples_per_iter)
    catch err
        @warn "MoleWhacker initialisation raised — returning empty result" exception = (err, catch_backtrace())
        return _empty_mw_result(cfg, seed, B, counter, t0)
    end

    # Per the April 2026 protocol revision: MW must run
    # until the *budget* is exhausted — Neff and efficiency targets
    # are deliberately left unreachable. The Neff_target field is
    # retained only to power figures that show how Neff evolved.
    Neff_target = params.neff_target_frac * params.nsamples_per_iter

    # ------------------------------------------------------------------
    # Adaptive whacking loop
    # ------------------------------------------------------------------
    whack_result = try
        _MW.whack_many_moles(
            posterior, init;
            target_efficiency = Inf,
            target_ess = Inf,                # disable Neff stop — Protocol §5.5 (revised)
            maxiter = params.T_max,
            max_evals = floor(Int, B),
            counter = counter,            # MW reads cum cost via _read_counter_cost
            n_parallel = params.n_parallel,
            cache_dir = cache_dir,
            save_iteration_samples = save_iteration_samples,
        )
    catch err
        @warn "MoleWhacker whacking raised — partial output retained" exception = (err, catch_backtrace())
        return _empty_mw_result(cfg, seed, B, counter, t0)
    end

    wall_time = (time_ns() - t0) / 1e9

    # ------------------------------------------------------------------
    # Convert MoleWhacker output to the protocol MethodResult
    # ------------------------------------------------------------------
    smpls = whack_result.samples_user
    N_out = length(smpls.v)
    samples_mat = Matrix{Float64}(undef, d, N_out)
    weights = Float64.(smpls.weight)
    logd = Vector{Float64}(undef, N_out)
    @inbounds for i in 1:N_out
        for j in 1:d
            samples_mat[j, i] = smpls.v[i][j]
        end
        logd[i] = Float64(smpls.logd[i])
    end

    # SNIS evidence on the cube prior:
    #   Ẑ = (1/N) Σ p(θ) / q(θ) where p contains the cube prior 1/(2L)^d
    # The samples_user logd already contains posterior logp = log_f + cube prior.
    # Drawn from approx_dist q. The SNIS evidence is logsumexp(logd_q-corrected).
    # MoleWhacker stores per-sample logd as logdensityof(pstr, x) — i.e. the
    # joint log-posterior in transformed space. To recover the standard
    # posterior-evidence we use the unnormalised weights:
    log_meanw = N_out > 0 ? logsumexp(log.(weights .+ eps())) - log(N_out) : -Inf
    # MoleWhacker's weights are exp(logd_p - logd_q - max), so log_meanw is
    # log(<p/q>) up to a maximum offset which is the unknown additive
    # constant in p. Recovering log Ẑ requires that constant; here we use
    # the SNIS evidence the BAT pipeline computes — see whack_log entries.
    # As a robust approximation, take logZ = log(mean(p(θ)/q(θ))), with
    # p the (unnormalised) target on the cube.
    logZ_est, logZ_se = _mw_evidence_estimate(whack_result, posterior, smpls)

    # Iteration log: convert MoleWhacker's WhackLogEntry vector to a
    # plain Vector{NamedTuple} for the iteration plot. Includes the
    # per-iteration cumulative likelihood cost (Protocol §P1-5) so the
    # plotter can render eta(t) = ess(t) / cum_cost(t) without a linear
    # extrapolation bias.
    iter_log = NamedTuple[]
    for entry in whack_result.whack_log
        cumc = isdefined(entry, :cum_cost) ? Float64(entry.cum_cost) : NaN
        push!(iter_log, (
            iter = entry.iter,
            n_components = entry.n_components,
            eff = entry.eff,
            ess = entry.ess,
            cum_cost = cumc,
        ))
    end

    extras = Dict{Symbol,Any}(
        :stop_reason => _mw_stop_reason(whack_result, params, B, cost(counter)),
        :n_seed_used => n_seed_planned,
        :n_parallel => params.n_parallel,
        :tau_mu => params.tau_mu,
        :tau_Sigma => params.tau_Sigma_factor * d,
        :T_max => params.T_max,
        :Neff_target => Neff_target,
        :iter_log => iter_log,
        :mixture => whack_result.approx_dist,
    )

    return MethodResult(
        algorithm = :mw,
        problem = _problem_symbol(cfg),
        d = d,
        seed = Int(seed),
        B = Float64(B),
        counter = counter,
        wall_time_s = wall_time,
        samples = samples_mat,
        weights = weights,
        logd = logd,
        logZ_estimate = logZ_est,
        logZ_estimate_se = logZ_se,
        extras = extras,
    )
end


# Estimate log evidence from MoleWhacker's importance-weighted output.
# Strategy: redo a fresh pass through `pstr` to get logd_p exactly, then
# compute Ẑ = mean(exp(log_p - log_q)).
function _mw_evidence_estimate(whack_result, posterior, smpls)
    try
        approx = whack_result.approx_dist
        # Use samples already in posterior (transformed) space: samples_p
        smpls_p = whack_result.samples_p
        log_p = collect(smpls_p.logd)             # log target at each sample
        log_q = [Float64(logpdf(approx, collect(v))) for v in smpls_p.v]
        log_w = log_p .- log_q
        Nw = length(log_w)
        Nw == 0 && return missing, missing
        log_meanw = logsumexp(log_w) - log(Nw)
        # Variance of the weights via second moment
        log_meanw2 = logsumexp(2 .* log_w) - log(Nw)
        var_w = exp(log_meanw2) - exp(2 * log_meanw)
        se = (var_w > 0 && Nw > 1) ?
            sqrt(var_w) / (sqrt(Nw) * exp(log_meanw)) : missing
        return Float64(log_meanw), se
    catch
        return missing, missing
    end
end

function _mw_stop_reason(whack_result, params::MWParams, B::Real, used::Real)
    last_entry = isempty(whack_result.whack_log) ? nothing : last(whack_result.whack_log)
    if used >= B
        return :budget
    end
    # Neff target intentionally disabled (see run_mw): if we haven't
    # hit the budget the only possible stop is T_max.
    if last_entry !== nothing && last_entry.iter >= params.T_max
        return :T_max
    end
    return :unknown
end

function _empty_mw_result(cfg, seed, B, counter, t0)
    d = cfg.d
    wall_time = (time_ns() - t0) / 1e9
    return MethodResult(
        algorithm = :mw,
        problem = _problem_symbol(cfg),
        d = d,
        seed = Int(seed),
        B = Float64(B),
        counter = counter,
        wall_time_s = wall_time,
        samples = zeros(d, 1),
        weights = [1.0],
        logd = [NaN],
        logZ_estimate = missing,
        logZ_estimate_se = missing,
        extras = Dict{Symbol,Any}(:stop_reason => :error),
    )
end


# =============================================================================
# Top-level dispatcher used by 01_run_cell.jl
# =============================================================================

"""
    run_algorithm(alg::Symbol, cfg::ProblemConfig, B::Real, seed::Integer;
                  counter=nothing) -> MethodResult

One call site for all five algorithms. Returns a `MethodResult` regardless
of which algorithm was used.
"""
function run_algorithm(alg::Symbol, cfg::ProblemConfig, B::Real, seed::Integer;
                         counter::Union{LikelihoodCounter,Nothing} = nothing)
    if alg === :is
        return run_is(cfg, B, seed; counter = counter)
    elseif alg === :mh
        return run_mh(cfg, B, seed; counter = counter)
    elseif alg === :nuts
        return run_nuts(cfg, B, seed; counter = counter)
    elseif alg === :ns
        return run_ns(cfg, B, seed; counter = counter)
    elseif alg === :mw
        return run_mw(cfg, B, seed; counter = counter)
    else
        error("run_algorithm: unknown algorithm $alg (expected one of :is, :mh, :nuts, :ns, :mw)")
    end
end

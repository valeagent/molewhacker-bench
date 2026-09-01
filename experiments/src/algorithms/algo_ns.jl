# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# algo_ns.jl — Ellipsoidal nested sampling (Protocol §5.4)
# =============================================================================
#
# NestedSamplers.jl backend via BAT. N_live = max(400, 25 * d²),
# Δ_logZ = 0.5 stopping criterion, hard cutoff at B cost units.

import BAT
using BAT: bat_sample
import NestedSamplers
using NestedSamplers: Nested, Bounds, Proposals
# Pull in Measurements so we can match its type explicitly for logZ.
import Measurements
using Measurements: Measurement


"""
    run_ns(cfg, B, seed; counter=nothing, dlogz=1e-6)

Per Protocol §6.1 (rev April 2026): for headline cells the
evidence-converged criterion is set so small (`dlogz = 1e-6`) that the
budget `B` (`max_ncalls`) is the binding constraint. NS therefore uses
between 80 % and 120 % of `B` like every other algorithm. An appendix
figure that documents NS's natural stopping behaviour with
`dlogz = 0.5` can be produced by passing the kwarg explicitly.
"""
function run_ns(cfg::ProblemConfig, B::Real, seed::Integer;
                  counter::Union{LikelihoodCounter,Nothing} = nothing,
                  dlogz::Real = 1e-6,
                  max_ncalls_override::Union{Nothing,Integer} = nothing)
    log_f = build_log_f(cfg)
    counter === nothing && (counter = LikelihoodCounter(log_f))
    reset!(counter)
    d = cfg.d
    L = cfg.L
    # V8 NOTE (B4): NestedSamplers.jl (via BAT's wrapper) consumes the
    # task-global RNG; BAT.EllipsoidalNestedSampling exposes no explicit
    # RNG argument. Reproducible for fixed seed + fixed thread count.
    Random.seed!(seed)

    n_live = max(400, 25 * d^2)
    posterior = posterior_measure(cfg, counter)

    # NestedSamplers.jl's `maxcall` is checked at the outer iteration
    # boundary and BAT's wrapper performs additional book-keeping
    # evaluations (initial live points, transformation, etc.). The V6
    # single-point calibration (factor 1.55, measured on mvn d=5 B=5e3)
    # was systematically off across the grid — realised cost landed at a
    # median 0.74·B (and up to 2.5·B at d = 10), which the honest V8-A4
    # relabelling turns into BUDGET-VIOLATION flags. V8-FIX-A4b:
    # `max_ncalls_override` lets the recalibration driver size the cap
    # per cell from that cell's own measured cap→cost response — a pure
    # harness-interface calibration (calls per cap unit), carrying no
    # information about the target density.
    overshoot_factor = 1.55
    max_ncalls_eff = max_ncalls_override !== nothing ?
        max(2 * Int(n_live), Int(max_ncalls_override)) :
        max(2 * Int(n_live), floor(Int, (B - Int(n_live)) / overshoot_factor))
    cap_at_floor = max_ncalls_eff == 2 * Int(n_live)

    t0 = time_ns()
    bat_result = nothing
    terminated_by = :unknown
    try
        bat_result = bat_sample(posterior, BAT.EllipsoidalNestedSampling(;
            num_live_points = Int(n_live),
            dlogz = Float64(dlogz),
            max_ncalls = max_ncalls_eff,
        ))
        # V8-FIX-A4: label by the actually binding constraint. The old
        # rule (`cost >= 0.8B ? :budget : :dlogz`) mislabelled call-cap
        # underruns as :dlogz, which is the budget-exemption key in
        # metrics.jl — under-budget NS cells escaped the
        # BUDGET-VIOLATION flag. :dlogz is now only assigned when the
        # sampler stopped clearly below its call cap, i.e. the evidence
        # criterion genuinely fired.
        terminated_by = if cost(counter) > 1.2 * B && cap_at_floor
            # V8-FIX-A4b: the cap was already at the structural minimum
            # (2·n_live) and the run still blew the ceiling — NS cannot
            # be run in band at this budget with the protocol-mandated
            # n_live. Honest N/A, mirroring the gradient-sampler rule.
            :budget_infeasible
        elseif cost(counter) >= 0.8 * B
            :budget
        elseif counter.n_primal >= max_ncalls_eff
            :maxcall
        else
            :dlogz
        end
    catch err
        @warn "NS sampling raised — partial output retained" exception = (err, catch_backtrace())
    end

    samples_v = Vector{Vector{Float64}}()
    weights_v = Float64[]
    logd_v = Float64[]
    logZ_est = missing
    logZ_se = missing

    if bat_result !== nothing
        for s in bat_result.result
            push!(samples_v, [Float64(x) for x in s.v])
            push!(weights_v, Float64(s.weight))
            push!(logd_v, Float64(s.logd))
        end
        # Extract logZ if present in result.logevidence or result.logintegral
        if hasproperty(bat_result, :logintegral)
            li = bat_result.logintegral
            logZ_est = _measurement_value(li)
            logZ_se = _measurement_uncertainty(li)
        elseif hasproperty(bat_result, :logevidence)
            le = bat_result.logevidence
            logZ_est = _measurement_value(le)
            logZ_se = _measurement_uncertainty(le)
        end
    end
    wall_time = (time_ns() - t0) / 1e9

    if isempty(samples_v)
        samples_mat = zeros(d, 1)
        weights_v = [1.0]
        logd_v = [NaN]
    else
        samples_mat = Matrix{Float64}(undef, d, length(samples_v))
        @inbounds for i in eachindex(samples_v)
            for j in 1:d
                samples_mat[j, i] = samples_v[i][j]
            end
        end
    end

    extras = Dict{Symbol,Any}(
        :stop_reason => terminated_by,
        :terminated_by => terminated_by,
        :budget_infeasible => terminated_by === :budget_infeasible,
        :num_live_points => Int(n_live),
        :dlogz_threshold => Float64(dlogz),
        :max_ncalls_eff => Int(max_ncalls_eff),
        :max_ncalls_overridden => max_ncalls_override !== nothing,
    )

    return MethodResult(
        algorithm = :ns,
        problem = _problem_symbol(cfg),
        d = d,
        seed = Int(seed),
        B = Float64(B),
        counter = counter,
        wall_time_s = wall_time,
        samples = samples_mat,
        weights = weights_v,
        logd = logd_v,
        logZ_estimate = logZ_est,
        logZ_estimate_se = logZ_se,
        extras = extras,
    )
end


# Tiny helpers for Measurements.Measurement objects.
_measurement_value(x::Measurement) = Float64(Measurements.value(x))
_measurement_value(x::Real) = Float64(x)
_measurement_value(x) = Float64(x)
_measurement_uncertainty(x::Measurement) = Float64(Measurements.uncertainty(x))
_measurement_uncertainty(x::Real) = missing
_measurement_uncertainty(x) = missing

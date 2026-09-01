# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# algo_is.jl — plain importance sampling from the box prior (Protocol §5.1)
# =============================================================================
#
# One primal evaluation per sample. Cost ≈ B exactly. The evidence
# estimate is the cube-prior-normalised marginal likelihood, which is
# what the truth files report:
#
#     Z = ∫_{[-L,L]^d} f(θ) · π(θ) dθ      with π = Uniform on cube
#       = (1/(2L)^d) ∫_{[-L,L]^d} f(θ) dθ
#       = E_{θ ~ π}[f(θ)]
#
# Therefore for samples θ_i ~ π,
#
#     Ẑ = (1/N) Σ exp(log_f(θ_i))
#     log Ẑ = logsumexp(log_f) - log N
#
# Note: the older +d·log(2L) term reported the *unnormalised* evidence,
# which differed from `truth.logZ` by exactly d·log(2L). The fix below
# brings IS into line with NS and MW, both of which already report the
# cube-prior-normalised evidence.

"""
    run_is(cfg::ProblemConfig, B::Real, seed::Integer; counter=nothing) -> MethodResult

Plain importance sampling from the cube prior. Each sample is one
primal evaluation; the evidence is the SNIS estimator.
"""
function run_is(cfg::ProblemConfig, B::Real, seed::Integer;
                  counter::Union{LikelihoodCounter,Nothing} = nothing)
    log_f = build_log_f(cfg)
    counter === nothing && (counter = LikelihoodCounter(log_f))
    reset!(counter)
    rng = Xoshiro(seed)
    d = cfg.d
    L = cfg.L
    N = floor(Int, B)
    @assert N > 0 "run_is: B too small to draw a sample"

    t0 = time_ns()
    # Draw N uniform-cube samples
    samples = (rand(rng, d, N) .- 0.5) .* (2L)
    logd = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        logd[i] = counter(view(samples, :, i))
    end
    wall_time = (time_ns() - t0) / 1e9

    # Cube-prior-normalised IS evidence (matches `truth.logZ` convention):
    #     Ẑ = (1/N) Σ exp(log_f(θ_i)),   θ_i ~ Uniform([-L, L]^d)
    log_meanw = logsumexp(logd) - log(N)
    logZ_est = log_meanw
    # Standard error of log Ẑ via delta-method on the weight mean.
    # Var(mean w) = (1/N) (mean w² − (mean w)²); divide by (mean w)².
    log2_meanw = logsumexp(2 .* logd) - log(N)
    var_w = exp(log2_meanw) - exp(2log_meanw)
    se_logZ = if var_w > 0 && N > 1
        sqrt(var_w) / (sqrt(N) * exp(log_meanw))
    else
        NaN
    end

    weights = exp.(logd .- maximum(logd))     # used by ESS / metrics
    Nlike_used = Float64(cost(counter))

    extras = Dict{Symbol,Any}(
        :stop_reason => :budget,
        :log_meanw => log_meanw,
        :ess_kish => sum(weights)^2 / sum(abs2, weights),
    )

    return MethodResult(
        algorithm = :is,
        problem = _problem_symbol(cfg),
        d = d,
        seed = Int(seed),
        B = Float64(B),
        counter = counter,
        wall_time_s = wall_time,
        samples = samples,
        weights = weights,
        logd = logd,
        logZ_estimate = logZ_est,
        logZ_estimate_se = se_logZ,
        extras = extras,
    )
end

# Map a config struct to the canonical problem symbol.
function _problem_symbol(cfg::ProblemConfig)
    cfg isa ConfigMVN && return :mvn
    cfg isa ConfigBanana && return :banana
    cfg isa ConfigFunnel && return :funnel
    cfg isa ConfigMRidges && return :mridges
    cfg isa ConfigShell && return :shell
    cfg isa ConfigMRidgesSpiky && return :mridges_spiky
    cfg isa ConfigEggbox && return :eggbox
    error("unknown ProblemConfig type")
end

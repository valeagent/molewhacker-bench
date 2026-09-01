# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# ProblemMVN.jl — correlated multivariate normal benchmark (Protocol §4.1)
# =============================================================================

"""
    abstract type ProblemConfig

Marker supertype for all problem-configuration structs in §4 of the
protocol.
"""
abstract type ProblemConfig end

"""
    ConfigMVN(d, sigma, rho, mu, L)

Configuration for the correlated MVN problem.

- `d::Int`            dimension
- `sigma::Float64`    common scalar standard deviation (each coordinate)
- `rho::Float64`      AR(1) correlation, `Σ_{ij} = σ² ρ^|i-j|`
- `mu::Vector{Float64}` length-d mean vector; default `0.5 * (-1)^(i+1)`
- `L::Float64`        cube prior half-width
"""
struct ConfigMVN <: ProblemConfig
    d::Int
    sigma::Float64
    rho::Float64
    mu::Vector{Float64}
    L::Float64
end

"""
    make_config_mvn(; d=5, sigma=1.0, rho=0.9, L=10.0) -> ConfigMVN

Constructs the canonical MVN configuration of Protocol §4.1.
"""
function make_config_mvn(; d::Int = 5, sigma::Real = 1.0,
                            rho::Real = 0.9, L::Real = 10.0)
    @assert d >= 2 "ConfigMVN: d must be ≥ 2"
    @assert sigma > 0 "ConfigMVN: sigma must be positive"
    @assert -1.0 < rho < 1.0 "ConfigMVN: |rho| must be < 1"
    @assert L > 0 "ConfigMVN: L must be positive"
    mu = [0.5 * (-1.0)^(i + 1) for i in 1:d]
    return ConfigMVN(d, Float64(sigma), Float64(rho), mu, Float64(L))
end


"""
    build_log_f(cfg::ConfigMVN) -> Function

Returns `log_f(theta::AbstractVector) -> Real`, the *unnormalised* log
of the geometric factor `f` (the Gaussian quadratic form). The cube
prior is **not** baked in; `posterior_measure` adds it via the BAT
prior.

Σ is precomputed once and inverted via PDMats for stability.
"""
function build_log_f(cfg::ConfigMVN)
    d = cfg.d
    σ = cfg.sigma
    ρ = cfg.rho
    μ = copy(cfg.mu)
    Σ = [σ^2 * ρ^abs(i - j) for i in 1:d, j in 1:d]
    Σ_chol = cholesky(Symmetric(Σ))
    function log_f(theta::AbstractVector)
        # Solve Σ x = (theta - μ) without inverting; quadratic form follows.
        diff = theta .- μ
        y = Σ_chol \ diff
        return -0.5 * dot(diff, y)
    end
    return log_f
end


# -----------------------------------------------------------------------------
# Box prior on [-L, L]^d
# -----------------------------------------------------------------------------

"""
    prior_box(L, d) -> a `Distributions.Product` of d uniforms.

Used by all problems as the BAT.jl prior; defining it once here keeps
the convention uniform.
"""
function prior_box(L::Real, d::Integer)
    return Distributions.product_distribution([Uniform(-L, L) for _ in 1:d])
end

prior_box(cfg::ProblemConfig) = prior_box(cfg.L, cfg.d)


"""
    posterior_measure(cfg, log_f) -> BAT.PosteriorMeasure

Wraps `log_f` and the cube prior into a `PosteriorMeasure` that all
BAT.jl-based samplers (`mh`, `nuts`, `ns`) consume.

The likelihood passed to BAT calls `log_f(theta)` directly, so the
counter (which is `log_f` itself, see `LikelihoodCounter`) is exercised
on every primal/dual call.
"""
function posterior_measure(cfg::ProblemConfig, log_f)
    prior = prior_box(cfg)
    likelihood = BAT.logfuncdensity(log_f)
    return BAT.PosteriorMeasure(likelihood, prior)
end

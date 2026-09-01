# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# ProblemBanana.jl — Rosenbrock-style twisted Gaussian (Protocol §4.2)
# =============================================================================

"""
    ConfigBanana(d, sigma, b, L)

Twisted-Gaussian (Rosenbrock) target.

The latent vector ξ ~ N(0, diag(σ²)) is twisted to θ via
    θ_1 = ξ_1
    θ_2 = ξ_2 + b * ξ_1^2
    θ_j = ξ_j  for j ≥ 3
This concentrates the mass along the parabolic ridge θ_2 = b θ_1²
in the (θ_1, θ_2) plane.

Equivalently, the geometric factor is

    log f(θ) = -½ (θ_1/σ_1)^2
              - ½ ((θ_2 - b θ_1^2) / σ_2)^2
              - ½ Σ_{j≥3} (θ_j/σ_j)^2.

Fields:
- `d::Int`              ≥ 2
- `sigma::Vector{Float64}`  per-coordinate σ, length d
- `b::Float64`          twist strength (b = 1 is the canonical setting)
- `L::Float64`          cube half-width
"""
struct ConfigBanana <: ProblemConfig
    d::Int
    sigma::Vector{Float64}
    b::Float64
    L::Float64
end

"""
    make_config_banana(; d=5, sigma=fill(1.0,d), b=1.0, L=10.0) -> ConfigBanana
"""
function make_config_banana(; d::Int = 5,
                              sigma::AbstractVector{<:Real} = fill(1.0, d),
                              b::Real = 1.0,
                              L::Real = 10.0)
    @assert d >= 2 "ConfigBanana: d must be ≥ 2"
    @assert length(sigma) == d "ConfigBanana: sigma must have length d"
    @assert all(sigma .> 0) "ConfigBanana: sigma entries must be positive"
    @assert L > 0 "ConfigBanana: L must be positive"
    return ConfigBanana(d, Vector{Float64}(sigma), Float64(b), Float64(L))
end


"""
    build_log_f(cfg::ConfigBanana) -> Function
"""
function build_log_f(cfg::ConfigBanana)
    σ = cfg.sigma
    b = cfg.b
    d = cfg.d
    function log_f(theta::AbstractVector)
        s = -0.5 * (theta[1] / σ[1])^2
        s += -0.5 * ((theta[2] - b * theta[1]^2) / σ[2])^2
        @inbounds for j in 3:d
            s += -0.5 * (theta[j] / σ[j])^2
        end
        return s
    end
    return log_f
end

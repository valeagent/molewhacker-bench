# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# ProblemFunnel.jl — Neal's funnel (Protocol §4.3)
# =============================================================================

"""
    ConfigFunnel(d, sigma1, L)

Neal's funnel:
    θ_1 ~ N(0, σ_1²)
    θ_j | θ_1 ~ N(0, e^{θ_1})         for j = 2..d.

The (unnormalised) log-density is

    log f(θ) = -½ (θ_1/σ_1)² - ½(d-1) θ_1
              - ½ e^{-θ_1} Σ_{j≥2} θ_j².

The factor `-½(d-1) θ_1` comes from the d-1 transverse-coordinate
normalisations `1/sqrt(2π e^{θ_1})`.
"""
struct ConfigFunnel <: ProblemConfig
    d::Int
    sigma1::Float64
    L::Float64
end

"""
    make_config_funnel(; d=5, sigma1=3.0, L=10.0) -> ConfigFunnel
"""
function make_config_funnel(; d::Int = 5, sigma1::Real = 3.0, L::Real = 10.0)
    @assert d >= 2 "ConfigFunnel: d must be ≥ 2"
    @assert sigma1 > 0 "ConfigFunnel: sigma1 must be positive"
    @assert L > 0 "ConfigFunnel: L must be positive"
    return ConfigFunnel(d, Float64(sigma1), Float64(L))
end

"""
    build_log_f(cfg::ConfigFunnel) -> Function
"""
function build_log_f(cfg::ConfigFunnel)
    σ1 = cfg.sigma1
    d = cfg.d
    function log_f(theta::AbstractVector)
        θ1 = theta[1]
        s = -0.5 * (θ1 / σ1)^2
        s += -0.5 * (d - 1) * θ1
        ssq = zero(eltype(theta))
        @inbounds for j in 2:d
            ssq += theta[j]^2
        end
        s += -0.5 * exp(-θ1) * ssq
        return s
    end
    return log_f
end

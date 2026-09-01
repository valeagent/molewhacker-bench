# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# ProblemMRidges.jl — smooth M-ridges (Protocol §4.4)
# =============================================================================
#
# A bimodal "M" in θ_1 coupled, through a funnel-style scale and a
# curved ridge offset, to bimodal ridges in θ_2..θ_d. With the default
# parameters and d=5 the target has 2^5 = 32 modes.
#
# In the protocol's parameter list:
#   Δ   :  mode separation (M peaks at ±Δ in θ_1)
#   τ   :  mode width                  (= σ_1)
#   a   :  constant ridge offset       (mean of |θ_j| floor)
#   b   :  ridge curvature             (m(θ_1) = a + b θ_1²)
#   c   :  funnel exponent             (s(θ_1) = s_0 exp(c θ_1))
#   s0  :  base width                  (transverse base scale)
#   L   :  cube half-width
#
# Posterior:
#   p(θ) ∝ 1{θ ∈ [-L,L]^d} · ℓ_1(θ_1) · Π_{j=2}^d ℓ(θ_j | θ_1)
# with
#   ℓ_1(θ_1)        = ½ N(θ_1; +Δ, τ²) + ½ N(θ_1; -Δ, τ²)
#   ℓ(θ_j | θ_1)    = ½ N(θ_j; +m(θ_1), s(θ_1)²) + ½ N(θ_j; -m(θ_1), s(θ_1)²)
#   m(θ_1)          = a + b θ_1²
#   s(θ_1)          = s_0 exp(c θ_1).
#
# The implementation uses log-sum-exp on every mixture branch to avoid
# catastrophic cancellation when one mode dominates.

const _LOG2 = log(2.0)
const _LOG2PI = log(2π)

"""
    ConfigMRidges(d, Δ, τ, a, b, c, s0, L)

Smooth M-ridges problem configuration. See file header for the
mathematical model.
"""
struct ConfigMRidges <: ProblemConfig
    d::Int
    Δ::Float64
    τ::Float64
    a::Float64
    b::Float64
    c::Float64
    s0::Float64
    L::Float64
end

"""
    make_config_mridges(; d=5, Δ=2.0, τ=0.5, a=1.0, b=0.3, c=0.2, s0=0.5, L=10.0)
"""
function make_config_mridges(;
        d::Int = 5,
        Δ::Real = 2.0,
        τ::Real = 0.5,
        a::Real = 1.0,
        b::Real = 0.3,
        c::Real = 0.2,
        s0::Real = 0.5,
        L::Real = 10.0,
    )
    @assert d >= 2 "ConfigMRidges: d must be ≥ 2"
    @assert τ > 0 "ConfigMRidges: τ must be positive"
    @assert s0 > 0 "ConfigMRidges: s0 must be positive"
    @assert L > 0 "ConfigMRidges: L must be positive"
    return ConfigMRidges(d, Float64(Δ), Float64(τ), Float64(a), Float64(b),
                          Float64(c), Float64(s0), Float64(L))
end


# -----------------------------------------------------------------------------
# Inner numerics
# -----------------------------------------------------------------------------

@inline function _log_normal_pdf(x::Real, μ::Real, logσ::Real)
    σ = exp(logσ)
    z = (x - μ) / σ
    return -0.5 * z * z - logσ - 0.5 * _LOG2PI
end

@inline function _log_M(x::Real, Δ::Real, logτ::Real)
    # ½ N(x; +Δ, τ²) + ½ N(x; -Δ, τ²) on log scale
    a = _log_normal_pdf(x, Δ, logτ)
    b = _log_normal_pdf(x, -Δ, logτ)
    m = max(a, b)
    return m + log(0.5 * (exp(a - m) + exp(b - m)))
end

"""
    build_log_f(cfg::ConfigMRidges) -> Function
"""
function build_log_f(cfg::ConfigMRidges)
    Δ = cfg.Δ
    logτ = log(cfg.τ)
    a = cfg.a
    bcurve = cfg.b
    c = cfg.c
    logs0 = log(cfg.s0)
    d = cfg.d
    function log_f(theta::AbstractVector)
        θ1 = theta[1]
        s = _log_M(θ1, Δ, logτ)
        if d > 1
            log_s = logs0 + c * θ1
            m = a + bcurve * θ1 * θ1
            @inbounds for j in 2:d
                # log of ½ N(θ_j; +m, s²) + ½ N(θ_j; -m, s²)
                p_plus = _log_normal_pdf(theta[j], m, log_s)
                p_minus = _log_normal_pdf(theta[j], -m, log_s)
                pm = max(p_plus, p_minus)
                s += pm + log(0.5 * (exp(p_plus - pm) + exp(p_minus - pm)))
            end
        end
        return s
    end
    return log_f
end

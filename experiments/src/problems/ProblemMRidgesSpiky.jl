# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# ProblemMRidgesSpiky.jl — controllable spiky multimodal target
# =============================================================================
#
# Protocol §4.7.1 (rev May 2026, V3-T-1).
#
# `mridges_spiky(K, M, σ_spike, d)` is a separable target on
# [-L, L]^d whose first coordinate is a `K × M` mixture of bumps:
#
#   p_1(θ_1) ∝ Σ_{k=1}^K w_k Σ_{m=1}^M κ((θ_1 − μ_{km}) / σ_spike)
#
# with `K = 5` outer ridges at `μ_k ∈ {-Δ_K, -Δ_K/2, 0, +Δ_K/2, +Δ_K}`
# (`Δ_K = 2.0`, matches v2 `mridges`), `M ∈ {1, 4}` Gaussian / Student-t
# spikes per ridge with means `μ_km = μ_k + δ_m`, `δ_m = 0.5 σ_jitter
# cos(2π m / M)`, `σ_jitter = 1.5`. Equal weights `w_k = 1/K`. The
# remaining `d − 1` coordinates are i.i.d. isotropic Gaussian with
# width `σ_⊥` (the v2 `mridges` τ; default 0.5).
#
# This file defines the *log-target*. Truth (analytical mixture moments,
# logZ via 1-D quadrature, sample generation by mixture-component
# inverse-CDF + cube rejection on the perpendicular axes) lives in
# `truths/truth_mridges_spiky.jl`.

const _LOG2PI_MRSP = log(2π)

"""
    ConfigMRidgesSpiky(d, K, M, σ_spike, σ_⊥, σ_jitter, Δ_K, kernel, ν, L)

`kernel = :gaussian` (headline grid) or `:studentt3` (Appendix B
stress test). `ν = 3` is the Student-t degrees-of-freedom for the
stress kernel; ignored for `:gaussian`.
"""
struct ConfigMRidgesSpiky <: ProblemConfig
    d::Int
    K::Int
    M::Int
    σ_spike::Float64
    σ_perp::Float64
    σ_jitter::Float64
    Δ_K::Float64
    kernel::Symbol
    ν::Float64
    L::Float64
end

"""
    make_config_mridges_spiky(; d=5, K=5, M=4, σ_spike=0.1, σ_perp=0.5,
                                σ_jitter=1.5, Δ_K=2.0, kernel=:gaussian,
                                ν=3.0, L=10.0)

The default `σ_spike = 0.1, M = 4` is the headline `mridges_spiky`
setting that is also lifted into the dimensional sweep (Protocol
§6.3).
"""
function make_config_mridges_spiky(;
        d::Integer = 5,
        K::Integer = 5,
        M::Integer = 4,
        σ_spike::Real = 0.1,
        σ_perp::Real = 0.5,
        σ_jitter::Real = 1.5,
        Δ_K::Real = 2.0,
        kernel::Symbol = :gaussian,
        ν::Real = 3.0,
        L::Real = 10.0,
    )
    @assert d >= 2 "ConfigMRidgesSpiky: d must be ≥ 2"
    @assert K >= 1 "ConfigMRidgesSpiky: K must be ≥ 1"
    @assert M >= 1 "ConfigMRidgesSpiky: M must be ≥ 1"
    @assert σ_spike > 0 "ConfigMRidgesSpiky: σ_spike must be positive"
    @assert σ_perp > 0 "ConfigMRidgesSpiky: σ_perp must be positive"
    @assert L > 0 "ConfigMRidgesSpiky: L must be positive"
    @assert kernel in (:gaussian, :studentt3) "ConfigMRidgesSpiky: kernel must be :gaussian or :studentt3"
    @assert ν > 0 "ConfigMRidgesSpiky: ν must be positive"
    return ConfigMRidgesSpiky(Int(d), Int(K), Int(M),
                              Float64(σ_spike), Float64(σ_perp),
                              Float64(σ_jitter), Float64(Δ_K),
                              kernel, Float64(ν), Float64(L))
end


# -----------------------------------------------------------------------------
# Mixture geometry helpers
# -----------------------------------------------------------------------------

"""
    mridges_spiky_outer_means(cfg) -> Vector{Float64}

The K outer-ridge centres `μ_k`, evenly spaced in [-Δ_K, +Δ_K].
"""
function mridges_spiky_outer_means(cfg::ConfigMRidgesSpiky)
    K = cfg.K
    if K == 1
        return [0.0]
    end
    return collect(range(-cfg.Δ_K, cfg.Δ_K; length = K))
end

"""
    mridges_spiky_jitters(cfg) -> Vector{Float64}

The M jitter offsets `δ_m = 0.5 σ_jitter cos(2π m / M)`.
"""
function mridges_spiky_jitters(cfg::ConfigMRidgesSpiky)
    M = cfg.M
    M == 1 && return [0.0]
    return [0.5 * cfg.σ_jitter * cos(2π * m / M) for m in 1:M]
end

"""
    mridges_spiky_modes_theta1(cfg) -> Vector{Float64}

Length `K * M` vector of all spike centres along `θ_1`.
"""
function mridges_spiky_modes_theta1(cfg::ConfigMRidgesSpiky)
    μs = mridges_spiky_outer_means(cfg)
    δs = mridges_spiky_jitters(cfg)
    out = Float64[]
    for μ in μs, δ in δs
        push!(out, μ + δ)
    end
    return out
end

"""
    mridges_spiky_modes(cfg) -> Matrix{Float64}  (d × K*M)

Mode list for the mode-recovery metric: the spike centres along
`θ_1`, with the remaining coordinates pinned at their conditional
mean (zero — the perpendicular directions are zero-mean Gaussians).
"""
function mridges_spiky_modes(cfg::ConfigMRidgesSpiky)
    centres = mridges_spiky_modes_theta1(cfg)
    Kmod = length(centres)
    M = zeros(cfg.d, Kmod)
    @inbounds for k in 1:Kmod
        M[1, k] = centres[k]
    end
    return M
end


# -----------------------------------------------------------------------------
# Spike kernel evaluation (log-density)
# -----------------------------------------------------------------------------

@inline function _log_kernel_gaussian(z::Real)
    return -0.5 * z * z - 0.5 * _LOG2PI_MRSP
end

@inline function _log_kernel_studentt(z::Real, ν::Real)
    # Standard Student-t pdf on z = (x - μ) / σ. The (1/σ) Jacobian
    # term is added at the call site so this returns the log of the
    # standardised density only.
    return logpdf(TDist(ν), z)
end

"""
    _log_spike_kernel(z, cfg) -> Float64

Log-density of the *standardised* spike kernel evaluated at `z`. To
get the un-standardised density at `(x − μ) / σ` subtract `log σ`.
"""
@inline function _log_spike_kernel(z::Real, cfg::ConfigMRidgesSpiky)
    return cfg.kernel == :gaussian ? _log_kernel_gaussian(z) :
           _log_kernel_studentt(z, cfg.ν)
end


# -----------------------------------------------------------------------------
# Log-target factory
# -----------------------------------------------------------------------------

"""
    build_log_f(cfg::ConfigMRidgesSpiky) -> Function

Returns `log_f(θ::AbstractVector{<:Real}) -> Real` where `θ` lies in
the cube `[-L, L]^d`. The marginal in the first coordinate is the
`K × M` mixture of spikes; the remaining coordinates are i.i.d.
Gaussian with width `σ_⊥`.
"""
function build_log_f(cfg::ConfigMRidgesSpiky)
    centres = mridges_spiky_modes_theta1(cfg)
    Kmod = length(centres)
    log_K = log(Float64(Kmod))
    log_σ_spike = log(cfg.σ_spike)
    log_σ_perp = log(cfg.σ_perp)
    inv_σ_perp_sq = 1.0 / (cfg.σ_perp * cfg.σ_perp)
    d = cfg.d
    function log_f(θ::AbstractVector{<:Real})
        # All accumulators must be initialised at `eltype(θ)` so the
        # Union{Float64,ForwardDiff.Dual} type-instability that breaks
        # `is_log_zero` in BAT.jl (and silently kills MW's `bat_findmode`
        # init step) cannot occur.
        T = eltype(θ)
        θ1 = θ[1]
        max_lp = T(-Inf)
        for c in centres
            z = (θ1 - c) / cfg.σ_spike
            lp = _log_spike_kernel(z, cfg) - log_σ_spike
            if lp > max_lp
                max_lp = lp
            end
        end
        s = zero(T)
        @inbounds for c in centres
            z = (θ1 - c) / cfg.σ_spike
            lp = _log_spike_kernel(z, cfg) - log_σ_spike
            s += exp(lp - max_lp)
        end
        log_marginal = max_lp + log(s) - log_K
        log_perp = zero(T)
        if d > 1
            ssq = zero(T)
            @inbounds for j in 2:d
                ssq += θ[j] * θ[j]
            end
            log_perp = -0.5 * inv_σ_perp_sq * ssq -
                       (d - 1) * (log_σ_perp + 0.5 * _LOG2PI_MRSP)
        end
        return log_marginal + log_perp
    end
    return log_f
end

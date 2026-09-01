# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# truth_mridges.jl — Truth for ConfigMRidges (Protocol §4.4, §6.21)
# =============================================================================
#
# Composition strategy:
#   1) Draw θ_1 from the bimodal "M": each peak with prob ½, then
#      Normal(±Δ, τ²) restricted to [-L, L].
#   2) For j ≥ 2, given θ_1 draw θ_j from a bimodal Normal(±m(θ_1),
#      s(θ_1)²) restricted to [-L, L].
#
# logZ via 1D quadrature on θ_1.

import QuadGK

# Internal: log of the per-θ_j cube-restricted mass given θ_1. The
# conditional density is ½ N(θ_j; +m, s²) + ½ N(θ_j; -m, s²); its
# integral over [-L, L] is
#    ½ [Φ((L-m)/s) - Φ((-L-m)/s)] + ½ [Φ((L+m)/s) - Φ((-L+m)/s)]
@inline function _log_cond_mass(θ1::Real, cfg::ConfigMRidges)
    a, bcurve, c, s0 = cfg.a, cfg.b, cfg.c, cfg.s0
    L = cfg.L
    s = s0 * exp(c * θ1)
    m = a + bcurve * θ1 * θ1
    p_plus = 0.5 * (cdf(Normal(0, 1), (L - m) / s) - cdf(Normal(0, 1), (-L - m) / s))
    p_minus = 0.5 * (cdf(Normal(0, 1), (L + m) / s) - cdf(Normal(0, 1), (-L + m) / s))
    return log(p_plus + p_minus)
end

@inline function _log_marginal_theta1_mr(θ1::Real, cfg::ConfigMRidges)
    Δ, τ = cfg.Δ, cfg.τ
    # Bimodal M density at θ_1
    p_M = 0.5 * pdf(Normal(Δ, τ), θ1) + 0.5 * pdf(Normal(-Δ, τ), θ1)
    # (d-1) copies of conditional cube mass
    return log(p_M) + (cfg.d - 1) * _log_cond_mass(θ1, cfg)
end


"""
    compute_truth_mridges(cfg::ConfigMRidges; N_ref=1_000_000, seed)
"""
function compute_truth_mridges(cfg::ConfigMRidges;
                                  N_ref::Integer = 1_000_000,
                                  seed::Integer = 20260501)
    d = cfg.d
    L = cfg.L

    # 1) Tabulate the marginal of θ_1 on a fine grid for sample-truth.
    n_grid = 8192
    θ1_grid = collect(range(-L, L; length = n_grid))
    log_p1 = [_log_marginal_theta1_mr(t, cfg) for t in θ1_grid]
    log_max = maximum(log_p1)
    p1 = exp.(log_p1 .- log_max)
    cdf_unnorm = zeros(n_grid)
    for i in 2:n_grid
        cdf_unnorm[i] = cdf_unnorm[i - 1] +
                        0.5 * (p1[i] + p1[i - 1]) * (θ1_grid[i] - θ1_grid[i - 1])
    end
    Z1 = cdf_unnorm[end]
    cdf_grid = cdf_unnorm ./ Z1

    # 2) Sample-truth via composition.
    samples = _sample_mridges_via_grid(cfg, N_ref, θ1_grid, cdf_grid; seed = seed)

    μ, Σ = _empirical_moments(samples)
    Q = _empirical_quantiles(samples)

    # 3) logZ = ∫_{-L}^{+L} marginal(θ_1) dθ_1
    log_int = log_max + log(Z1)
    logZ = log_int - d * log(2L)

    extras = Dict{Symbol,Any}(
        :theta1_grid => θ1_grid,
        :theta1_cdf => cdf_grid,
    )

    return TruthSet(:mridges, d, cfg, samples, μ, Σ, Q, logZ, extras)
end


"""
    sample_truth_mridges(cfg, N; seed) -> d × N
"""
function sample_truth_mridges(cfg::ConfigMRidges, N::Integer; seed::Integer = 20260601)
    L = cfg.L
    n_grid = 8192
    θ1_grid = collect(range(-L, L; length = n_grid))
    log_p1 = [_log_marginal_theta1_mr(t, cfg) for t in θ1_grid]
    p1 = exp.(log_p1 .- maximum(log_p1))
    cdf_unnorm = zeros(n_grid)
    for i in 2:n_grid
        cdf_unnorm[i] = cdf_unnorm[i - 1] +
                        0.5 * (p1[i] + p1[i - 1]) * (θ1_grid[i] - θ1_grid[i - 1])
    end
    cdf_grid = cdf_unnorm ./ cdf_unnorm[end]
    return _sample_mridges_via_grid(cfg, N, θ1_grid, cdf_grid; seed = seed)
end


function _sample_mridges_via_grid(cfg::ConfigMRidges, N::Integer,
                                    θ1_grid::AbstractVector,
                                    cdf::AbstractVector;
                                    seed::Integer)
    d = cfg.d
    L = cfg.L
    a, bcurve, c, s0 = cfg.a, cfg.b, cfg.c, cfg.s0
    rng = Random.MersenneTwister(seed)
    out = Matrix{Float64}(undef, d, N)
    written = 0
    while written < N
        u = rand(rng)
        i = searchsortedfirst(cdf, u)
        i = clamp(i, 2, length(cdf))
        c0 = cdf[i - 1]
        c1 = cdf[i]
        t0 = θ1_grid[i - 1]
        t1 = θ1_grid[i]
        frac = (u - c0) / max(c1 - c0, eps())
        θ1 = t0 + frac * (t1 - t0)
        if abs(θ1) >= L
            continue
        end
        s = s0 * exp(c * θ1)
        m = a + bcurve * θ1 * θ1
        ok = true
        rest = Vector{Float64}(undef, d - 1)
        @inbounds for j in 1:(d - 1)
            sign_branch = rand(rng) < 0.5 ? 1.0 : -1.0
            # Sample θ_j from Normal(sign_branch * m, s²) restricted to [-L, L]
            attempts = 0
            x = 0.0
            while attempts < 10_000
                x = sign_branch * m + s * randn(rng)
                if abs(x) < L
                    break
                end
                attempts += 1
            end
            if abs(x) >= L
                ok = false
                break
            end
            rest[j] = x
        end
        ok || continue
        written += 1
        out[1, written] = θ1
        @inbounds for j in 2:d
            out[j, written] = rest[j - 1]
        end
    end
    return out
end

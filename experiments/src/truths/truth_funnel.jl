# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# truth_funnel.jl — Truth for ConfigFunnel (Protocol §4.3, Appendix A.3.3)
# =============================================================================
#
# Hierarchical model:
#   θ_1 ~ N(0, σ_1²)
#   θ_j | θ_1 ~ N(0, exp(θ_1))   for j = 2..d
# restricted to the cube [-L, L]^d.
#
# Truth strategy:
#   * sample by composition: draw θ_1 from a tabulated marginal CDF
#     (which has been corrected for the cube), then draw θ_j | θ_1
#     iid from the *truncated* normal N(0, e^{θ_1}) restricted to
#     (-L, L), by per-coordinate rejection.
#   * 1D quadrature for log Z.
#
# V8-FIX-A1 (Jul 2026): the previous sampler rejected the WHOLE tuple
# (restarting θ_1) whenever any transverse coordinate fell outside the
# cube. Because the tabulated θ_1 marginal already contains the
# transverse truncation factor erf(L/√(2 e^{θ_1}))^{d-1}, the tuple
# rejection applied that factor a second time: the accepted θ_1 density
# was ∝ p_1(θ_1)·erf(·)^{d-1} — erf *squared* relative to the target
# (verified numerically: at d=5, σ_1=3, L=10 the 97.5% θ_1 quantile was
# shifted by −0.39). The fix resamples only the offending coordinate,
# which draws θ_j | θ_1 exactly from the truncated normal and leaves
# the θ_1 marginal untouched.

import QuadGK

# Internal: log of the marginal of the *unnormalised* funnel target
# `exp(log_f(θ))` after integrating out θ_2..θ_d ∈ [-L, L]. This
# matches the convention used by `build_log_f(cfg::ConfigFunnel)`,
# which (like every other ProblemX/build_log_f in this protocol) drops
# the Gaussian normalising constants. Without this fix the algorithms
# (IS, NS, MW) — which estimate `log ∫_cube exp(log_f) dθ / (2L)^d` —
# disagree with `truth.logZ` by a constant
# `0.5·d·log(2π) + log(σ_1)` (≈ 5.69 nats for d = 5, σ_1 = 3,
# explaining the V4-FIX-2 symptom).
#
# Derivation. log_f(θ) = -½(θ_1/σ_1)² - ½(d-1)θ_1 - ½e^{-θ_1} Σ_{j≥2} θ_j².
# ∫_{[-L,L]^{d-1}} exp(log_f) dθ_2..dθ_d
#   = exp(-½(θ_1/σ_1)²) · exp(-½(d-1)θ_1) ·
#     [√(2π e^{θ_1}) · erf(L / √(2 e^{θ_1}))]^{d-1}
#   = exp(-½(θ_1/σ_1)²) · (2π)^{(d-1)/2} · erf(L / √(2 e^{θ_1}))^{d-1}
# (the e^{0.5(d-1)θ_1} from each √(σ_j²)= e^{θ_1/2} cancels the −½(d-1)θ_1 prefactor).
function _log_marginal_theta1(θ1::Real, cfg::ConfigFunnel)
    σ1 = cfg.sigma1
    # Unnormalised marginal: −½(θ_1/σ_1)² + 0.5·(d-1)·log(2π) +
    # (d-1)·log(erf(L/√(2 e^{θ_1}))).
    lp = -0.5 * (θ1 / σ1)^2 + 0.5 * (cfg.d - 1) * log(2π)
    z = cfg.L / sqrt(2 * exp(θ1))
    lerf = log(_erf(z))
    return lp + (cfg.d - 1) * lerf
end

# Pure-Julia erf (avoids dragging in SpecialFunctions for one call)
@inline _erf(x::Real) = sign(x) * _erf_pos(abs(x))
function _erf_pos(x::Real)
    # Abramowitz & Stegun 7.1.26 — adequate for our quadrature
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    t = 1.0 / (1.0 + p * x)
    return 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * exp(-x * x)
end


"""
    compute_truth_funnel(cfg::ConfigFunnel; N_ref=1_000_000, seed)

V4-FIX-2 (May 2026): the previous implementation computed `logZ` by
trapezoidal integration on a 4 096-point grid, which under-resolved
the sharp peak of the marginal `p_1(θ_1)` near zero and over-cut the
tails — `is`/`ns`/`mw` all reported `logẐ ≈ truth.logZ + 5.5` on the
funnel. The fix replaces the trapezoid with `QuadGK.quadgk` at
`rtol = 1e-12, order = 21, maxevals = 10^7`, and rescales the
integrand against `log_max` for numerical stability. The 4 096-point
grid is retained only for inverse-CDF sampling, where modest
precision is fine.
"""
function compute_truth_funnel(cfg::ConfigFunnel; N_ref::Integer = 1_000_000,
                                                    seed::Integer = 20260501)
    d = cfg.d
    L = cfg.L

    # 1) Sampling grid for inverse-CDF on θ_1. Trapezoidal CDF here is
    #    fine — sample-truth resolution is dominated by `N_ref`, not by
    #    grid noise.
    n_grid = 4096
    θ1_grid = collect(range(-L, L; length = n_grid))
    log_p1_grid = [_log_marginal_theta1(t, cfg) for t in θ1_grid]
    log_max = maximum(log_p1_grid)
    p1 = exp.(log_p1_grid .- log_max)
    cdf_unnorm = zeros(n_grid)
    for i in 2:n_grid
        cdf_unnorm[i] = cdf_unnorm[i - 1] +
                        0.5 * (p1[i] + p1[i - 1]) * (θ1_grid[i] - θ1_grid[i - 1])
    end
    cdf = cdf_unnorm ./ cdf_unnorm[end]   # normalised CDF on the grid

    # 2) Sample i.i.d. truth via composition (uses the grid CDF).
    samples = _sample_funnel_via_grid(cfg, N_ref, θ1_grid, cdf; seed = seed)

    # 3) Empirical moments / quantiles from the truth samples.
    μ, Σ = _empirical_moments(samples)
    Q = _empirical_quantiles(samples)

    # 4) High-precision logZ via QuadGK on the rescaled marginal
    #    integrand, `θ_1 → exp(_log_marginal_theta1(θ_1) − log_max)`.
    integrand_rescaled(θ1) = exp(_log_marginal_theta1(θ1, cfg) - log_max)
    Z1, Z1_err = QuadGK.quadgk(integrand_rescaled, -L, L;
                                  rtol = 1e-12, atol = 0.0,
                                  order = 21, maxevals = 10^7)
    log_int = log_max + log(Z1)
    # Posterior-evidence relative to uniform cube prior (Protocol §5.1).
    logZ = log_int - d * log(2L)

    extras = Dict{Symbol,Any}(
        :theta1_grid       => θ1_grid,
        :theta1_cdf        => cdf,
        :marginal_log_norm => log_int,
        :logZ_quad_abs_err => Z1_err,
        :logZ_method       => "quadgk(rtol=1e-12, order=21)",
    )

    return TruthSet(:funnel, d, cfg, samples, μ, Σ, Q, logZ, extras)
end


"""
    sample_truth_funnel(cfg, N; seed) -> d × N

Composition sampler. Used by `compute_truth_funnel` and by callers
who want only sample-truth without recomputing the truth set.
"""
function sample_truth_funnel(cfg::ConfigFunnel, N::Integer; seed::Integer = 20260601)
    L = cfg.L
    n_grid = 4096
    θ1_grid = collect(range(-L, L; length = n_grid))
    log_p1 = [_log_marginal_theta1(t, cfg) for t in θ1_grid]
    p1 = exp.(log_p1 .- maximum(log_p1))
    cdf_unnorm = zeros(n_grid)
    for i in 2:n_grid
        cdf_unnorm[i] = cdf_unnorm[i - 1] +
                        0.5 * (p1[i] + p1[i - 1]) * (θ1_grid[i] - θ1_grid[i - 1])
    end
    cdf = cdf_unnorm ./ cdf_unnorm[end]
    return _sample_funnel_via_grid(cfg, N, θ1_grid, cdf; seed = seed)
end


function _sample_funnel_via_grid(cfg::ConfigFunnel, N::Integer,
                                  θ1_grid::AbstractVector,
                                  cdf::AbstractVector;
                                  seed::Integer)
    d = cfg.d
    L = cfg.L
    rng = Random.MersenneTwister(seed)
    out = Matrix{Float64}(undef, d, N)
    written = 0
    while written < N
        u = rand(rng)
        i = searchsortedfirst(cdf, u)
        i = clamp(i, 2, length(cdf))
        # Linear interp between grid points
        c0 = cdf[i - 1]
        c1 = cdf[i]
        t0 = θ1_grid[i - 1]
        t1 = θ1_grid[i]
        frac = (u - c0) / max(c1 - c0, eps())
        θ1 = t0 + frac * (t1 - t0)
        if abs(θ1) >= L
            continue
        end
        σj = sqrt(exp(θ1))
        written += 1
        out[1, written] = θ1
        # V8-FIX-A1: draw each transverse coordinate from the truncated
        # normal N(0, σj²) on (-L, L) by per-coordinate rejection. Never
        # restart θ_1 — its tabulated marginal already accounts for the
        # cube truncation of the transverse block.
        @inbounds for j in 2:d
            xj = σj * randn(rng)
            while abs(xj) >= L
                xj = σj * randn(rng)
            end
            out[j, written] = xj
        end
    end
    return out
end

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# truth_shell.jl — Truth for ConfigShell (Protocol §4.5, Appendix A.3.5)
# =============================================================================
#
# The Gaussian shell density depends only on r = ‖θ‖. The radial truth
# distribution has density (up to a normalising constant)
#
#     p(r) ∝ r^{d-1} exp(-½ ((r - ρ)/w)²)             for r ≥ 0.
#
# Sampling strategy: draw r from this 1D density by inverse-CDF on a
# fine tabulated grid, then place at a uniformly random direction on
# S^{d-1}. Reject if any |θ_j| > L (negligible rate for L ≫ ρ + 5w).
#
# V8-FIX-A2 (Jul 2026): the previous radial sampler used rejection from
# a Normal(r_mode, σ_q²) proposal whose acceptance ratio
# p(r)/(q_full(r)·p_mode) exceeded 1 near the mode (≈ 2.8 at the peak
# for σ_q = w√d), so the ratio was silently clipped and the accepted
# radial law followed the too-wide *proposal* near the peak instead of
# the target (verified numerically: radial std 0.598 vs true 0.486 at
# d=5, ρ=4, w=0.5). The inverse-CDF grid draw below is exact and
# deterministic in the same way as the funnel/mridges truths.

import QuadGK

@inline function _log_radial_density(r::Real, cfg::ConfigShell)
    if r < 0
        return -Inf
    end
    return (cfg.d - 1) * log(max(r, eps())) - 0.5 * ((r - cfg.ρ) / cfg.w)^2
end


"""
    compute_truth_shell(cfg::ConfigShell; N_ref=1_000_000, seed)
"""
function compute_truth_shell(cfg::ConfigShell; N_ref::Integer = 1_000_000,
                                                  seed::Integer = 20260501)
    d = cfg.d
    ρ, w, L = cfg.ρ, cfg.w, cfg.L

    # 1) Sample-truth via radial-density sampling + isotropic direction.
    samples = sample_truth_shell(cfg, N_ref; seed = seed)

    μ, Σ = _empirical_moments(samples)
    Q = _empirical_quantiles(samples)

    # 2) logZ = log ∫_{cube} f(θ) dθ - d log(2L)
    #     ≈ log[A_d * ∫_0^∞ p(r) dr]      (for L ≫ ρ + 5w)
    # where A_d = 2 π^{d/2} / Γ(d/2) is the (d-1)-sphere surface area.
    A_d = _surface_area_d(d)
    radial_int, _ = QuadGK.quadgk(r -> exp(_log_radial_density(r, cfg)),
                                    0.0, ρ + 10w; rtol = 1e-10)
    log_int = log(A_d) + log(radial_int)
    logZ = log_int - d * log(2L)

    extras = Dict{Symbol,Any}(
        :surface_area_d => A_d,
        :radial_integral => radial_int,
    )

    return TruthSet(:shell, d, cfg, samples, μ, Σ, Q, logZ, extras)
end


"""
    sample_truth_shell(cfg, N; seed) -> d × N

Composition: r from radial density, direction uniform on S^{d-1}.
"""
function sample_truth_shell(cfg::ConfigShell, N::Integer; seed::Integer = 20260601)
    d = cfg.d
    ρ, w, L = cfg.ρ, cfg.w, cfg.L
    rng = Random.MersenneTwister(seed)

    # V8-FIX-A2: exact inverse-CDF draw of r from the tabulated radial
    # density. The grid spans [max(0, ρ−12w), ρ+12w]; the density mass
    # outside is < exp(-72) of the total and is negligible by the same
    # argument as the logZ quadrature above.
    n_grid = 8192
    r_lo = max(0.0, ρ - 12w)
    r_hi = ρ + 12w
    r_grid = collect(range(r_lo, r_hi; length = n_grid))
    log_p = [_log_radial_density(r, cfg) for r in r_grid]
    p = exp.(log_p .- maximum(log_p))
    cdf_unnorm = zeros(n_grid)
    for i in 2:n_grid
        cdf_unnorm[i] = cdf_unnorm[i - 1] +
                        0.5 * (p[i] + p[i - 1]) * (r_grid[i] - r_grid[i - 1])
    end
    cdf = cdf_unnorm ./ cdf_unnorm[end]

    out = Matrix{Float64}(undef, d, N)
    written = 0
    while written < N
        # Inverse-CDF draw of r (linear interpolation between grid nodes)
        u = rand(rng)
        i = clamp(searchsortedfirst(cdf, u), 2, n_grid)
        c0, c1 = cdf[i - 1], cdf[i]
        frac = (u - c0) / max(c1 - c0, eps())
        r = r_grid[i - 1] + frac * (r_grid[i] - r_grid[i - 1])
        # Uniform direction on S^{d-1}
        v = randn(rng, d)
        v ./= norm(v)
        x = r .* v
        if all(abs.(x) .< L)
            written += 1
            @inbounds for j in 1:d
                out[j, written] = x[j]
            end
        end
    end
    return out
end

@inline function _surface_area_d(d::Integer)
    # A_d = 2 π^{d/2} / Γ(d/2)
    return 2 * pi^(d / 2) / _gamma_simple(d / 2)
end

# Stirling-good Γ for moderate arguments; no SpecialFunctions dep.
function _gamma_simple(x::Real)
    if x == 0.5
        return sqrt(pi)
    elseif x == 1.0
        return 1.0
    elseif x == 1.5
        return 0.5 * sqrt(pi)
    elseif x == 2.0
        return 1.0
    elseif x == 2.5
        return 0.75 * sqrt(pi)
    elseif x == 3.0
        return 2.0
    elseif x == 5.0
        return 24.0
    end
    # Lanczos approximation
    g = 7
    p = (
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    )
    if x < 0.5
        return pi / (sin(pi * x) * _gamma_simple(1 - x))
    else
        x -= 1
        a = p[1]
        t = x + g + 0.5
        for i in 2:length(p)
            a += p[i] / (x + i - 1)
        end
        return sqrt(2π) * t^(x + 0.5) * exp(-t) * a
    end
end

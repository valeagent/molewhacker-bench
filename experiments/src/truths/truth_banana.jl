# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# truth_banana.jl — Truth for ConfigBanana (Protocol §4.2, Appendix A.3.2)
# =============================================================================
#
# Latent space:  ξ ~ N(0, diag(σ²))
# Forward map:   T_b(ξ) = (ξ_1, ξ_2 + b ξ_1², ξ_3, …, ξ_d)
# This map is volume-preserving (Jacobian = 1).
#
# Cube-restricted: keep θ = T_b(ξ) iff θ ∈ [-L, L]^d. Sampling is
# composition + rejection.
#
# Closed forms in the unrestricted (no-cube) limit:
#   E[θ_1]   = 0
#   E[θ_2]   = b σ_1²       (because E[ξ_1²] = σ_1²)
#   Var θ_1  = σ_1²
#   Var θ_2  = σ_2² + 2 b² σ_1⁴
#   Cov θ_1, θ_2 = 0
#
# These hold exactly without the cube; with the cube and the default
# parameters (σ = 1, b = 1, L = 10) the rejection rate is ~0.2%
# (P(θ₂ = ξ₂ + ξ₁² > 10) ≈ 1.8e-3), which shifts E[θ₂] by ~0.02 and
# Var[θ₂] by ~7%, so we report the cube-truncated empirical moments
# from the truth samples instead of the unrestricted closed forms.

"""
    compute_truth_banana(cfg::ConfigBanana; N_ref=1_000_000, seed)
"""
function compute_truth_banana(cfg::ConfigBanana; N_ref::Integer = 1_000_000,
                                                    seed::Integer = 20260501)
    d = cfg.d
    σ = cfg.sigma
    b = cfg.b
    L = cfg.L

    samples = sample_truth_banana(cfg, N_ref; seed = seed)

    # Empirical moments (cube-truncated) — use these as the truth for
    # the metrics. They differ from the analytical unrestricted values
    # by a tiny amount but those values are the truth for the *target*.
    μ, Σ = _empirical_moments(samples)
    Q = _empirical_quantiles(samples)

    # log Z — analytical for the unrestricted (no-cube) Banana
    # because |J(T_b)| = 1, so logZ_unrestricted = sum_j(log σ_j) +
    # 0.5 d log(2π). Add the truncation correction estimated by MC:
    # the fraction of samples (drawn before cube rejection) that fall
    # in the cube.
    logZ_gauss = sum(log.(σ)) + 0.5 * d * log(2π)
    rng_aux = Random.MersenneTwister(seed + 31337)
    n_total_attempt = 200_000
    accepted = 0
    @inbounds for _ in 1:n_total_attempt
        ξ1 = σ[1] * randn(rng_aux)
        ξ2 = σ[2] * randn(rng_aux)
        θ1 = ξ1
        θ2 = ξ2 + b * ξ1^2
        in_cube = (abs(θ1) < L) & (abs(θ2) < L)
        if in_cube
            ok = true
            for j in 3:d
                if abs(σ[j] * randn(rng_aux)) >= L
                    ok = false
                    break
                end
            end
            if ok
                accepted += 1
            end
        end
    end
    p_in_cube = max(accepted / n_total_attempt, 1e-12)
    logZ_in_cube = logZ_gauss + log(p_in_cube)
    logZ = logZ_in_cube - d * log(2L)

    extras = Dict{Symbol,Any}(
        :p_in_cube => p_in_cube,
        :logZ_unrestricted => logZ_gauss,
    )

    return TruthSet(:banana, d, cfg, samples, μ, Σ, Q, logZ, extras)
end


"""
    sample_truth_banana(cfg, N; seed) -> d × N

Composition sampling: draw ξ ~ N(0, diag(σ²)), apply T_b, reject if
outside the cube.
"""
function sample_truth_banana(cfg::ConfigBanana, N::Integer; seed::Integer = 20260601)
    d = cfg.d
    σ = cfg.sigma
    b = cfg.b
    L = cfg.L
    rng = Random.MersenneTwister(seed)
    out = Matrix{Float64}(undef, d, N)
    written = 0
    while written < N
        ξ1 = σ[1] * randn(rng)
        θ1 = ξ1
        if abs(θ1) >= L
            continue
        end
        ξ2 = σ[2] * randn(rng)
        θ2 = ξ2 + b * ξ1^2
        if abs(θ2) >= L
            continue
        end
        ok = true
        rest = Vector{Float64}(undef, d - 2)
        @inbounds for j in 3:d
            x = σ[j] * randn(rng)
            if abs(x) >= L
                ok = false
                break
            end
            rest[j - 2] = x
        end
        ok || continue
        written += 1
        out[1, written] = θ1
        out[2, written] = θ2
        @inbounds for j in 3:d
            out[j, written] = rest[j - 2]
        end
    end
    return out
end

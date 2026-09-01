# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# truth_mvn.jl — Truth for ConfigMVN (Protocol §4.1, Appendix A.3.1)
# =============================================================================
#
# The target is a multivariate Gaussian N(μ, Σ) restricted to the cube
# [-L, L]^d. With L = 10 and σ = 1, the cube-tail probability is
# < 1e-22 per dimension, so for all practical purposes the truth is
# the unrestricted Gaussian. We still rejection-sample to stay honest.

"""
    compute_truth_mvn(cfg::ConfigMVN; N_ref=1_000_000, seed)
"""
function compute_truth_mvn(cfg::ConfigMVN; N_ref::Integer = 1_000_000,
                                              seed::Integer = 20260501)
    d = cfg.d
    σ, ρ = cfg.sigma, cfg.rho
    Σ_full = [σ^2 * ρ^abs(i - j) for i in 1:d, j in 1:d]
    Σ_sym = Symmetric(Σ_full)

    samples = sample_truth_mvn(cfg, N_ref; seed = seed)

    # Analytical first moments hold to 1e-22 within the cube
    μ_truth = copy(cfg.mu)
    Σ_truth = Matrix(Σ_sym)
    Q = Matrix{Float64}(undef, d, length(QUANTILE_LEVELS))
    for j in 1:d
        for (k, p) in enumerate(QUANTILE_LEVELS)
            Q[j, k] = quantile(Normal(cfg.mu[j], σ), p)
        end
    end

    # log Z = log ∫_{[-L,L]^d} f(θ) dθ - d * log(2L)  [the unnormalised f
    # has Gaussian normalising constant Z_full = (2π)^{d/2} sqrt(det Σ);
    # the cube prior is uniform on (2L)^d so the *posterior-evidence*
    # cancels the (2L)^d factor that the cube prior introduces — the
    # quantity reported here is the standard "model evidence" with
    # respect to the unit-mass cube prior].
    logdetΣ = logdet(Σ_sym)
    logZ_full = 0.5 * d * log(2π) + 0.5 * logdetΣ
    # Cube truncation correction
    cdf_each = [cdf(Normal(cfg.mu[j], σ), cfg.L) -
                cdf(Normal(cfg.mu[j], σ), -cfg.L) for j in 1:d]
    log_trunc = sum(log.(cdf_each))
    logZ_in_cube = logZ_full + log_trunc
    logZ = logZ_in_cube - d * log(2 * cfg.L)

    return TruthSet(:mvn, d, cfg, samples, μ_truth, Σ_truth, Q, logZ,
                     Dict{Symbol,Any}())
end


"""
    sample_truth_mvn(cfg, N; seed) -> d × N matrix

i.i.d. samples from N(μ, Σ) ∩ cube via Cholesky + (very rare) rejection.
"""
function sample_truth_mvn(cfg::ConfigMVN, N::Integer; seed::Integer = 20260601)
    d = cfg.d
    σ, ρ = cfg.sigma, cfg.rho
    Σ_full = [σ^2 * ρ^abs(i - j) for i in 1:d, j in 1:d]
    L_chol = cholesky(Symmetric(Σ_full)).L
    μ = cfg.mu
    Lcube = cfg.L
    rng = Random.MersenneTwister(seed)
    out = Matrix{Float64}(undef, d, N)
    written = 0
    while written < N
        z = randn(rng, d)
        x = μ .+ L_chol * z
        if all(abs.(x) .< Lcube)
            written += 1
            out[:, written] = x
        end
    end
    return out
end

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# truth_eggbox.jl — Truth for ConfigEggbox (Protocol §4.7.2)
# =============================================================================
#
# By separability, the un-normalised target factors as
#   f(θ) = exp[5 log(2 + ∏_i cos((θ_i + 5π/2)/2))]
# = (2 + ∏_i c_i)^5
# which does *not* split into a product over `i`. The integral is
# reduced by the binomial expansion to six 1-D quadratures (see
# `compute_truth_eggbox` below); reference samples are exact rejection
# draws from the cube prior under the analytical envelope f ≤ 3^5.
#
# Mode-list (V8-FIX-A3, Jul 2026). The previous implementation placed
# the "modes" on the lattice θ_i ∈ {-2π, -π, 0, π, 2π}, where the
# per-coordinate cosine factor is ±cos(π/4) = ±0.7071 — plateau points,
# NOT maxima of the implemented density. The true interior maxima of
#   f(θ) = (2 + ∏_i cos((θ_i + 5π/2)/2))^5
# are the stationary points where every coordinate sits at a cosine
# extremum, θ_i = 2kπ − 5π/2 (per-coordinate factor (−1)^k), AND the
# product of the factors is +1. On [-10, 10] the admissible
# per-coordinate positions are
#   θ ∈ {−5π/2 ≈ −7.854 (+1),  −π/2 ≈ −1.571 (−1),  3π/2 ≈ 4.712 (+1)}
# and a lattice point is a mode iff it uses an EVEN number of the −1
# position. Mode count: (3^d + 1)/2 (5 at d = 2), not 5^d.

import QuadGK


"""
    eggbox_modes(cfg) -> Matrix{Float64}  (d × min(K, 256))

V8-FIX-A3: the true analytical modes of the implemented density.
Per-coordinate cosine-extremum positions inside [-L, L] are
θ = 2kπ − 5π/2 with factor (−1)^k; a lattice point is a mode iff the
product of its factors is +1 (even count of −1 positions). Mode count
is (n₊ + n₋)-lattice restricted to even parity — (3^d + 1)/2 for the
canonical L = 10 domain. For counts above `max_modes` we subsample
random parity-valid representatives.
"""
function eggbox_modes(cfg::ConfigEggbox; max_modes::Integer = 256,
                                       seed::Integer = 19960517)
    d = cfg.d
    L = cfg.L
    # Per-coordinate extremum positions θ = 2kπ − 5π/2 inside [-L, L],
    # with their cosine factor (−1)^k.
    positions = Float64[]
    signs = Int[]
    kmin = Int(cld(-L + _EGGBOX_HALF_PERIOD, 2π))
    kmax = Int(fld(L + _EGGBOX_HALF_PERIOD, 2π))
    for k in kmin:kmax
        θ = 2π * k - _EGGBOX_HALF_PERIOD
        if -L <= θ <= L
            push!(positions, θ)
            push!(signs, iseven(k) ? 1 : -1)
        end
    end
    n_pos = length(positions)
    # Count parity-valid lattice points: (n_pos^d + (n_plus - n_minus)^d)/2
    n_plus = count(==(1), signs)
    n_minus = n_pos - n_plus
    Ktot_big = (big(n_pos)^d + big(n_plus - n_minus)^d) ÷ 2
    if Ktot_big <= max_modes
        Ktot = Int(Ktot_big)
        modes = Matrix{Float64}(undef, d, Ktot)
        idx = 1
        for tup in Iterators.product(ntuple(_ -> 1:n_pos, d)...)
            s = 1
            @inbounds for j in 1:d
                s *= signs[tup[j]]
            end
            s == 1 || continue
            @inbounds for j in 1:d
                modes[j, idx] = positions[tup[j]]
            end
            idx += 1
        end
        @assert idx == Ktot + 1 "eggbox_modes: parity enumeration mismatch"
        return modes
    end
    # Subsample parity-valid lattice points by rejection.
    rng = MersenneTwister(seed)
    modes = Matrix{Float64}(undef, d, max_modes)
    k = 1
    while k <= max_modes
        s = 1
        @inbounds for j in 1:d
            i = rand(rng, 1:n_pos)
            modes[j, k] = positions[i]
            s *= signs[i]
        end
        s == 1 || continue
        k += 1
    end
    return modes
end


"""
    compute_truth_eggbox(cfg; N_ref, seed) -> TruthSet

V4-FIX-5b (May 2026): the previous implementation estimated `logZ` by
plain Monte Carlo on `(2 + ∏ cos)^5`, which carried a Monte Carlo
standard error of order 0.01–0.03 nats on `n_quad = 200_000` and
quickly degraded for `d > 5`.  The fix exploits the binomial
expansion

    f(θ) = (2 + ∏ c_i)^5 = Σ_{k=0..5} C(5,k) · 2^(5-k) · (∏ c_i)^k

so that, by separability of the cube `[-L, L]^d`,

    Z = Σ_{k=0..5} C(5,k) · 2^(5-k) · I_k^d ,
    I_k = ∫_{-L}^{L} cos((t + 5π/2)/2)^k dt .

`I_0 = 2L` exactly; `I_1..I_5` are computed once with `QuadGK` at
`rtol = 1e-13`, giving a deterministic `logZ` accurate to machine
precision in any dimension.
"""
function compute_truth_eggbox(cfg::ConfigEggbox;
                                  N_ref::Integer = 1_000_000,
                                  seed::Integer = 20260501)
    d = cfg.d
    L = cfg.L

    # 1) High-precision 1-D integrals  I_k  for  k = 0, 1, …, 5.
    #    Each is independent of `d`; the sum below couples them.
    I = Vector{Float64}(undef, 6)
    I_err = Vector{Float64}(undef, 6)
    I[1] = 2 * L            # k = 0  (cos^0 = 1)
    I_err[1] = 0.0
    for k in 1:5
        kk = k     # capture for the closure
        integrand_k(t::Real) = cos(0.5 * (t + _EGGBOX_HALF_PERIOD)) ^ kk
        I[k + 1], I_err[k + 1] =
            QuadGK.quadgk(integrand_k, -L, L;
                              rtol = 1e-13, atol = 0.0,
                              order = 21, maxevals = 10^7)
    end

    # 2) Z  =  Σ_{k=0..5} binomial(5, k) · 2^(5-k) · I_k^d.
    Z = 0.0
    @inbounds for k in 0:5
        Z += binomial(5, k) * (2.0 ^ (5 - k)) * (I[k + 1] ^ d)
    end
    log_int = log(Z)
    logZ = log_int - d * log(2 * L)

    # 3) Truth samples via rejection on an envelope (unchanged).
    rng = MersenneTwister(seed)
    f_max = (2.0 + 1.0) ^ 5    # ∏ cos ≤ 1
    samples = _sample_eggbox(cfg, N_ref, f_max, rng)

    μ, Σ = _empirical_moments(samples)
    Q = _empirical_quantiles(samples)
    modes = eggbox_modes(cfg)

    extras = Dict{Symbol,Any}(
        :modes        => modes,
        :integral_est => Z,
        :integral_se  => sqrt(sum(@. (I_err / max(I, eps()))^2)) * Z,
        :I_k          => I,
        :I_k_quad_err => I_err,
        :logZ_method  => "binomial-expansion + 6×quadgk(rtol=1e-13)",
    )

    return TruthSet(:eggbox, d, cfg, samples, μ, Σ, Q, logZ, extras)
end

"""
    sample_truth_eggbox(cfg, N; seed) -> d × N
"""
function sample_truth_eggbox(cfg::ConfigEggbox, N::Integer;
                                seed::Integer = 20260601)
    rng = MersenneTwister(seed)
    f_max = (2.0 + 1.0) ^ 5
    return _sample_eggbox(cfg, N, f_max, rng)
end


function _sample_eggbox(cfg::ConfigEggbox, N::Integer, f_max::Real,
                            rng::AbstractRNG)
    d = cfg.d
    L = cfg.L
    out = Matrix{Float64}(undef, d, N)
    written = 0
    log_fmax = log(f_max)
    while written < N
        prod_c = 1.0
        cand = Vector{Float64}(undef, d)
        @inbounds for j in 1:d
            t = -L + 2 * L * rand(rng)
            cand[j] = t
            prod_c *= cos(0.5 * (t + _EGGBOX_HALF_PERIOD))
        end
        log_f = 5.0 * log(2.0 + prod_c)
        # Accept with probability exp(log_f - log_fmax).
        if log(rand(rng)) <= log_f - log_fmax
            written += 1
            @inbounds for j in 1:d
                out[j, written] = cand[j]
            end
        end
    end
    return out
end

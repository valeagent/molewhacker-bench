# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# metrics.jl — pure (MethodResult, TruthSet) → Float64 functions (Protocol §9)
# =============================================================================

# -----------------------------------------------------------------------------
# §9.1 / §9.2 — fixed evaluation sample size
# -----------------------------------------------------------------------------
#
# The W1 estimator's bias and variance scale as O(N^{-1/2}). To avoid
# favouring algorithms that produce more output samples (Protocol §9.1
# rev April 2026), the evaluation sample count is fixed across every
# algorithm. Every algorithm output is resampled to W1_EVAL_N before the
# L1 sort-pairing.

const W1_EVAL_N = 10_000


"""
    _resample_to_eval_N(mr, N, rng) -> Matrix

Resample the (possibly weighted) output of `mr` to exactly `N`
equal-weight samples. Empty / zero-weight inputs return a `d × 1`
matrix of zeros so downstream metrics fail gracefully (NaN).
"""
function _resample_to_eval_N(mr::MethodResult, N::Integer, rng::AbstractRNG)
    d, M = size(mr.samples)
    if M == 0
        return zeros(d, 1)
    end
    if M == 1
        return repeat(mr.samples, 1, N)
    end
    if isempty(mr.weights) || all(mr.weights .== mr.weights[1])
        idx = rand(rng, 1:M, Int(N))
        return mr.samples[:, idx]
    end
    wsum = sum(mr.weights)
    if !isfinite(wsum) || wsum <= 0
        idx = rand(rng, 1:M, Int(N))
        return mr.samples[:, idx]
    end
    wn = mr.weights ./ wsum
    idx = StatsBase.wsample(rng, 1:M, wn, Int(N); replace = true)
    return mr.samples[:, idx]
end

"""
    _truth_subsample(truth, N, rng) -> Matrix

Resample the truth-set to `N` samples (with replacement if `N > N_ref`).
"""
function _truth_subsample(truth::TruthSet, N::Integer, rng::AbstractRNG)
    Nref = size(truth.samples, 2)
    if N <= Nref
        perm = Random.randperm(rng, Nref)[1:N]
        return truth.samples[:, perm]
    else
        return truth.samples[:, rand(rng, 1:Nref, N)]
    end
end


# -----------------------------------------------------------------------------
# §9.1 — Marginal-averaged Wasserstein-1
# -----------------------------------------------------------------------------

"""
    w1_marginal_avg(mr::MethodResult, truth::TruthSet) -> Float64

Average over `j ∈ 1..d` of the 1-D Wasserstein-1 distance between the
`j`-th marginal of `mr.samples` (resampled to `W1_EVAL_N` equal-weight
samples) and the `j`-th marginal of `truth.samples` (subsampled to the
same size). For two equal-length sample vectors with the same N the
optimal coupling in 1-D is the sort-pairing, so

    W₁(F̂_j, F_j_ref) = (1/N) Σ_i |sort(s_j)[i] - sort(t_j)[i]|.
"""
function w1_marginal_avg(mr::MethodResult, truth::TruthSet)::Float64
    d = size(mr.samples, 1)
    N = W1_EVAL_N
    rng = MersenneTwister(20260601 + mr.seed)
    samples_eq = _resample_to_eval_N(mr, N, rng)
    truth_sub = _truth_subsample(truth, N, rng)
    M_use = min(size(samples_eq, 2), size(truth_sub, 2))
    M_use == 0 && return NaN
    total = 0.0
    @inbounds for j in 1:d
        s = sort(view(samples_eq, j, 1:M_use))
        t = sort(view(truth_sub, j, 1:M_use))
        local_sum = 0.0
        for k in 1:M_use
            local_sum += abs(s[k] - t[k])
        end
        total += local_sum / M_use
    end
    return total / d
end


# -----------------------------------------------------------------------------
# §9.2 — Sliced Wasserstein
# -----------------------------------------------------------------------------

"""
    sliced_wasserstein(mr, truth; K=100, seed=12345) -> Float64

Mean over K random unit-direction projections of the 1-D W1 distance.
Same fixed-N convention as §9.1.
"""
function sliced_wasserstein(mr::MethodResult, truth::TruthSet;
                              K::Integer = 100, seed::Integer = 12345)::Float64
    d = size(mr.samples, 1)
    N = W1_EVAL_N
    # V8-FIX-B4: the caller already passes `12345 + mr.seed`; the old
    # `Xoshiro(seed + mr.seed)` added the cell seed a second time.
    # Harmless (still deterministic) but unintended — single-add now.
    rng = Xoshiro(seed)
    samples_eq = _resample_to_eval_N(mr, N, rng)
    truth_sub = _truth_subsample(truth, N, rng)
    M_use = min(size(samples_eq, 2), size(truth_sub, 2))
    M_use == 0 && return NaN
    total = 0.0
    u = Vector{Float64}(undef, d)
    @inbounds for k in 1:K
        randn!(rng, u)
        u ./= norm(u)
        ps = sort((u' * samples_eq)[1, 1:M_use])
        pt = sort((u' * truth_sub)[1, 1:M_use])
        s = 0.0
        for i in 1:M_use
            s += abs(ps[i] - pt[i])
        end
        total += s / M_use
    end
    return total / K
end


# -----------------------------------------------------------------------------
# §9.3 — Effective sample size
# -----------------------------------------------------------------------------

"""
    neff_kish(mr) -> Float64

Kish ESS for weighted output:  (Σ w)² / Σ w².
"""
function neff_kish(mr::MethodResult)::Float64
    s = sum(mr.weights)
    s2 = sum(abs2, mr.weights)
    s2 <= 0 && return 0.0
    return s^2 / s2
end

"""
    _expand_by_weights(x, w) -> Vector{Float64}

V8-FIX-B2 helper. Chain samplers store rejected transitions as integer
repeat weights on the retained draws (`nonzero_weights = true`), so the
stored sequence is DEDUPLICATED — its apparent autocorrelation is
shorter than the true chain's, which biases ACF-based ESS upward.
Expanding each draw by its repeat weight reconstructs the actual chain
sequence. Unit (or absent) weights return the input unchanged.
"""
function _expand_by_weights(x::AbstractVector{<:Real},
                             w::AbstractVector{<:Real})
    if isempty(w) || all(==(1.0), w)
        return collect(Float64, x)
    end
    out = Float64[]
    sizehint!(out, max(round(Int, sum(w)), length(x)))
    @inbounds for i in eachindex(x)
        r = max(round(Int, Float64(w[i])), 0)
        for _ in 1:r
            push!(out, Float64(x[i]))
        end
    end
    return isempty(out) ? collect(Float64, x) : out
end

"""
    neff_chain(mr) -> Float64

Autocorrelation-based ESS for chain-style samples. Implements the
Geyer-style initial monotone sequence estimator on each coordinate.

V8-FIX-B2: (i) repeat weights are expanded first, so the ACF sees the
true chain sequence rather than the deduplicated one (the old code
ignored `mr.weights`, inflating NUTS/MH ESS and hence η); (ii) the
returned value is the MINIMUM per-coordinate ESS (the conservative
standard), not the mean.
"""
function neff_chain(samples::AbstractMatrix{<:Real},
                      weights::AbstractVector{<:Real})::Float64
    d, N = size(samples)
    N <= 10 && return Float64(N)
    w = length(weights) == N ? weights : ones(N)
    ess_per_coord = zeros(d)
    @inbounds for j in 1:d
        x = _expand_by_weights(view(samples, j, :), w)
        ess_per_coord[j] = _ess_geyer(x)
    end
    return minimum(ess_per_coord)
end

neff_chain(mr::MethodResult)::Float64 = neff_chain(mr.samples, mr.weights)

# ACF at a single lag (two-pass, mean/variance precomputed).
@inline function _acf_at(x::AbstractVector{<:Real}, μ::Float64,
                          var_x::Float64, k::Int)
    n = length(x)
    s = 0.0
    @inbounds for i in 1:(n - k)
        s += (x[i] - μ) * (x[i + k] - μ)
    end
    return s / ((n - k) * var_x)
end

function _ess_geyer(x::AbstractVector{<:Real})
    n = length(x)
    n <= 10 && return Float64(n)
    μ = mean(x)
    var_x = var(x; corrected = false)
    var_x <= 0 && return Float64(n)
    # V8-FIX-B4: lags are evaluated LAZILY until the initial monotone
    # sequence terminates, with the cap raised 1000 → 5000 (the fixed
    # 1000-lag cap could truncate τ on long slow-mixing chains, biasing
    # ESS upward). Well-mixed chains stop after a handful of lags, so
    # this is also faster than the old eager 1000-lag loop.
    Mmax = min(div(n, 2) - 1, 5_000)
    sum_rho = 0.0
    prev_pair = Inf
    k = 1
    while k + 1 <= Mmax
        pair = _acf_at(x, μ, var_x, k) + _acf_at(x, μ, var_x, k + 1)
        pair = min(pair, prev_pair)
        pair < 0 && break
        sum_rho += pair
        prev_pair = pair
        k += 2
    end
    τ = 1 + 2 * sum_rho
    return n / max(τ, 1.0)
end

"""
    neff(mr) -> Float64

Dispatch to Kish for weighted methods, autocorrelation for chains.
"""
function neff(mr::MethodResult)::Float64
    if mr.algorithm in (:is, :ns, :mw)
        return neff_kish(mr)
    else
        return neff_chain(mr)
    end
end


"""
    convergence_curve(mr; npoints=12) -> (n, cost, neff) NamedTuple

Running-convergence reconstruction: the triple
(samples produced, likelihood-equivalents consumed, effective sample
size accumulated) at a sequence of points WITHIN a single stored run,
so that η = N_eff/N_L can be plotted against either the sample count
or the consumed cost. "Samples produced" is the per-algorithm natural
count: i.i.d. draws (IS), true chain steps (MH), transitions summed
over chains (NUTS), dead points (NS), accumulated population size
(MoleWhacker). Reconstruction
fidelity differs by algorithm and is documented per branch:

* `:mw`   — exact: the recorded iteration log stores (cum_cost, ess).
* `:is`   — exact: draws are i.i.d. in storage order; the prefix Kish
            ESS at n draws costs exactly n evaluations.
* `:mh`   — exact: stored draws are deduplicated with integer repeat
            weights, so the cumulative repeat weight IS the true chain
            position; the tuning/pilot overhead (`Nlike_used` minus
            total production steps) is added as a constant offset.
* `:nuts` — chunk-resolved: the production controller's extension_log
            records the exact realized cost after every measured chunk;
            ESS is evaluated at those chunk boundaries (the per-chain
            prefix is located through the cumulative repeat weights).
            The ~8-transition pilot burst on the first chain is not
            resolved (≤ 8 transitions of blur).
* `:ns`   — cost axis NOT reconstructible: dead points are stored in
            order (the prefix Kish ESS is exact as a function of the
            iteration index), but the per-iteration rejection cost is
            not recorded, so the cost is smeared uniformly across the
            run (total cost × n/N). The smear distorts the horizontal
            placement — nested sampling's rejection cost grows as the
            likelihood contour tightens — and any figure using this
            branch must label the NS curve as approximate.
"""
function convergence_curve(mr::MethodResult; npoints::Int = 12)
    alg = mr.algorithm
    nsamp = Float64[]     # samples produced so far (per-algorithm meaning
                          # documented above: draws / chain steps /
                          # transitions / dead points / population size)
    cost = Float64[]
    nf = Float64[]
    N = size(mr.samples, 2)

    if alg === :mw
        il = get(mr.extras, :iter_log, NamedTuple[])
        for e in il
            c = haskey(e, :cum_cost) ? Float64(e.cum_cost) : NaN
            (isfinite(c) && c > 0) || continue
            ess = Float64(e.ess)
            eff = Float64(e.eff)
            push!(nsamp, eff > 0 ? ess / eff : NaN)   # |S_t| = ess/eff
            push!(cost, c)
            push!(nf, ess)
        end

    elseif alg === :is
        N < 20 && return (n = nsamp, cost = cost, neff = nf)
        ns = unique(round.(Int,
            exp10.(range(log10(20.0), log10(Float64(N)); length = npoints))))
        for n in ns
            w = view(mr.weights, 1:n)
            s = sum(w); s2 = sum(abs2, w)
            s2 > 0 || continue
            push!(nsamp, Float64(n))
            push!(cost, Float64(n))
            push!(nf, s^2 / s2)
        end

    elseif alg === :mh
        nchains = max(Int(get(mr.extras, :nchains, 1)), 1)
        npc = div(N, nchains)
        npc > 10 || return (n = nsamp, cost = cost, neff = nf)
        w = length(mr.weights) == N ? Float64.(mr.weights) : ones(N)
        cums = [cumsum(view(w, ((c - 1) * npc + 1):(c * npc))) for c in 1:nchains]
        T_end = minimum(last.(cums))
        overhead = max(mr.Nlike_used - sum(last.(cums)), 0.0)
        T_end > 50 || return (n = nsamp, cost = cost, neff = nf)
        Ts = unique(round.(Int,
            exp10.(range(log10(50.0), log10(T_end); length = npoints))))
        for T in Ts
            idx = [min(searchsortedfirst(cums[c], Float64(T)), npc)
                   for c in 1:nchains]
            cols = Int[]
            steps = 0.0
            for c in 1:nchains
                append!(cols, ((c - 1) * npc + 1):((c - 1) * npc + idx[c]))
                steps += min(cums[c][idx[c]], Float64(T))
            end
            push!(nsamp, steps)
            push!(cost, overhead + steps)
            push!(nf, neff_chain(mr.samples[:, cols], w[cols]))
        end

    elseif alg === :nuts
        elog = get(mr.extras, :extension_log, NamedTuple[])
        nchains = max(Int(get(mr.extras, :nchains_eff,
                              get(mr.extras, :nchains, 1))), 1)
        npc = div(N, nchains)
        (npc > 5 && !isempty(elog)) || return (n = nsamp, cost = cost, neff = nf)
        w = length(mr.weights) == N ? Float64.(mr.weights) : ones(N)
        cums = [cumsum(view(w, ((c - 1) * npc + 1):(c * npc))) for c in 1:nchains]
        T = 0.0
        for e in elog
            haskey(e, :phase) || continue
            e.phase in (:probe_chunk, :fill) || continue
            (haskey(e, :m) && haskey(e, :cost)) || continue
            T += Float64(e.m)
            idx = [min(searchsortedfirst(cums[c], T), npc) for c in 1:nchains]
            all(i -> i >= 3, idx) || continue
            cols = Int[]
            steps = 0.0
            for c in 1:nchains
                append!(cols, ((c - 1) * npc + 1):((c - 1) * npc + idx[c]))
                steps += min(cums[c][idx[c]], T)
            end
            push!(nsamp, steps)
            push!(cost, Float64(e.cost))
            push!(nf, neff_chain(mr.samples[:, cols], w[cols]))
        end

    elseif alg === :ns
        N < 20 && return (n = nsamp, cost = cost, neff = nf)
        total = mr.Nlike_used
        ns = unique(round.(Int,
            exp10.(range(log10(20.0), log10(Float64(N)); length = npoints))))
        for n in ns
            wv = view(mr.weights, 1:n)
            s = sum(wv); s2 = sum(abs2, wv)
            s2 > 0 || continue
            push!(nsamp, Float64(n))
            push!(cost, total * n / N)     # uniform smear — approximate
            push!(nf, s^2 / s2)
        end
    end
    return (n = nsamp, cost = cost, neff = nf)
end


# W1 on an explicit (samples, weights) pair — the prefix-capable core of
# w1_marginal_avg (same estimator: equal-weight multinomial resample to
# N_eval, truth subsample to the same size, per-coordinate sort pairing).
function _w1_matrix(samples::AbstractMatrix{<:Real},
                     weights::AbstractVector{<:Real},
                     truth::TruthSet, N_eval::Integer,
                     rng::AbstractRNG)::Float64
    d, M = size(samples)
    M < 2 && return NaN
    idx = if isempty(weights) || all(w -> w == weights[1], weights)
        rand(rng, 1:M, Int(N_eval))
    else
        ws = sum(weights)
        (isfinite(ws) && ws > 0) || return NaN
        StatsBase.wsample(rng, 1:M, weights ./ ws, Int(N_eval); replace = true)
    end
    eq = samples[:, idx]
    tr = _truth_subsample(truth, Int(N_eval), rng)
    Mu = min(size(eq, 2), size(tr, 2))
    Mu == 0 && return NaN
    total = 0.0
    @inbounds for j in 1:d
        s = sort(view(eq, j, 1:Mu))
        t = sort(view(tr, j, 1:Mu))
        acc = 0.0
        for i in 1:Mu
            acc += abs(s[i] - t[i])
        end
        total += acc / Mu
    end
    return total / d
end

"""
    accuracy_curve(mr, truth; npoints=8, n_eval=10_000) -> (n, cost, w1)

Accuracy-over-time reconstruction: the marginal Wasserstein-1
distance to the truth of the output the run HAD ACCUMULATED at a
sequence of points within a single stored run. The prefix semantics
and cost mapping are identical to `convergence_curve` per algorithm
(exact for `:is`/`:mh`, chunk-resolved for `:nuts`, uniform cost smear
for `:ns`).

`:mw` deliberately returns an empty curve: the stored weights are the
FINAL global weights against the last mixture, so a stored-run prefix
does not reproduce what the algorithm would have reported at an
earlier iteration (every iteration reweights the whole population).
The exact MoleWhacker accuracy history requires the per-iteration
snapshot instrumentation (`run_mw(...; cache_dir, 
save_iteration_samples = true)`), from which the pooled population and
its iteration-t weights are reconstructed exactly.
"""
function accuracy_curve(mr::MethodResult, truth::TruthSet;
                          npoints::Int = 8, n_eval::Int = 10_000)
    alg = mr.algorithm
    nsamp = Float64[]
    cost = Float64[]
    w1 = Float64[]
    N = size(mr.samples, 2)
    rng = MersenneTwister(20260830 + mr.seed)

    if alg === :is || alg === :ns
        N < 50 && return (n = nsamp, cost = cost, w1 = w1)
        total = mr.Nlike_used
        ns = unique(round.(Int,
            exp10.(range(log10(50.0), log10(Float64(N)); length = npoints))))
        for n in ns
            v = _w1_matrix(view(mr.samples, :, 1:n),
                           view(mr.weights, 1:n), truth, n_eval, rng)
            isfinite(v) || continue
            push!(nsamp, Float64(n))
            push!(cost, alg === :is ? Float64(n) : total * n / N)
            push!(w1, v)
        end

    elseif alg === :mh || alg === :nuts
        nchains = max(Int(get(mr.extras, :nchains_eff,
                              get(mr.extras, :nchains, 1))), 1)
        npc = div(N, nchains)
        npc > 10 || return (n = nsamp, cost = cost, w1 = w1)
        w = length(mr.weights) == N ? Float64.(mr.weights) : ones(N)
        cums = [cumsum(view(w, ((c - 1) * npc + 1):(c * npc))) for c in 1:nchains]

        # Prefix boundaries: log grid of true-chain positions for MH,
        # the controller's chunk boundaries for NUTS (with the logged
        # exact cost).
        bounds = Tuple{Float64,Float64}[]   # (T, cost)
        if alg === :mh
            T_end = minimum(last.(cums))
            overhead = max(mr.Nlike_used - sum(last.(cums)), 0.0)
            T_end > 50 || return (n = nsamp, cost = cost, w1 = w1)
            for T in unique(round.(Int,
                    exp10.(range(log10(50.0), log10(T_end); length = npoints))))
                push!(bounds, (Float64(T), overhead + nchains * Float64(T)))
            end
        else
            elog = get(mr.extras, :extension_log, NamedTuple[])
            T = 0.0
            for e in elog
                haskey(e, :phase) || continue
                e.phase in (:probe_chunk, :fill) || continue
                (haskey(e, :m) && haskey(e, :cost)) || continue
                T += Float64(e.m)
                push!(bounds, (T, Float64(e.cost)))
            end
        end
        for (T, c) in bounds
            idx = [min(searchsortedfirst(cums[ch], T), npc) for ch in 1:nchains]
            all(i -> i >= 3, idx) || continue
            cols = Int[]
            for ch in 1:nchains
                append!(cols, ((ch - 1) * npc + 1):((ch - 1) * npc + idx[ch]))
            end
            v = _w1_matrix(mr.samples[:, cols], w[cols], truth, n_eval, rng)
            isfinite(v) || continue
            push!(nsamp, T * nchains)
            push!(cost, c)
            push!(w1, v)
        end
    end
    return (n = nsamp, cost = cost, w1 = w1)
end


# -----------------------------------------------------------------------------
# §9.3a — Gelman–Rubin R-hat for chain samplers
# -----------------------------------------------------------------------------

"""
    rhat_per_coord(mr; nchains=nothing) -> Vector{Float64}

Split-R̂ (Gelman et al. 2013) per coordinate, computed on the
chain-major output of a chain sampler (`:mh`, `:nuts`).

V8-FIX-B2: (i) each stored chain is first expanded by its repeat
weights, so R̂ is computed on the actual chain sequence (duplicates
included) rather than the deduplicated one; (ii) each chain is then
split in half — split-R̂ — so mid-chain trends that plain
Gelman–Rubin misses are detected.

The number of chains is read from `mr.extras[:nchains]` if available
(this is the chain-major contract — see §9.3a of the protocol). If
`nchains` is provided as a kwarg, it overrides; otherwise the default
of 4 is used.
"""
function rhat_per_coord(mr::MethodResult; nchains::Union{Nothing,Integer} = nothing)
    nc = nchains === nothing ? Int(get(mr.extras, :nchains, 4)) : Int(nchains)
    d, N = size(mr.samples)
    if nc < 1 || N < 4 * max(nc, 1)
        return fill(NaN, d)
    end
    n = div(N, nc)
    w = length(mr.weights) == N ? mr.weights : ones(N)
    rhat = Vector{Float64}(undef, d)
    @inbounds for j in 1:d
        # Expand each stored chain by its repeat weights, then truncate
        # all chains to the common minimum length.
        chains = Vector{Vector{Float64}}(undef, nc)
        for c in 1:nc
            lo = (c - 1) * n + 1
            hi = c * n
            chains[c] = _expand_by_weights(view(mr.samples, j, lo:hi),
                                           view(w, lo:hi))
        end
        n_min = minimum(length.(chains))
        if n_min < 4
            rhat[j] = NaN
            continue
        end
        # Split each chain in half → 2·nc half-chains of length n_half.
        n_half = div(n_min, 2)
        m = 2 * nc
        chain_mean = Vector{Float64}(undef, m)
        chain_var = Vector{Float64}(undef, m)
        for c in 1:nc
            x = chains[c]
            h1 = view(x, 1:n_half)
            h2 = view(x, (n_min - n_half + 1):n_min)
            chain_mean[2c - 1] = mean(h1)
            chain_mean[2c] = mean(h2)
            chain_var[2c - 1] = var(h1; corrected = true)
            chain_var[2c] = var(h2; corrected = true)
        end
        W = mean(chain_var)
        Bvar = n_half * var(chain_mean; corrected = true)
        if W <= 0
            rhat[j] = NaN
            continue
        end
        var_plus = ((n_half - 1) / n_half) * W + Bvar / n_half
        rhat[j] = sqrt(var_plus / W)
    end
    return rhat
end

"""
    rhat_max(mr) -> Float64

Maximum R̂ across coordinates; `NaN` for non-chain samplers.
"""
function rhat_max(mr::MethodResult)::Float64
    if !(mr.algorithm in (:mh, :nuts))
        return NaN
    end
    r = rhat_per_coord(mr)
    finite_r = filter(isfinite, r)
    return isempty(finite_r) ? NaN : maximum(finite_r)
end


# -----------------------------------------------------------------------------
# §9.4 — Log-evidence error
# -----------------------------------------------------------------------------

"""
    dlogZ(mr, truth) -> Union{Float64,Missing}

|logZ_estimate − logZ_truth| for `is`, `ns`, `mw`. Missing for `mh`,
`nuts` because they do not produce evidence estimates.
"""
function dlogZ(mr::MethodResult, truth::TruthSet)
    if mr.algorithm in (:mh, :nuts)
        return missing
    end
    if mr.logZ_estimate === missing || !isfinite(mr.logZ_estimate)
        return missing
    end
    if !isfinite(truth.logZ)
        return missing
    end
    return abs(mr.logZ_estimate - truth.logZ)
end


# -----------------------------------------------------------------------------
# §9.5 — Quantile errors (marginal coverage)
# -----------------------------------------------------------------------------

"""
    quantile_errors(mr, truth) -> Dict{Symbol, Float64}

Returns absolute quantile errors at five canonical levels, averaged
across coordinates. Keys: :p025, :p160, :p500, :p840, :p975. Uses the
fixed-N resampling convention of §9.1.
"""
function quantile_errors(mr::MethodResult, truth::TruthSet)
    levels = (:p025, :p160, :p500, :p840, :p975)
    α = (0.025, 0.16, 0.5, 0.84, 0.975)
    qe = Dict{Symbol,Float64}()
    d = size(mr.samples, 1)
    rng = MersenneTwister(20260801 + mr.seed)
    samples_eq = _resample_to_eval_N(mr, W1_EVAL_N, rng)
    # After resampling to W1_EVAL_N the samples are equal-weight; pass
    # explicit unit weights of matching length so the inverse-CDF
    # weighted-quantile function does not assert on a length mismatch
    # (the bug surfaced in smoke_budget.jl, May 2026).
    N_eq = size(samples_eq, 2)
    eq_w = ones(Float64, N_eq)
    for (sym, p) in zip(levels, α)
        total = 0.0
        for j in 1:d
            x = view(samples_eq, j, :)
            q_emp = weighted_quantile(collect(x), eq_w, p)
            q_ref = truth.quantiles[j, quantile_idx_of(p)]
            total += abs(q_emp - q_ref)
        end
        qe[sym] = total / d
    end
    return qe
end


# -----------------------------------------------------------------------------
# §9.6 — KL divergence on the cube (MoleWhacker only)
# -----------------------------------------------------------------------------

"""
    kl_cube_mw(mr, truth, cfg; N_eval=10_000, seed=54321) -> Float64

Forward KL D_KL(q_T || p_truth) on the cube, computed by Monte Carlo
from `N_eval` samples drawn from the final mixture proposal `q_T`.

The stored mixture lives in the PriorToNormal-TRANSFORMED space
(z-coordinates): MoleWhacker runs its entire adaptation there. For
the uniform box prior the componentwise transform and its Jacobian
are closed form,

    θ_i = 2L·Φ(z_i) − L,        dθ_i/dz_i = 2L·φ(z_i),

so the user-space proposal density at a mapped draw is
log q_user(θ) = log q_z(z) − Σ_i [log(2L) + log φ(z_i)].

NOTE: an earlier implementation evaluated the user-space target
`log_f` directly at the z-space draws — no transform, no Jacobian —
which made the recorded KL_cube values meaningless (it compared
densities of different variables at inconsistent points). Because the
transform maps every draw strictly into the open cube, the cube
restriction constant of the proposal is exactly 1 and drops out.
"""
function kl_cube_mw(mr::MethodResult, truth::TruthSet, cfg::ProblemConfig;
                      N_eval::Integer = 10_000, seed::Integer = 54321)::Float64
    if mr.algorithm != :mw
        return NaN
    end
    mix = get(mr.extras, :mixture, nothing)
    if mix === nothing
        return NaN
    end
    rng = Xoshiro(seed + mr.seed)
    d = cfg.d
    Lcube = cfg.L

    z = rand(rng, mix, N_eval)          # z-space draws from the mixture
    log_norm_p = truth.logZ + d * log(2 * Lcube)
    log_f = build_log_f(cfg)
    stdn = Normal()

    # V8-FIX-B4 (retained): non-finite integrand terms are DROPPED and
    # counted; if more than 1 % of the terms are non-finite the
    # estimate is unreliable and NaN is returned instead.
    kl_terms = Float64[]
    sizehint!(kl_terms, N_eval)
    n_bad = 0
    θ = Vector{Float64}(undef, d)
    for i in 1:N_eval
        log_jac = 0.0
        for j in 1:d
            zj = z[j, i]
            θ[j] = 2 * Lcube * cdf(stdn, zj) - Lcube
            log_jac += log(2 * Lcube) + logpdf(stdn, zj)
        end
        log_q_z = try
            logpdf(mix, collect(view(z, :, i)))
        catch
            NaN
        end
        log_q_user = log_q_z - log_jac
        log_p_user = log_f(θ) - log_norm_p
        if !isfinite(log_q_user) || !isfinite(log_p_user)
            n_bad += 1
        else
            push!(kl_terms, log_q_user - log_p_user)
        end
    end
    (isempty(kl_terms) || n_bad > 0.01 * N_eval) && return NaN
    return mean(kl_terms)
end


# -----------------------------------------------------------------------------
# §9.7 — Mode recovery rate (V3-M-1)
# -----------------------------------------------------------------------------
#
# Mode recovery rate is the fraction of true modes for which at least
# one cluster centroid in the algorithm output lies within ε of that
# mode. Clustering uses a single-pass radius-based algorithm with
# bandwidth `eps_radius / 2` (the protocol's DBSCAN spec maps to this
# threshold for the multimodal targets of interest because all spikes
# are well separated relative to ε). Per Protocol §9.7 the metric
# returns `NaN` when the truth has no `modes` field — `cells.csv` then
# carries `NaN` and the headline figures skip the metric for unimodal
# targets.

"""
    _radius_clusters(samples, eps; min_neighbors=5) -> Vector{Vector{Int}}

Lightweight density-based clustering that returns the indices of each
cluster. Two samples belong to the same cluster iff their L2 distance
is below `eps`; a cluster is kept only if it has ≥ `min_neighbors`
points. The implementation is `O(N · K)` where `K` is the cluster
count, which is fine for `N = 10 000` × a few hundred modes.

This emulates DBSCAN with `min_pts = min_neighbors` for our
purposes (well-separated multimodal targets).
"""
function _radius_clusters(samples::AbstractMatrix{<:Real}, eps_radius::Real;
                                min_neighbors::Integer = 5)
    d, N = size(samples)
    eps2 = eps_radius * eps_radius
    centroids = Vector{Vector{Float64}}()
    counts = Int[]
    members = Vector{Vector{Int}}()
    @inbounds for i in 1:N
        x = view(samples, :, i)
        # Find the nearest existing centroid.
        best = 0
        best_d = Inf
        for k in 1:length(centroids)
            c = centroids[k]
            δ2 = 0.0
            for j in 1:d
                tmp = x[j] - c[j]
                δ2 += tmp * tmp
                if δ2 > best_d
                    break
                end
            end
            if δ2 < best_d
                best_d = δ2
                best = k
            end
        end
        if best > 0 && best_d < eps2
            push!(members[best], i)
            counts[best] += 1
            # Online centroid update (running mean).
            cb = centroids[best]
            inv_n = 1.0 / counts[best]
            for j in 1:d
                cb[j] += inv_n * (x[j] - cb[j])
            end
        else
            push!(centroids, collect(x))
            push!(counts, 1)
            push!(members, [i])
        end
    end
    # Drop singletons / clusters smaller than `min_neighbors`.
    keep = findall(c -> c >= min_neighbors, counts)
    return [members[k] for k in keep], [centroids[k] for k in keep]
end

"""
    _compute_truth_modes(truth) -> Union{Matrix{Float64},Nothing}

Returns the analytical mode list for known multimodal targets,
falling back to `truth.extras[:modes]` if present. The fallback is
needed because the v2 truth files were generated before the
`:modes` field existed, and we want `mode_recovery` to work without
re-running 00_generate_truth.jl on every cell.
"""
function _compute_truth_modes(truth::TruthSet)
    cached = get(truth.extras, :modes, nothing)
    cached !== nothing && return cached
    cfg = truth.cfg
    if cfg isa ConfigMRidgesSpiky
        return mridges_spiky_modes(cfg)
    elseif cfg isa ConfigEggbox
        return eggbox_modes(cfg)
    elseif cfg isa ConfigMRidges
        return _mridges_modes_from_cfg(cfg)
    end
    return nothing
end

"""
    _mridges_modes_from_cfg(cfg) -> Matrix{Float64}  (d × 2^d)

The `2^d` modes of the v2 `mridges` target. By construction the
modes lie at `θ_1 ∈ {±Δ}` and the conditional spike means
`±m(θ_1) = ±(a + b θ_1²)` — so for each combination of signs in
`d` coordinates we have one mode.
"""
function _mridges_modes_from_cfg(cfg::ConfigMRidges)
    d = cfg.d
    Δ = cfg.Δ
    a = cfg.a
    bcurve = cfg.b
    K = 2 ^ d
    modes = Matrix{Float64}(undef, d, K)
    for k in 0:(K - 1)
        sign_θ1 = (k & 0x1) == 1 ? -1.0 : 1.0
        θ1 = sign_θ1 * Δ
        m = a + bcurve * θ1 * θ1
        @inbounds modes[1, k + 1] = θ1
        @inbounds for j in 2:d
            sj = (k >> (j - 1)) & 0x1 == 1 ? -1.0 : 1.0
            modes[j, k + 1] = sj * m
        end
    end
    return modes
end

"""
    mode_recovery(mr, truth, eps_radius; min_neighbors=5) -> Float64

Fraction of true modes (from `_compute_truth_modes(truth)`) that
have at least one output-cluster centroid within `eps_radius`.
Returns `NaN` for unimodal targets.
"""
function mode_recovery(mr::MethodResult, truth::TruthSet,
                          eps_radius::Real;
                          min_neighbors::Integer = 5,
                          rng_seed::Integer = 20260605)
    truth_modes = _compute_truth_modes(truth)
    truth_modes === nothing && return NaN
    rng = MersenneTwister(rng_seed + mr.seed)
    samples = _resample_to_eval_N(mr, W1_EVAL_N, rng)
    _, centroids = _radius_clusters(samples, eps_radius / 2;
                                       min_neighbors = min_neighbors)
    isempty(centroids) && return 0.0
    K = size(truth_modes, 2)
    K == 0 && return NaN
    hits = 0
    @inbounds for k in 1:K
        target = view(truth_modes, :, k)
        # Search nearest centroid.
        best = Inf
        for c in centroids
            δ2 = 0.0
            for j in 1:length(c)
                tmp = c[j] - target[j]
                δ2 += tmp * tmp
                if δ2 > best
                    break
                end
            end
            if δ2 < best
                best = δ2
            end
        end
        if sqrt(best) < eps_radius
            hits += 1
        end
    end
    return hits / K
end

"""
    _eps_radius_for_problem(truth) -> Float64

Default mode-recovery `ε` per Protocol §9.7 (rev May 2026, V4-FIX-1).

Rationale (V4-FIX-1):
  * `mridges_spiky`: a sample-cloud cluster centroid for a Gaussian
    spike of width σ sits at offset O(σ) from the true mode for
    finite N. Tolerance 4σ tests *coverage*, not centroid bias. The
    previous 2σ rule was geometrically too tight and read 0.0 on every
    cell.
  * `eggbox` (V8-FIX-A3): the mode list now holds the true density
    maxima (per-coordinate positions 2kπ − 5π/2, even count of −1
    factors), whose minimum inter-mode distance is 2π√2 ≈ 8.89 and
    whose local Gaussian width is σ ≈ 1.55 per coordinate. ε = 1.0
    (≈ 0.65 σ, ≪ half the inter-mode distance) tests centroid
    proximity without cross-mode leakage.
  * `mridges`: unchanged at 2σ_⊥ (= 2 × s0).

Returns NaN for unimodal targets so the metric is silently skipped.
"""
function _eps_radius_for_problem(truth::TruthSet)
    p = truth.problem
    if p == :mridges
        # ε = 2 σ_⊥ ≈ 2 × 0.5 = 1.0 (unchanged).
        return truth.cfg isa ConfigMRidges ? 2.0 * truth.cfg.s0 : 1.0
    elseif p == :mridges_spiky
        # V4-FIX-1: 4 σ_spike (was 2 σ_spike).
        return truth.cfg isa ConfigMRidgesSpiky ? 4.0 * truth.cfg.σ_spike : 0.4
    elseif p == :eggbox
        # V8-FIX-A3: true modes at separation ≥ 2π√2, local width σ ≈ 1.55.
        return 1.0
    end
    return NaN
end


# -----------------------------------------------------------------------------
# §9.8 — Maximum mean discrepancy (V3) — RBF kernel + median heuristic
# -----------------------------------------------------------------------------
#
# Computed post-hoc from the saved `MethodResult.samples`. The
# bandwidth uses Gretton et al. (2012, eq. 7) median-pairwise-distance
# heuristic and the standard *unbiased* MMD² estimator with
# off-diagonal corrections.

@inline function _pairwise_sqeuclidean!(out::AbstractMatrix{<:Real},
                                              X::AbstractMatrix{<:Real},
                                              Y::AbstractMatrix{<:Real})
    d, NX = size(X)
    _, NY = size(Y)
    @inbounds for i in 1:NX, j in 1:NY
        s = 0.0
        for k in 1:d
            δ = X[k, i] - Y[k, j]
            s += δ * δ
        end
        out[i, j] = s
    end
    return out
end

@inline function _self_pairwise_sqeuclidean!(out::AbstractMatrix{<:Real},
                                                 X::AbstractMatrix{<:Real})
    d, N = size(X)
    @inbounds for i in 1:N
        out[i, i] = 0.0
        for j in (i + 1):N
            s = 0.0
            for k in 1:d
                δ = X[k, i] - X[k, j]
                s += δ * δ
            end
            out[i, j] = s
            out[j, i] = s
        end
    end
    return out
end

"""
    median_pairwise_distance(samples) -> Float64

Median pairwise Euclidean distance over a sample matrix `d × N`. The
median-of-N(N−1)/2 pairs implementation; `N` is capped at 4 000 in the
caller for memory.
"""
function median_pairwise_distance(samples::AbstractMatrix{<:Real};
                                       max_pairs::Integer = 200_000,
                                       rng::AbstractRNG = MersenneTwister(20260606))
    d, N = size(samples)
    M = N * (N - 1) ÷ 2
    if M == 0
        return 0.0
    end
    if M <= max_pairs
        out = Vector{Float64}(undef, M)
        idx = 0
        @inbounds for i in 1:(N - 1)
            for j in (i + 1):N
                s = 0.0
                for k in 1:d
                    δ = samples[k, i] - samples[k, j]
                    s += δ * δ
                end
                idx += 1
                out[idx] = sqrt(s)
            end
        end
        return median(out)
    else
        # Random subsample of pairs.
        out = Vector{Float64}(undef, max_pairs)
        @inbounds for p in 1:max_pairs
            i = rand(rng, 1:N)
            j = rand(rng, 1:N)
            while j == i
                j = rand(rng, 1:N)
            end
            s = 0.0
            for k in 1:d
                δ = samples[k, i] - samples[k, j]
                s += δ * δ
            end
            out[p] = sqrt(s)
        end
        return median(out)
    end
end

"""
    mmd_rbf(mr, truth; N=512, sigma_cache=nothing) -> Float64

RBF-kernel MMD with the median heuristic bandwidth. Computed
post-hoc on `N` samples per side (default 512 — three N×N kernel
matrices of ~2 MB each, fast to fill). For `W1_EVAL_N` samples
(10 000) the matrices would balloon to 800 MB each.

The bandwidth from the median heuristic is the same for every cell
sharing a `(problem, d)` truth — pass `sigma_cache::Dict` to avoid
recomputing it on every call.
"""
function mmd_rbf(mr::MethodResult, truth::TruthSet;
                  N::Integer = 512,
                  rng_seed::Integer = 20260607,
                  sigma_cache::Union{Nothing,AbstractDict} = nothing)
    rng = MersenneTwister(rng_seed + mr.seed)
    X = _resample_to_eval_N(mr, N, rng)
    Y = _truth_subsample(truth, N, rng)
    NX = size(X, 2); NY = size(Y, 2)
    if NX < 2 || NY < 2
        return NaN
    end
    # V8-FIX-B4: the cached bandwidth used to be computed with the
    # PER-CELL rng and per-cell truth subsample, so its value depended
    # on which cell a thread processed first — mmd_rbf was not
    # deterministic across aggregation runs. The bandwidth is now
    # derived from a truth subsample drawn with a FIXED problem/d-keyed
    # RNG, independent of the calling cell.
    σ_med = if sigma_cache !== nothing
        get!(sigma_cache, (truth.problem, truth.d, N)) do
            rng_bw = MersenneTwister(hash((truth.problem, truth.d, N, :mmd_bw)) % UInt32)
            Ybw = _truth_subsample(truth, N, rng_bw)
            median_pairwise_distance(Ybw; rng = rng_bw, max_pairs = 50_000)
        end
    else
        median_pairwise_distance(Y; rng = rng, max_pairs = 50_000)
    end
    (σ_med isa Real) && (σ_med <= 0 || !isfinite(σ_med)) && return NaN
    γ = 1.0 / (2 * σ_med * σ_med)
    # Compute the three pairwise sums.
    Kxx = Matrix{Float64}(undef, NX, NX)
    Kyy = Matrix{Float64}(undef, NY, NY)
    Kxy = Matrix{Float64}(undef, NX, NY)
    _self_pairwise_sqeuclidean!(Kxx, X)
    _self_pairwise_sqeuclidean!(Kyy, Y)
    _pairwise_sqeuclidean!(Kxy, X, Y)
    Kxx .= exp.(-γ .* Kxx)
    Kyy .= exp.(-γ .* Kyy)
    Kxy .= exp.(-γ .* Kxy)
    # Unbiased MMD² with diagonal removed for K_xx, K_yy.
    if NX > 1 && NY > 1
        m_xx = (sum(Kxx) - NX) / (NX * (NX - 1))      # diag is 1
        m_yy = (sum(Kyy) - NY) / (NY * (NY - 1))
    else
        m_xx = mean(Kxx)
        m_yy = mean(Kyy)
    end
    m_xy = mean(Kxy)
    val = m_xx + m_yy - 2 * m_xy
    return sqrt(max(val, 0.0))
end


# -----------------------------------------------------------------------------
# §9.9 — Bootstrap median CIs (V3-S-2)
# -----------------------------------------------------------------------------

"""
    bootstrap_median_ci(values; nboot=1000, rng) -> (lo, median, hi)

Bootstrap 95 % confidence interval on the median of `values`. Empty /
all-NaN inputs return `(NaN, NaN, NaN)`.
"""
function bootstrap_median_ci(values::AbstractVector{<:Real};
                                  nboot::Integer = 1000,
                                  rng::AbstractRNG = MersenneTwister(31337))
    v = collect(Float64, values)
    v = filter(isfinite, v)
    n = length(v)
    n == 0 && return (NaN, NaN, NaN)
    n == 1 && return (v[1], v[1], v[1])
    boots = Vector{Float64}(undef, nboot)
    @inbounds for b in 1:nboot
        s = 0.0
        # Sample with replacement.
        ix = rand(rng, 1:n, n)
        boots[b] = median(v[ix])
        s = boots[b]
    end
    lo = quantile(boots, 0.025)
    hi = quantile(boots, 0.975)
    return (lo, median(v), hi)
end

"""
    bootstrap_paired_median_ci(matrix; nboot=1000, rng)

Paired bootstrap CI on the median for a `n_seeds × n_metrics` matrix
where the rows are the *paired observations* (same seed across
algorithms in a cell). Resamples row indices, then computes the
median per column on the resampled rows. Returns a tuple
`(los, medians, his)` of length `n_metrics`.
"""
function bootstrap_paired_median_ci(matrix::AbstractMatrix{<:Real};
                                        nboot::Integer = 1000,
                                        rng::AbstractRNG = MersenneTwister(31337))
    n_seeds, n_metrics = size(matrix)
    if n_seeds == 0
        return (fill(NaN, n_metrics), fill(NaN, n_metrics), fill(NaN, n_metrics))
    end
    boots = Matrix{Float64}(undef, nboot, n_metrics)
    @inbounds for b in 1:nboot
        ix = rand(rng, 1:n_seeds, n_seeds)
        for k in 1:n_metrics
            col = view(matrix, ix, k)
            v = filter(isfinite, collect(Float64, col))
            boots[b, k] = isempty(v) ? NaN : median(v)
        end
    end
    los = Vector{Float64}(undef, n_metrics)
    medians = Vector{Float64}(undef, n_metrics)
    his = Vector{Float64}(undef, n_metrics)
    @inbounds for k in 1:n_metrics
        col = filter(isfinite, boots[:, k])
        los[k]    = isempty(col) ? NaN : quantile(col, 0.025)
        his[k]    = isempty(col) ? NaN : quantile(col, 0.975)
        col_obs   = filter(isfinite, collect(Float64, view(matrix, :, k)))
        medians[k] = isempty(col_obs) ? NaN : median(col_obs)
    end
    return (los, medians, his)
end


# -----------------------------------------------------------------------------
# §11.4 — Pairwise statistical tests (V3-S-3) and Friedman+Nemenyi (V3-S-4)
# -----------------------------------------------------------------------------

"""
    wilcoxon_signed_rank(a, b; tail=:left) -> (statistic, p_value, n_used)

Exact Wilcoxon paired signed-rank test (lower-tailed `tail = :left`
tests the alternative `median(a − b) < 0`, i.e. *`a` is smaller than
`b`* — used when "smaller is better" — e.g. lower W₁ for MW vs.
comparator). Ties are broken by mid-rank assignment; pairs with
zero difference are dropped. The exact-null p-value is computed via
the recursion `c[n, S]` of integer-rank-sum partitions; for `n ≤ 50`
this is essentially free. For `n > 50` we fall back to the
asymptotic normal approximation with continuity correction.
"""
function wilcoxon_signed_rank(a::AbstractVector{<:Real},
                                  b::AbstractVector{<:Real};
                                  tail::Symbol = :left)
    @assert length(a) == length(b) "wilcoxon_signed_rank: length mismatch"
    diffs = collect(Float64, a) .- collect(Float64, b)
    finite = isfinite.(diffs)
    diffs = diffs[finite]
    diffs = diffs[diffs .!= 0.0]   # drop zero-diff pairs
    n = length(diffs)
    if n == 0
        return (NaN, 1.0, 0)
    end
    abs_d = abs.(diffs)
    sgn = sign.(diffs)
    # Mid-ranks (handling ties).
    perm = sortperm(abs_d)
    ranks = Vector{Float64}(undef, n)
    i = 1
    while i <= n
        j = i
        while j < n && abs_d[perm[j + 1]] == abs_d[perm[i]]
            j += 1
        end
        avg = (i + j) / 2
        for k in i:j
            ranks[perm[k]] = avg
        end
        i = j + 1
    end
    Wpos = sum(ranks[k] for k in 1:n if sgn[k] > 0; init = 0.0)
    Wneg = sum(ranks[k] for k in 1:n if sgn[k] < 0; init = 0.0)
    statistic = min(Wpos, Wneg)
    if n <= 50
        # Exact null distribution via integer-rank-sum convolution.
        max_S = round(Int, n * (n + 1) / 2)
        # Coefficients of Π (1 + x^r_k); each rank contributes one factor.
        # With mid-ranks the "rank" can be a half-integer; multiply by 2 to
        # work in integers.
        scale = 2
        max_int = round(Int, scale * max_S)
        poly = zeros(Float64, max_int + 1)
        poly[1] = 1.0
        for r in ranks
            r_i = round(Int, scale * r)
            r_i <= 0 && continue
            new_poly = copy(poly)
            for k in 1:length(poly)
                if poly[k] != 0.0 && k + r_i <= length(new_poly)
                    new_poly[k + r_i] += poly[k]
                end
            end
            poly = new_poly
        end
        total = sum(poly)
        if tail == :left
            # Want P(W ≤ Wpos) for "a < b" alternative; the test
            # statistic under the null with mean n(n+1)/4 is symmetric.
            # We score with the *positive* sum: small Wpos ⇒ a < b.
            target_int = round(Int, scale * Wpos)
            cum = sum(poly[1:min(target_int + 1, length(poly))])
            p = cum / total
        elseif tail == :right
            target_int = round(Int, scale * Wpos)
            cum = sum(poly[max(target_int + 1, 1):end])
            p = cum / total
        else
            # Two-sided: 2 × min of one-sided.
            tl = sum(poly[1:min(round(Int, scale * Wpos) + 1, length(poly))]) / total
            tr = sum(poly[max(round(Int, scale * Wpos) + 1, 1):end]) / total
            p = min(1.0, 2 * min(tl, tr))
        end
        return (statistic, clamp(p, 0.0, 1.0), n)
    else
        μ = n * (n + 1) / 4
        # V8-FIX-B4: tie-corrected variance (Σ(t³−t)/48 subtracted per
        # tie group of size t) and a PER-TAIL continuity correction —
        # the old code applied −0.5 to both tails, which made the left
        # tail anti-conservative: P(W ≤ w) needs +0.5, P(W ≥ w) −0.5.
        tie_corr = 0.0
        i = 1
        sorted_abs = sort(abs_d)
        while i <= n
            j = i
            while j < n && sorted_abs[j + 1] == sorted_abs[i]
                j += 1
            end
            t = j - i + 1
            t > 1 && (tie_corr += t^3 - t)
            i = j + 1
        end
        σ2 = n * (n + 1) * (2n + 1) / 24 - tie_corr / 48
        σ = sqrt(max(σ2, eps()))
        p = if tail == :left
            cdf(Normal(), (Wpos - μ + 0.5) / σ)
        elseif tail == :right
            1 - cdf(Normal(), (Wpos - μ - 0.5) / σ)
        else
            2 * min(cdf(Normal(), (Wpos - μ + 0.5) / σ),
                    1 - cdf(Normal(), (Wpos - μ - 0.5) / σ))
        end
        return (statistic, clamp(p, 0.0, 1.0), n)
    end
end

"""
    cliffs_delta(a, b) -> Float64

Non-parametric effect size in `[-1, 1]`. δ > 0 indicates `a` tends to
be larger than `b`. For "lower is better" headlines (W₁), MW is
better when `δ < 0`.
"""
function cliffs_delta(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    av = collect(Float64, a); bv = collect(Float64, b)
    av = av[isfinite.(av)]
    bv = bv[isfinite.(bv)]
    isempty(av) && return NaN
    isempty(bv) && return NaN
    pos = 0; neg = 0
    @inbounds for x in av, y in bv
        if x > y
            pos += 1
        elseif x < y
            neg += 1
        end
    end
    return (pos - neg) / (length(av) * length(bv))
end

"""
    holm_bonferroni(pvals) -> Vector{Float64}

Holm-Bonferroni step-down adjustment on a vector of raw p-values.
Returns adjusted p-values in the same order as the input.
"""
function holm_bonferroni(pvals::AbstractVector{<:Real})
    p = collect(Float64, pvals)
    n = length(p)
    n == 0 && return p
    perm = sortperm(p)
    adj = similar(p)
    cur = 0.0
    @inbounds for i in 1:n
        m = n - i + 1
        x = m * p[perm[i]]
        cur = max(cur, x)
        adj[perm[i]] = min(cur, 1.0)
    end
    return adj
end

"""
    friedman_chi2(scores) -> (chi2, df, p_value)

Friedman test on a `n_blocks × n_treatments` score matrix; lower
score = better treatment per row. Returns Friedman χ² statistic, its
degrees of freedom (`n_treatments − 1`), and the asymptotic p-value
(Chi-squared null).
"""
function friedman_chi2(scores::AbstractMatrix{<:Real})
    n, k = size(scores)
    n == 0 && return (NaN, k - 1, NaN)
    # Per-row average ranks (handles ties via mid-rank).
    ranks = zeros(Float64, n, k)
    @inbounds for i in 1:n
        row = collect(Float64, scores[i, :])
        finite = isfinite.(row)
        if !any(finite)
            ranks[i, :] .= NaN
            continue
        end
        perm = sortperm(row)
        # Compute mid-ranks
        rk = Vector{Float64}(undef, k)
        idx = 1
        while idx <= k
            j = idx
            while j < k && row[perm[j + 1]] == row[perm[idx]]
                j += 1
            end
            avg = (idx + j) / 2
            for s in idx:j
                rk[perm[s]] = avg
            end
            idx = j + 1
        end
        ranks[i, :] = rk
    end
    valid = .!any(isnan, ranks; dims = 2)[:]
    ranks = ranks[valid, :]
    n_eff = size(ranks, 1)
    if n_eff < 2
        return (NaN, k - 1, NaN)
    end
    R̄ = vec(mean(ranks; dims = 1))
    χ2 = (12 * n_eff / (k * (k + 1))) * sum(R̄ .^ 2) -
         3 * n_eff * (k + 1)
    # V8-FIX-B4: tie correction (Iman & Davenport / Conover): divide by
    # C = 1 − Σ_blocks Σ_groups (t³ − t) / (n · k · (k² − 1)). Without
    # it the statistic is conservative when blocks contain tied scores.
    tie_sum = 0.0
    @inbounds for i in 1:n_eff
        row = sort(collect(Float64, ranks[i, :]))
        j = 1
        while j <= k
            l = j
            while l < k && row[l + 1] == row[j]
                l += 1
            end
            t = l - j + 1
            t > 1 && (tie_sum += t^3 - t)
            j = l + 1
        end
    end
    C = 1.0 - tie_sum / (n_eff * k * (k^2 - 1))
    C > 0 && (χ2 /= C)
    df = k - 1
    p = 1 - cdf(Chisq(df), χ2)
    return (χ2, df, p)
end

"""
    nemenyi_critical_difference(k, n; α=0.05) -> Float64

Nemenyi critical difference for `k` algorithms over `n` blocks at
significance `α`. Studentised range constants from Demšar (2006)
table 5; for `k ≤ 10` we hard-code the values, otherwise the
asymptotic formula `q_α / sqrt(2)` is used.
"""
function nemenyi_critical_difference(k::Integer, n::Integer; α::Real = 0.05)
    # q_α values from Demšar (2006) tables 5a/5b.
    q05 = Dict(
        2 => 1.960, 3 => 2.343, 4 => 2.569, 5 => 2.728,
        6 => 2.850, 7 => 2.949, 8 => 3.031, 9 => 3.102,
        10 => 3.164,
    )
    q01 = Dict(
        2 => 2.576, 3 => 2.913, 4 => 3.113, 5 => 3.255,
        6 => 3.364, 7 => 3.452, 8 => 3.526, 9 => 3.590,
        10 => 3.646,
    )
    qmap = α <= 0.011 ? q01 : q05
    q = get(qmap, k, k <= 10 ? 3.164 :
            (α <= 0.011 ? 3.646 + 0.05 * (k - 10) : 3.164 + 0.05 * (k - 10)))
    return q * sqrt(k * (k + 1) / (6 * n))
end


# -----------------------------------------------------------------------------
# Bundled metric extraction (one row of cells.csv)
# -----------------------------------------------------------------------------

const _BUDGET_TOL = 0.20    # §6.1: |Nlike_used / B - 1| > 0.20 ⇒ BUDGET-VIOLATION
const _RHAT_THRESH = 1.05    # §9.3a: max R̂ > 1.05 ⇒ RHAT-FAIL

# §6.1.1 (rev May 2026, V2-FIX-1) — algorithms whose under-spend is
# protocol-allowed when the named stop reason fires. The plot pipeline
# treats these cells as advisory (rendered with a saturation glyph,
# §10.16.12) instead of dropping them as BUDGET-VIOLATION.
#
# - MoleWhacker stops at `T_max` or `Neff_target` per the April 2026
#   protocol revision (the proposal has saturated; further cost
#   would not improve coverage).
# - Nested sampling exits at `dlogz` once the evidence has converged,
#   even if `B` is unconsumed.
const _BUDGET_EXEMPT_TERMINATORS = Dict{Symbol,Tuple{Vararg{Symbol}}}(
    :mw => (:T_max, :Neff_target),
    :ns => (:dlogz,),
)

function _is_budget_exempt(alg::Symbol,
                           terminated_by::Union{Symbol,AbstractString,Nothing,Missing})
    tb = if terminated_by === nothing || terminated_by === missing
        :unknown
    elseif terminated_by isa AbstractString
        Symbol(terminated_by)
    else
        terminated_by
    end
    haskey(_BUDGET_EXEMPT_TERMINATORS, alg) || return false
    return tb in _BUDGET_EXEMPT_TERMINATORS[alg]
end

"""
    compute_metrics_for_cell(mr, truth) -> Dict{Symbol,Any}

Computes every metric the protocol's cells.csv (Protocol §8) needs.
Augments `notes` with `BUDGET-VIOLATION` (§6.1) and / or `RHAT-FAIL`
(§9.3a) flags so downstream aggregation can exclude these cells from
the headline tables.

The budget rule honours the protocol carve-out of §6.1.1: a cell is
flagged `BUDGET-VIOLATION` only when its under- or over-spend is *not*
explained by an algorithm-specific protocol-allowed terminator
(`mw → {T_max, Neff_target}`, `ns → {dlogz}`). Advisory cells are
rendered with a saturation glyph in the plotting layer; the value is
preserved.
"""
function compute_metrics_for_cell(mr::MethodResult, truth::TruthSet;
                                       sigma_cache::Union{Nothing,AbstractDict} = nothing,
                                       with_mmd::Bool = false)
    d = size(mr.samples, 1)
    out = Dict{Symbol,Any}()
    out[:problem]              = String(mr.problem)
    out[:algorithm]            = String(mr.algorithm)
    out[:d]                    = mr.d
    out[:B]                    = mr.B
    out[:seed]                 = mr.seed
    out[:Nlike_used]           = mr.Nlike_used
    out[:wall_time_s]          = mr.wall_time_s
    out[:n_primal]             = mr.n_primal
    out[:n_grad_partials]      = mr.n_grad_partials
    # Number of ForwardDiff gradient evaluations.
    # `-1` marks a cell whose result.h5 predates the field (only NUTS is
    # re-run for V6, so non-NUTS cells keep the sentinel; the V6 sanity
    # gate only inspects NUTS rows, which carry the true count).
    out[:n_dual_calls]         = Int(get(mr.extras, :n_dual_calls, -1))

    # Budget-infeasible cells carry NO quality
    # metrics (honest N/A, never padded/guessed). When one honestly-tuned
    # gradient-sampler trajectory (minimum warmup + one transition) costs
    # > 1.2·B, the runner returns a 1-row placeholder tagged
    # `:budget_infeasible`. We keep the cost/eval columns (they document
    # WHY the cell is infeasible: `Nlike_used` is the minimum-trajectory
    # cost, > 1.2·B) and the `budget_infeasible` terminator, but write
    # every distributional metric as NaN so no figure or table reads a
    # number derived from the placeholder sample.
    if get(mr.extras, :budget_infeasible, false) === true ||
       String(string(get(mr.extras, :stop_reason, :unknown))) == "budget_infeasible"
        for k in (:Neff, :eta_Nlike, :W1_marginal_avg, :SWD, :dlogZ,
                  :QE_p025, :QE_p160, :QE_p500, :QE_p840, :QE_p975,
                  :KL_cube, :mode_recovery, :mmd_rbf, :Rhat_max)
            out[k] = NaN
        end
        out[:terminated_by] = "budget_infeasible"
        ratio = mr.B > 0 ? round(mr.Nlike_used / mr.B; digits = 2) : NaN
        out[:notes] = "budget-infeasible (one trajectory ~ $(ratio)xB > 1.2B)"
        return out
    end

    out[:Neff]                 = neff(mr)
    out[:eta_Nlike]            = out[:Neff] / max(mr.Nlike_used, 1.0)
    out[:W1_marginal_avg]      = w1_marginal_avg(mr, truth)
    out[:SWD]                  = sliced_wasserstein(mr, truth; seed = 12345 + mr.seed)
    dz = dlogZ(mr, truth)
    out[:dlogZ]                = dz === missing ? NaN : dz
    qe = quantile_errors(mr, truth)
    out[:QE_p025]              = qe[:p025]
    out[:QE_p160]              = qe[:p160]
    out[:QE_p500]              = qe[:p500]
    out[:QE_p840]              = qe[:p840]
    out[:QE_p975]              = qe[:p975]
    out[:KL_cube]              = mr.algorithm == :mw ?
        kl_cube_mw(mr, truth, truth.cfg) : NaN
    out[:Rhat_max]             = rhat_max(mr)
    out[:terminated_by]        = String(get(mr.extras, :stop_reason, :unknown))

    # Protocol §9.7 — mode recovery rate for multimodal targets.
    eps_mr = _eps_radius_for_problem(truth)
    out[:mode_recovery] = isnan(eps_mr) ? NaN :
        mode_recovery(mr, truth, eps_mr)
    # Protocol §9.8 — RBF-MMD with median heuristic.
    # OFF by default (~9 min per 375 cells); enable via aggregator's
    # `--with-mmd` flag for the appendix figures.
    out[:mmd_rbf] = if with_mmd
        try
            mmd_rbf(mr, truth; sigma_cache = sigma_cache)
        catch
            NaN
        end
    else
        NaN
    end

    # §6.1 + §9.3a flags
    flags = String[]
    if mr.B > 0
        ratio = mr.Nlike_used / mr.B
        out_of_band = ratio < (1 - _BUDGET_TOL) || ratio > (1 + _BUDGET_TOL)
        if out_of_band &&
           !_is_budget_exempt(mr.algorithm,
                              get(mr.extras, :stop_reason, :unknown))
            push!(flags, "BUDGET-VIOLATION")
        end
    end
    if mr.algorithm in (:mh, :nuts)
        rmax = out[:Rhat_max]
        if isfinite(rmax) && rmax > _RHAT_THRESH
            push!(flags, "RHAT-FAIL")
        end
    end
    base_notes = String(get(mr.extras, :notes, ""))
    if !isempty(flags)
        flag_str = join(flags, ";")
        out[:notes] = isempty(base_notes) ? flag_str : (base_notes * ";" * flag_str)
    else
        out[:notes] = base_notes
    end
    return out
end

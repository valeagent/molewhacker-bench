# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# algo_mh.jl — random-walk Metropolis–Hastings (Protocol §5.2, §6.1, §9.3a)
# =============================================================================
#
# Three-phase tuning, all consuming the budget B (cost = primal calls):
#   1. Pilot     :   3% of B at σ = 1.0,        ≥ 200 steps
#   2. Tuning    :   5% of B, binary search σ to hit acceptance ∈ [0.20, 0.30]
#   3. Production:  remaining 92% of B, recorded as the chain.
#
# **Multi-chain output (§9.3a).** To make the Gelman–Rubin R̂ diagnostic
# meaningful on multimodal targets we run `nchains = 4` chains, each
# starting from an independent uniform-cube draw. Production samples
# are written in chain-major order so the metrics layer can reshape
# `samples` to `n_per_chain × nchains` directly. Tuning is shared
# across chains (the proposal is a single-σ random-walk; tuning σ on
# one chain transfers to the others by Roberts–Gelman–Gilks).
#
# Out-of-cube proposals evaluate to `-Inf` (the wrapped log-density
# below); these still bump the counter so the budget contract is
# honoured even when the chain rejects.

"""
    run_mh(cfg::ProblemConfig, B::Real, seed::Integer; counter=nothing,
           pilot_frac=0.03, tune_frac=0.05, nchains=4) -> MethodResult
"""
function run_mh(cfg::ProblemConfig, B::Real, seed::Integer;
                  counter::Union{LikelihoodCounter,Nothing} = nothing,
                  pilot_frac::Real = 0.03,
                  tune_frac::Real = 0.05,
                  nchains::Integer = 4)
    log_f = build_log_f(cfg)
    counter === nothing && (counter = LikelihoodCounter(log_f))
    reset!(counter)
    d = cfg.d
    L = cfg.L

    pilot_B = max(200, floor(Int, pilot_frac * B))
    tune_B  = max(100, floor(Int, tune_frac * B))

    # Wrap log_f so that out-of-cube ⇒ -Inf (and counter still bumps).
    function logπ(θ)
        if any(abs.(θ) .>= L)
            counter(θ)              # still a primal call
            return -Inf
        end
        return counter(θ)
    end

    rng = Xoshiro(seed)

    t0 = time_ns()

    # ------------------------------------------------------------------
    # Pilot + tuning (shared across chains).
    # ------------------------------------------------------------------
    σ = 1.0
    x = (rand(rng, d) .- 0.5) .* (2L * 0.5)
    lp = logπ(x)
    while !isfinite(lp)
        x = (rand(rng, d) .- 0.5) .* (2L * 0.5)
        lp = logπ(x)
    end
    pilot_acc = _mh_run!(rng, x, lp, σ, pilot_B, logπ, d)
    pilot_rate = pilot_acc.rate

    σ_lo, σ_hi = σ / 64, σ * 64
    target_lo, target_hi = 0.20, 0.30
    spent = 0
    last_rate = pilot_rate
    last_σ = σ
    chunk = max(50, div(tune_B, 8))
    while spent < tune_B && (last_rate < target_lo || last_rate > target_hi)
        if last_rate > target_hi
            σ_lo = last_σ
            σ = sqrt(last_σ * σ_hi)
        elseif last_rate < target_lo
            σ_hi = last_σ
            σ = sqrt(last_σ * σ_lo)
        end
        if !isfinite(logπ(x))
            x = (rand(rng, d) .- 0.5) .* (2L * 0.5)
        end
        lp = logπ(x)
        run_n = min(chunk, tune_B - spent)
        res = _mh_run!(rng, x, lp, σ, run_n, logπ, d)
        last_rate = res.rate
        last_σ = σ
        spent += res.n
    end
    σ_final = σ

    # ------------------------------------------------------------------
    # Production — 4 chains in chain-major order.
    # ------------------------------------------------------------------
    spent_total = cost(counter)
    remaining_total = max(B - spent_total, 100.0)
    # Equal share per chain
    per_chain_steps = max(50, floor(Int, remaining_total / nchains))

    samples_chains = Vector{Matrix{Float64}}()
    logd_chains = Vector{Vector{Float64}}()
    chain_acc_rates = Float64[]
    n_per_chain = Int[]

    for c in 1:nchains
        # Independent starting state per chain
        chain_rng = Xoshiro(Int(seed) * 100 + c)
        x_c = (rand(chain_rng, d) .- 0.5) .* (2L * 0.95)
        lp_c = logπ(x_c)
        # Re-init if we land outside cube on a degenerate problem
        tries = 0
        while !isfinite(lp_c) && tries < 100
            x_c = (rand(chain_rng, d) .- 0.5) .* (2L * 0.95)
            lp_c = logπ(x_c)
            tries += 1
        end
        if !isfinite(lp_c)
            push!(samples_chains, zeros(d, 1))
            push!(logd_chains, [NaN])
            push!(chain_acc_rates, 0.0)
            push!(n_per_chain, 0)
            continue
        end

        # Run this chain until either per_chain_steps or global budget.
        max_steps_c = per_chain_steps
        samples_c = Matrix{Float64}(undef, d, max_steps_c)
        logds_c = Vector{Float64}(undef, max_steps_c)
        n_acc = 0
        written = 0
        for it in 1:max_steps_c
            if cost(counter) >= B
                break
            end
            prop = x_c .+ σ_final .* randn(chain_rng, d)
            lp_new = logπ(prop)
            if log(rand(chain_rng)) < lp_new - lp_c
                x_c = prop
                lp_c = lp_new
                n_acc += 1
            end
            written += 1
            @views samples_c[:, written] .= x_c
            logds_c[written] = lp_c
        end
        if written == 0
            push!(samples_chains, zeros(d, 1))
            push!(logd_chains, [NaN])
            push!(chain_acc_rates, 0.0)
            push!(n_per_chain, 0)
        else
            push!(samples_chains, samples_c[:, 1:written])
            push!(logd_chains, logds_c[1:written])
            push!(chain_acc_rates, n_acc / written)
            push!(n_per_chain, written)
        end
    end

    wall_time = (time_ns() - t0) / 1e9

    # Truncate every chain to the shortest, write in chain-major order.
    valid = filter(>(0), n_per_chain)
    if isempty(valid)
        # No usable production chain; emit a placeholder.
        samples = zeros(d, 1)
        logds = [NaN]
        nchains_eff = 0
        n_min = 0
    else
        n_min = minimum(valid)
        nchains_eff = length(valid)
        samples = Matrix{Float64}(undef, d, n_min * nchains_eff)
        logds = Vector{Float64}(undef, n_min * nchains_eff)
        idx_out = 0
        for c in 1:length(samples_chains)
            n_per_chain[c] == 0 && continue
            sc = samples_chains[c]
            lc = logd_chains[c]
            for i in 1:n_min
                idx_out += 1
                @views samples[:, idx_out] .= sc[:, i]
                logds[idx_out] = lc[i]
            end
        end
    end

    extras = Dict{Symbol,Any}(
        :stop_reason => :budget,
        :pilot_steps => pilot_B,
        :pilot_acc_rate => pilot_rate,
        :tune_steps => spent,
        :tune_final_acc_rate => last_rate,
        :sigma_final => σ_final,
        :nchains => nchains_eff,
        :n_per_chain => fill(n_min, nchains_eff),
        :n_per_chain_raw => n_per_chain,
        :chain_acc_rates => chain_acc_rates,
        :production_steps_total => sum(n_per_chain),
    )

    return MethodResult(
        algorithm = :mh,
        problem = _problem_symbol(cfg),
        d = d,
        seed = Int(seed),
        B = Float64(B),
        counter = counter,
        wall_time_s = wall_time,
        samples = samples,
        weights = Float64[],
        logd = logds,
        logZ_estimate = missing,
        logZ_estimate_se = missing,
        extras = extras,
    )
end


# Inner MH sampler that updates state in place; returns acceptance rate
# and step count actually taken (may stop early on budget).
function _mh_run!(rng, x::AbstractVector, lp::Real, σ::Real, n_steps::Int,
                    logπ::Function, d::Int)
    n_acc = 0
    n = 0
    @inbounds for _ in 1:n_steps
        prop = x .+ σ .* randn(rng, d)
        lp_new = logπ(prop)
        if log(rand(rng)) < lp_new - lp
            x .= prop
            lp = lp_new
            n_acc += 1
        end
        n += 1
    end
    return (rate = n_acc / max(n, 1), n = n, lp = lp)
end

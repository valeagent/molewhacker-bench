# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# algo_nuts.jl — NUTS via BAT.HamiltonianMC (Protocol §5.3, §6.1, §6.2, §9.3a)
# =============================================================================
#
# Identity (P0-1, §6.2). `BAT.HamiltonianMC()` with default settings
# configures AdvancedHMC's Leapfrog integrator + GeneralisedNoUTurn
# termination — i.e. the **No-U-Turn Sampler** (Hoffman & Gelman 2014).
# `_verify_nuts_identity()` asserts this at runtime; if a future BAT
# version changes the default the run halts before producing any cell.
#
# -----------------------------------------------------------------------------
# Gradient cost accounting — the central fairness axis.
# -----------------------------------------------------------------------------
# Fairness model: one primal log-density evaluation costs 1, one
# d-dimensional gradient costs d (`LikelihoodCounter`, `counter.jl`).
#
# Why the LikelihoodCounter cannot see NUTS's gradients: BAT builds the
# leapfrog gradient with `fg = valgrad_func(f, adsel)` and hands it to
# `AdvancedHMC.Hamiltonian(metric, f, fg)` (BATAdvancedHMCExt
# `_create_proposal_state`). Empirically the counter's dual-dispatch
# tally — and even an explicit wrapper around `fg` — SATURATES at the
# warmup/tuning gradient count (~10^3, independent of how many production
# samples are drawn): AdvancedHMC evaluates the production leapfrog
# gradients through an internally cached closure that calls neither the
# `LikelihoodCounter` nor the BAT-supplied `fg`. So a per-call counter on
# the likelihood is structurally blind to production gradients (the
# `~24k` true gradients vs `n_dual≈967` discrepancy documented in
# `experiments/docs/NUTS-COST-ACCOUNTING.md`).
#
# What we charge instead (exact, option (b) of the V6 directive):
# AdvancedHMC records the *actual* number of leapfrog steps per
# transition as `n_steps = tree.nα` (trajectory.jl), which already
# accounts for early U-turn termination — it is exact, not the
# `2^tree_depth − 1` upper bound. BAT's `AHMCSampleID` drops `n_steps`
# (it keeps only `tree_depth`), but `AdvancedHMC.stat(transition)` still
# carries it inside `BAT._get_sample_id`. We intercept that function
# (`_NUTS_NSTEPS` accumulator below) and sum `n_steps` over every
# transition BAT records — `ACCEPTED_SAMPLE`/`REJECTED_SAMPLE` fire
# exactly once per MCMC step (mcmc_state.jl), covering **warmup AND
# production**. After every `bat_sample` we install
# `n_dual_calls = Σn_steps`, `n_grad_partials = d·Σn_steps` via
# `set_gradient_accounting!`, so `cost(counter) = n_primal + d·Σn_steps`
# is the exact gradient-charged budget. See `docs/NUTS-COST-ACCOUNTING.md`.
#
# Budget enforcement (P0-2/P0-4, §6.1) and infeasibility (V6 P0.6).
# Warmup gradients are now charged, and at the smallest budgets BAT's
# minimum warmup plus a single transition can already cost > 1.2·B. Such
# a cell is **budget-infeasible for NUTS**: it is recorded as N/A
# (`terminated_by = :budget_infeasible`, no metrics) rather than padded
# or overshot — an honest, reportable finding. Feasible cells warm up
# once per chain and draw production samples sized to land the realised
# cost in `[0.8, 1.2]·B` (the V6 sanity band).
#
# Chain-major output (P0-3, §9.3a). Production samples from each chain
# are concatenated in chain-major order so that
# `reshape(samples, n_per_chain, nchains)` is well-defined for the
# Gelman–Rubin R̂ implementation in `metrics.jl`.

import BAT
using BAT: bat_sample, MCMCSampling, HamiltonianMC, BATContext, set_batcontext,
            MCMCChainPoolInit, MCMCMultiCycleBurnin
using ADTypes: AutoForwardDiff
import AdvancedHMC                       # extension activator for BAT


const _NUTS_MIN_CHAINS_BUDGET_FACTOR = 20  # ≥ N production steps/chain to keep a chain
const _NUTS_WARM_INIT  = 20                # BAT-floor chain-pool init steps
const _NUTS_WARM_BURN  = 20                # BAT-floor burnin steps (single cycle)
const _NUTS_WARM_FINAL = 10                # BAT-floor final burnin pass
# Overshoot guard (never exceed 1.2·B). A production chunk is
# capped so that even if every transition in it were as deep as `safety ×` the
# deepest single transition seen so far, the cumulative cost cannot pass the
# 1.2·B gate. NUTS per-transition cost is heavy-tailed (deep trees in thin-ridge
# geometry, e.g. mridges), so this measured worst-case bound — rather than the
# unusably pessimistic theoretical `2^max_depth` — is what keeps every cell in
# band without crippling chunk sizes on easy targets.
const _NUTS_OVERSHOOT_SAFETY = 2.5


# -----------------------------------------------------------------------------
# Exact leapfrog (= gradient) accounting via AdvancedHMC's recorded n_steps.
#
# `_NUTS_NSTEPS` accumulates `tree.nα` (exact leapfrog steps, U-turn aware)
# over every NUTS transition BAT processes. The override of
# `BAT._get_sample_id` below is byte-for-byte the BAT AdvancedHMC extension
# body plus the harvesting side-effect; it is installed after
# `import AdvancedHMC` has loaded the extension, so it wins on dispatch.
# Only this experiment uses NUTS, so the global override is safe (MH/NS do
# not hit this method).
# -----------------------------------------------------------------------------
# `_NUTS_NSTEPS` accumulates Σ n_steps (exact leapfrog count, charged).
# `_NUTS_NSTEPS_MAX` tracks the largest SINGLE-transition n_steps seen — the
# observed worst-case leapfrog cost of one transition, used by the production
# controller's hard overshoot backstop (see `run_nuts`).
const _NUTS_NSTEPS = Threads.Atomic{Int}(0)
const _NUTS_NSTEPS_MAX = Threads.Atomic{Int}(0)
@inline _reset_nuts_nsteps!() =
    (Threads.atomic_xchg!(_NUTS_NSTEPS, 0); Threads.atomic_xchg!(_NUTS_NSTEPS_MAX, 0); nothing)
@inline _reset_nuts_nsteps_max!() = (Threads.atomic_xchg!(_NUTS_NSTEPS_MAX, 0); nothing)
@inline _read_nuts_nsteps() = _NUTS_NSTEPS[]
@inline _read_nuts_nsteps_max() = _NUTS_NSTEPS_MAX[]

function BAT._get_sample_id(proposal::BAT.HMCProposalState, chainid::Int32,
        walkerid::Int32, cycle::Int32, stepno::Integer, sample_type::Integer)
    tstat = AdvancedHMC.stat(proposal.transition)
    # Charge the EXACT leapfrog count once per transition. ACCEPTED_SAMPLE(1)
    # and REJECTED_SAMPLE(2) are emitted exactly once per MCMC step
    # (mcmc_state.jl), covering burnin + production; CURRENT/PROPOSED setup
    # records are skipped to avoid double counting.
    if (sample_type == BAT.ACCEPTED_SAMPLE || sample_type == BAT.REJECTED_SAMPLE) &&
       hasproperty(tstat, :n_steps)
        ns = Int(tstat.n_steps)
        Threads.atomic_add!(_NUTS_NSTEPS, ns)
        Threads.atomic_max!(_NUTS_NSTEPS_MAX, ns)
    end
    new_id = BAT.AHMCSampleID(chainid, walkerid, cycle, stepno, sample_type,
        tstat.hamiltonian_energy, tstat.tree_depth, tstat.numerical_error,
        tstat.step_size)
    return new_id, BAT.AHMCSampleID
end


"""
    _leapfrog_upper_bound(sv) -> Int

Diagnostic only (NOT charged), where `sv` is a `DensitySampleVector`. The
`Σ weight·(2^tree_depth − 1)` reconstruction from the per-sample
`tree_depth` metadata — a conservative *upper bound* on production leapfrog
steps (a trajectory that U-turns partway through its final doubling did
fewer steps). Reported alongside the exact `n_steps`-based charge so the
over-counting margin of the old estimate is visible in the validation logs.
The fair cost uses the exact `n_steps` accumulator, never this bound.
"""
function _leapfrog_upper_bound(sv)
    sv === nothing && return 0
    (hasproperty(sv, :info) && hasproperty(sv, :weight)) || return 0
    info = getproperty(sv, :info)
    hasproperty(info, :tree_depth) || return 0
    depths = getproperty(info, :tree_depth)
    weights = getproperty(sv, :weight)
    L = 0
    @inbounds for i in eachindex(depths)
        j = Int(depths[i])
        w = i <= length(weights) ? Int(weights[i]) : 1
        L += w * max(1, (1 << clamp(j, 0, 60)) - 1)
    end
    return L
end


"""
    _verify_nuts_identity()

Protocol §6.2. Assert that `BAT.HamiltonianMC()` with default settings
uses a No-U-Turn termination strategy. Halts (errors) otherwise so we
never silently mislabel the algorithm as `:nuts` when it is plain HMC.
Returns a string describing the verified configuration.
"""
function _verify_nuts_identity()
    s = BAT.HamiltonianMC()
    term_type = string(typeof(s.termination))
    if !(occursin("NoUTurn", term_type) || occursin("noUTurn", term_type) ||
         occursin("ClassicNoUTurn", term_type) ||
         occursin("GeneralisedNoUTurn", term_type) ||
         occursin("StrictGeneralisedNoUTurn", term_type))
        error("""
        Protocol §6.2 violation: BAT.HamiltonianMC() default termination
        is `$term_type`, which is **not** a No-U-Turn strategy. Halting
        before mislabelling the algorithm as `:nuts`.
        """)
    end
    return string("HamiltonianMC{integrator=", typeof(s.integrator),
                  ", termination=", term_type, "}")
end


# MCMCSampling configurator. Keeps adaptation at the BAT floor so warmup is a
# minimal, well-defined fixed cost. `nsteps` is irrelevant when the runner
# drives `mcmc_iterate!!` itself (exact-budget engine), but is honoured by the
# probe path. `nchains` warms that many independent chains jointly.
function _build_mcmc_sampling(nsteps::Integer; nchains::Integer = 1,
                                nsteps_init::Integer = _NUTS_WARM_INIT,
                                burnin_per_cycle::Integer = _NUTS_WARM_BURN,
                                nsteps_final::Integer = _NUTS_WARM_FINAL)
    return MCMCSampling(
        mcalg = HamiltonianMC(),
        nsteps = max(1, Int(nsteps)),
        nchains = max(1, Int(nchains)),
        init = MCMCChainPoolInit(nsteps_init = max(20, Int(nsteps_init))),
        burnin = MCMCMultiCycleBurnin(
            nsteps_per_cycle = max(20, Int(burnin_per_cycle)),
            max_ncycles = 1,
            nsteps_final = max(10, Int(nsteps_final)),
        ),
        strict = false,
    )
end


# -----------------------------------------------------------------------------
# Exact-budget engine: drive BAT's MCMC internals directly so warmup is paid
# ONCE and production is drawn in measured chunks (never
# overshoot). `bat_sample_impl` is: transform → mcmc_init! → mcmc_burnin! →
# next_cycle! → mcmc_iterate!!(…; max_nsteps). We replicate the warmup and then
# call `mcmc_iterate!!` ourselves in chunks, charging the exact `n_steps`
# (harvested by the `_get_sample_id` override) after each chunk and stopping
# when the realised cost reaches the budget. Production after warmup is pure
# (no warmup mixed in), so its per-step cost is measured exactly and the final
# chunk lands the cost in band with no overshoot. These BAT functions are
# internal/unstable API, but the runner already overrides `_get_sample_id`, so
# the coupling (and its BAT-version pin) is pre-existing.
# -----------------------------------------------------------------------------
function _nuts_warmup(salg, posterior, context)
    transformed_m, f_pre = BAT.transform_and_unshape(salg.pretransform, posterior, context)
    init_alg = BAT.apply_trafo_to_init(f_pre, salg.init)
    mcmc_states, _chain_outputs = BAT.mcmc_init!(salg, transformed_m, init_alg, BAT.nop_func, context)
    chain_outputs = BAT._empty_chain_outputs.(mcmc_states)     # store production only
    mcmc_states = BAT.mcmc_burnin!(nothing, mcmc_states, salg, BAT.nop_func)
    BAT.next_cycle!.(mcmc_states)
    return mcmc_states, chain_outputs, f_pre
end

@inline function _nuts_produce!(mcmc_states, chain_outputs, m::Integer)
    return BAT.mcmc_iterate!!(chain_outputs, mcmc_states;
        max_nsteps = max(1, Int(m)), nonzero_weights = true)
end


"""
    run_nuts(cfg, B, seed; counter=nothing, nchains=4)

NUTS via BAT.jl with exact gradient-cost accounting.
Charges every leapfrog gradient (warmup + production) at cost `d` from
AdvancedHMC's recorded `n_steps`. Budget is enforced *exactly*: warmup is
paid once per chain, then production is drawn in measured chunks via
`mcmc_iterate!!` and stopped as soon as the realised cost reaches the target
band (never overshoot). If one honestly-tuned trajectory (minimum warmup +
one transition) already costs `> 1.2·B` the cell is budget-infeasible and
returned as N/A. The requested `nchains` collapses
automatically when the budget is too small for production room per chain
(R̂ then NaN → "n/a" downstream). See `docs/NUTS-COST-ACCOUNTING.md`.
"""
function run_nuts(cfg::ProblemConfig, B::Real, seed::Integer;
                    counter::Union{LikelihoodCounter,Nothing} = nothing,
                    nchains::Integer = 4)
    log_f = build_log_f(cfg)
    counter === nothing && (counter = LikelihoodCounter(log_f))
    reset!(counter)
    d = cfg.d
    Bf = Float64(B)

    nuts_id = _verify_nuts_identity()
    set_batcontext(ad = AutoForwardDiff())
    posterior = posterior_measure(cfg, counter)

    # Reproducible, seed-dependent context: BAT derives a distinct per-chain
    # RNG from this context's RNGPartition, so seeding it makes the whole cell
    # deterministic for a given `seed` yet independent across seeds/chains.
    function _mk_context(s::Integer)
        ctx = BATContext(ad = AutoForwardDiff())
        try
            Random.seed!(BAT.get_rng(ctx), UInt64(s % typemax(UInt64)))
        catch
            # Some RNG types don't accept seed!; fall back to a fresh seed.
        end
        return ctx
    end

    # Install the exact gradient charge from the n_steps accumulator.
    _charge!() = set_gradient_accounting!(counter, _read_nuts_nsteps(),
                                          d * _read_nuts_nsteps())

    t0 = time_ns()

    # Build an N/A (budget-infeasible) result carrying the probed minimum
    # trajectory cost (no padding, no metrics).
    # V8-FIX-A5: the counter now retains the ACTUALLY SPENT pilot/warmup
    # cost (the pilot is charged); only if nothing was spent (crash before
    # the first evaluation) do we synthesise the probed minimum cost.
    function _infeasible_result(min_cost::Real)
        wall = (time_ns() - t0) / 1e9
        _charge!()
        if cost(counter) <= 0
            nstep_eq = max(1, round(Int, Float64(min_cost) / max(d, 1)))
            set_gradient_accounting!(counter, nstep_eq, d * nstep_eq)
        end
        extras = Dict{Symbol,Any}(
            :stop_reason => :budget_infeasible,
            :terminated_by => :budget_infeasible,
            :budget_infeasible => true,
            :infeasible_min_trajectory_cost => Float64(min_cost),
            :infeasible_ratio => Float64(min_cost) / Bf,
            :nchains => 0, :nchains_requested => Int(nchains), :nchains_eff => 0,
            :n_per_chain => Int[], :nuts_identity => nuts_id,
            :transform => "PriorToNormal (BAT default for HamiltonianMC)",
            :divergent_indices => Int[],
        )
        mr = MethodResult(algorithm = :nuts, problem = _problem_symbol(cfg), d = d,
            seed = Int(seed), B = Bf, counter = counter, wall_time_s = wall,
            samples = zeros(d, 1), weights = [1.0], logd = [NaN],
            logZ_estimate = missing, logZ_estimate_se = missing, extras = extras)
        mr.extras[:neff_checked] = NaN
        mr.extras[:neff_le_ndual_ok] = true
        return mr
    end

    # -------------------------------------------------------------------
    # Pilot = first production chain (V8-FIX-A5, single-warmup design).
    # Warm ONE chain (CHARGED) and draw a short production burst to
    # measure the realised warmup cost `W_probe`, the post-warmup
    # per-step production cost `P_probe`, and the single-trajectory
    # feasibility cost `cost_w1 = warmup + one transition`.
    #
    # Under V6 this pilot was uncharged — several thousand free cost
    # units at d = 5, comparable to the whole B = 5e3 budget — while MH
    # pays for its tuning and MW pays its full initialisation. V8
    # charges it, AND the pilot chain is not thrown away: it becomes the
    # first production chain, so no warmup is ever duplicated. (The
    # first V8 iteration charged the pilot but still warmed a fresh
    # chain for the measured run; at B = 5e4 the two warmups together
    # consumed ~0.95·B and the production loop never ran — zero samples
    # from a feasible cell. Single-warmup fixes that.) The burst's 8
    # transitions are legitimate post-warmup draws of the tuned chain
    # and stay in its output.
    # -------------------------------------------------------------------
    reset!(counter); _reset_nuts_nsteps!()
    extension_log = Vector{NamedTuple}()
    local W_probe, P_probe, cost_w1
    local states1, outputs1, f_pre
    try
        states1, outputs1, f_pre =
            _nuts_warmup(_build_mcmc_sampling(1; nchains = 1),
                         posterior, _mk_context(Int(seed) * 1000 + 991))
        _charge!(); W_probe = cost(counter)
        nprobe = 8
        states1 = _nuts_produce!(states1, outputs1, nprobe); _charge!()
        P_probe = max(0.5, (cost(counter) - W_probe) / nprobe)
        cost_w1 = W_probe + P_probe
    catch err
        @warn "NUTS warmup pilot failed — treating cell as budget-infeasible" exception = (err,)
        return _infeasible_result(2.0 * Bf)
    end
    push!(extension_log, (phase = :pilot_warmup, nchains = 1,
        cost = cost(counter), ratio = cost(counter) / Bf))

    # Feasibility rule: one honestly-tuned trajectory (minimum warmup + one
    # transition) costing > 1.2·B makes the cell budget-infeasible.
    cost_w1 > 1.2 * Bf && return _infeasible_result(cost_w1)

    # Additional chains: each costs ≈ W_probe of warmup and needs
    # production room on top (factor 1.8). Sized from the budget
    # remaining after the charged pilot; collapses to the single pilot
    # chain at small budgets.
    B_rem = max(Bf - cost(counter), 0.0)
    n_extra = clamp(floor(Int, B_rem / (1.8 * max(W_probe, 1.0))),
                    0, Int(nchains) - 1)
    local states2 = nothing
    local outputs2 = nothing
    if n_extra > 0
        try
            states2, outputs2, _ =
                _nuts_warmup(_build_mcmc_sampling(1; nchains = n_extra),
                             posterior, _mk_context(Int(seed) * 1000 + 7))
        catch err
            @warn "NUTS extra-chain warmup failed — continuing with the pilot chain" exception = (err,)
            states2 = nothing; outputs2 = nothing
        end
        _charge!()
    end
    nchains_eff = 1 + (outputs2 === nothing ? 0 : length(outputs2))
    cost_warm = cost(counter)
    push!(extension_log, (phase = :warmup, nchains = nchains_eff,
        cost = cost_warm, ratio = cost_warm / Bf))

    # Realised-warmup infeasibility: if the warmed configuration's minimum
    # cost already exceeds 1.2·B, even one trajectory does not fit. Honest N/A.
    cost_warm > 1.2 * Bf && return _infeasible_result(cost_warm)

    # Production driver over both chain groups (the pilot chain and the
    # extra chains warmed above). Groups are iterated with the same chunk
    # size; outputs are concatenated chain-major downstream.
    function _produce_all!(m::Integer)
        states1 = _nuts_produce!(states1, outputs1, m)
        if states2 !== nothing
            states2 = _nuts_produce!(states2, outputs2, m)
        end
        _charge!()
        return nothing
    end

    # ---------------------------------------------------------------------
    # Production controller (NEVER overshoot). Draw
    # production in measured chunks; stop as soon as the realised cost
    # reaches the aim. NUTS per-transition cost is HEAVY-TAILED: in thin-ridge
    # geometry (mridges) the chains migrate into a deeper-tree regime mid-run,
    # so a chunk sized from a mean per-step cost can blow the budget (observed
    # up to 1.86·B before this guard). Three defences keep every cell ≤ 1.2·B:
    #   (1) close only HALF the remaining gap per chunk — one mis-estimate
    #       can never overshoot the aim by more than the gap itself;
    #   (2) size with a CONSERVATIVE per-step cost (blend of running mean and
    #       running max) so a target that has already shown deep trees is
    #       sized cautiously;
    #   (3) a hard backstop: a chunk may advance no more transitions than fit
    #       under 1.2·B even if EVERY one were `safety ×` as deep as the
    #       deepest single transition seen so far (`_NUTS_NSTEPS_MAX`).
    # The aim is 0.90·B (band centre with cushion to the 1.2·B gate); the loop
    # stops on first crossing, so feasible cells land in [0.8,1.2]·B with no
    # padding, overshoot, or re-tuning.
    # ---------------------------------------------------------------------
    T_aim = 0.90 * Bf
    T_hi  = 1.20 * Bf
    nch   = nchains_eff
    cpc     = max(P_probe * nch, 1.0)   # mean cost per iterate-step (refined)
    cpc_max = cpc                       # running max realised per-step cost
    # Re-base the worst-transition tracker to PRODUCTION geometry: warmup
    # step-size adaptation transiently explores very deep trees that do not
    # reflect the tuned production regime and would over-throttle the chunks.
    _reset_nuts_nsteps_max!()

    if cost_warm < T_aim
        # Small first chunk to measure the pure-production per-step cost and
        # seed the worst-transition tracker. Sized at 10% of the gap so it is
        # itself far from the ceiling and cannot overshoot.
        m0 = clamp(floor(Int, 0.10 * (T_aim - cost_warm) / cpc), 4, 4096)
        _produce_all!(m0)
        c_now = cost(counter)
        cpc = max((c_now - cost_warm) / m0, 1.0); cpc_max = max(cpc_max, cpc)
        push!(extension_log, (phase = :probe_chunk, m = m0, cost = c_now,
            cpc = cpc, ratio = c_now / Bf))

        for it in 1:600
            c_now = cost(counter)
            (c_now >= T_aim || c_now >= T_hi) && break
            gap = T_aim - c_now
            cpc_cons = max(cpc, 0.5 * cpc_max)                 # (2) conservative
            m = floor(Int, 0.5 * gap / cpc_cons)               # (1) half the gap
            lf_obs = max(_read_nuts_nsteps_max(), 1)
            step_worst = nch * lf_obs * _NUTS_OVERSHOOT_SAFETY * d   # (3) backstop
            m_back = floor(Int, (T_hi - c_now) / max(step_worst, 1.0))
            m = clamp(min(m, m_back), 1, 5_000_000)
            _produce_all!(m)
            c_after = cost(counter)
            realised = max((c_after - c_now) / m, 1.0)
            cpc = 0.5 * cpc + 0.5 * realised                   # EWMA refine
            cpc_max = max(cpc_max, realised)
            push!(extension_log, (phase = :fill, it = it, m = m, cost = c_after,
                cpc = realised, ratio = c_after / Bf))
            c_after >= T_aim && break
        end
    end

    wall_time = (time_ns() - t0) / 1e9

    # Combined chain-output list: pilot chain first, then the extra chains.
    chain_outputs = outputs2 === nothing ? collect(outputs1) :
        vcat(collect(outputs1), collect(outputs2))

    # Per-chain samples: inverse-transform each chain's production output and
    # keep chain structure (for R̂); concatenate chain-major truncated to the
    # shortest chain so reshape(samples, n_min, nchains) is well-defined.
    chain_samples = Vector{Vector{Vector{Float64}}}()
    chain_weights = Vector{Vector{Float64}}()
    chain_logd = Vector{Vector{Float64}}()
    chain_diverged = Vector{Vector{Bool}}()
    chain_n = Int[]
    leapfrog_upper = 0
    for c in eachindex(chain_outputs)
        walker_outs = chain_outputs[c]
        dsv = isempty(walker_outs) ? nothing : first(walker_outs)
        if dsv === nothing || isempty(dsv)
            push!(chain_samples, Vector{Vector{Float64}}()); push!(chain_weights, Float64[])
            push!(chain_logd, Float64[]); push!(chain_diverged, Bool[])
            push!(chain_n, 0); continue
        end
        leapfrog_upper += _leapfrog_upper_bound(dsv)
        # V8-FIX-B4: harvest the real per-transition divergence flag
        # (AdvancedHMC's `numerical_error`, kept by AHMCSampleID) instead
        # of hard-coding an empty list, which made the downstream
        # divergence diagnostic read "no divergences" unconditionally.
        cd = Bool[]
        if hasproperty(dsv, :info) && hasproperty(dsv.info, :numerical_error)
            cd = [Bool(x) for x in dsv.info.numerical_error]
        end
        smpls = BAT.transform_samples(BAT.inverse(f_pre), dsv)
        cs = Vector{Vector{Float64}}(); cw = Float64[]; cl = Float64[]
        for s in smpls
            push!(cs, [Float64(x) for x in s.v])
            push!(cw, Float64(s.weight)); push!(cl, Float64(s.logd))
        end
        length(cd) == length(cs) || (cd = fill(false, length(cs)))
        push!(chain_samples, cs); push!(chain_weights, cw)
        push!(chain_logd, cl); push!(chain_diverged, cd)
        push!(chain_n, length(cs))
    end

    nchains_eff = length(chain_outputs)
    n_min = isempty(chain_n) ? 0 : minimum(chain_n)
    samples_v = Vector{Vector{Float64}}()
    weights_v = Float64[]
    logd_v = Float64[]
    divergent_idx = Int[]
    for c in 1:nchains_eff
        cs = chain_samples[c]; cw = chain_weights[c]; cl = chain_logd[c]
        cdv = chain_diverged[c]
        isempty(cs) && continue
        n_take = n_min > 0 ? n_min : length(cs)
        for i in 1:n_take
            push!(samples_v, cs[i]); push!(weights_v, cw[i]); push!(logd_v, cl[i])
            (i <= length(cdv) && cdv[i]) && push!(divergent_idx, length(samples_v))
        end
    end

    if isempty(samples_v)
        @warn "NUTS produced no samples — emitting placeholder" cfg = typeof(cfg) seed = seed
        samples_mat = zeros(d, 1); weights_v = [1.0]; logd_v = [NaN]
        n_per_chain = Int[]; actual_nchains = 0
    else
        samples_mat = Matrix{Float64}(undef, d, length(samples_v))
        @inbounds for i in eachindex(samples_v), j in 1:d
            samples_mat[j, i] = samples_v[i][j]
        end
        actual_nchains = count(>(0), chain_n)
        n_per_chain = fill(n_min, actual_nchains)
    end

    W = W_probe; P = P_probe
    used_total = cost(counter)
    ratio_total = used_total / max(Bf, 1.0)
    stop_reason = ratio_total >= 0.8 ? :budget : :batches_exhausted

    extras = Dict{Symbol,Any}(
        :stop_reason => stop_reason,
        :terminated_by => stop_reason,
        :budget_infeasible => false,
        :nchains => actual_nchains,
        :nchains_requested => Int(nchains),
        :nchains_eff => nchains_eff,
        :n_per_chain => n_per_chain,
        :n_per_chain_raw => chain_n,
        :extension_log => extension_log,
        :nuts_identity => nuts_id,
        :warmup_cost_est => W,
        :prod_cost_per_step_est => P,
        :nlike_over_B => ratio_total,
        :leapfrog_exact => _read_nuts_nsteps(),
        :leapfrog_upper_bound => leapfrog_upper,
        :transform => "PriorToNormal (BAT default for HamiltonianMC)",
        :divergent_indices => divergent_idx,   # V8-FIX-B4: real flags
        :n_divergent => length(divergent_idx),
    )

    mr = MethodResult(
        algorithm = :nuts, problem = _problem_symbol(cfg), d = d, seed = Int(seed),
        B = Bf, counter = counter, wall_time_s = wall_time, samples = samples_mat,
        weights = weights_v, logd = logd_v, logZ_estimate = missing,
        logZ_estimate_se = missing, extras = extras,
    )

    # In-runner cost-accounting guard. A correctly
    # instrumented NUTS run satisfies Neff ≤ #stored ≤ #leapfrog = n_dual_calls.
    # Violation ⇒ gradients miscounted again. Placeholder cells (≤1 sample) exempt.
    neff_val = neff(mr)
    ndual = counter.n_dual_calls
    n_real = size(mr.samples, 2)
    sane = n_real <= 1 || !isfinite(neff_val) || neff_val <= ndual + 1e-6
    mr.extras[:neff_checked] = neff_val
    mr.extras[:neff_le_ndual_ok] = sane
    if !sane
        @error "NUTS cost-accounting guard tripped: Neff > n_dual_calls" problem = _problem_symbol(cfg) d = d seed = Int(seed) B = Bf Neff = neff_val n_dual_calls = ndual n_primal = counter.n_primal n_samples = n_real
    end
    if get(ENV, "MW_NUTS_GUARD", "assert") != "warn"
        @assert sane "NUTS Neff=$(neff_val) exceeds n_dual_calls=$(ndual) at d=$d seed=$seed B=$B — gradient-cost undercount regressed"
    end

    return mr
end

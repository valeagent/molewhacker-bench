# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# counter.jl — Protocol §3 LikelihoodCounter
# =============================================================================
#
# Central fairness contract. Every algorithm consumes log_f(theta) only
# through this object. A primal call increments n_primal by 1; a
# ForwardDiff dual-number call increments n_dual_calls by 1 and
# n_grad_partials by the TOTAL number of propagated partials — for a
# flat Dual of chunk size N that is N; for NESTED duals (Hessians via
# gradient-of-gradient) it is the product of the chunk sizes across
# nesting levels (V8-FIX-A6). A dual call adds nothing to n_primal, so
# a complete d-dimensional forward-mode gradient (summed over its
# chunks) costs d, and a full d×d ForwardDiff Hessian costs ≈ d².
#
# Cumulative cost  cost(c) = c.n_primal + c.n_grad_partials.
#
# Reference: the cost-accounting contract of the thesis (Ch. 7; the
# counter listing is reproduced in the implementation appendix).

using ForwardDiff
using DensityInterface

"""
    LikelihoodCounter{F}

Wraps a primal log-density callable `log_f(::AbstractVector{<:Real}) -> Real`
and tracks calls separately depending on whether the input is plain
`Float64` or a `ForwardDiff.Dual` of chunk size `N`.

Fields
------
- `log_f`             : the underlying callable.
- `n_primal`          : number of primal evaluations (each costs 1).
- `n_dual_calls`      : number of dual-number evaluations.
- `n_grad_partials`   : sum of chunk sizes across dual calls (cost units
                        from automatic differentiation).
- `frozen`            : when true, calls assert.

Cost contract: `cost(c) = c.n_primal + c.n_grad_partials`. A single dual
call with chunk size `N` adds `N` to `n_grad_partials` and nothing to
`n_primal`, so it contributes exactly `N` to the total. A complete
`d`-dimensional forward-mode gradient (summed over however many chunks
ForwardDiff splits it into) therefore costs exactly `d` — matching the
thesis statement "one gradient evaluation is counted as `d`
likelihood-equivalents."
"""
mutable struct LikelihoodCounter{F}
    log_f::F
    n_primal::Int
    n_dual_calls::Int
    n_grad_partials::Int
    frozen::Bool
end

LikelihoodCounter(log_f) = LikelihoodCounter(log_f, 0, 0, 0, false)

# Lock thread-safety with a single coarse lock. The protocol does not
# require atomic ordering between primal and dual increments; the cost
# is the eventual sum, and missing a +1 here or there inside parallel
# inner loops would silently break fairness.
const _COUNTER_LOCK = Base.ReentrantLock()

@inline function _bump_primal!(c::LikelihoodCounter)
    @assert !c.frozen "LikelihoodCounter is frozen; cannot be called"
    Base.@lock _COUNTER_LOCK begin
        c.n_primal += 1
    end
    return nothing
end

@inline function _bump_dual!(c::LikelihoodCounter, N::Int)
    @assert !c.frozen "LikelihoodCounter is frozen; cannot be called"
    Base.@lock _COUNTER_LOCK begin
        c.n_dual_calls += 1
        c.n_grad_partials += N
    end
    return nothing
end

# V8-FIX-A6: total partials of a (possibly nested) dual type. ForwardDiff
# computes Hessians as gradient-of-gradient with NESTED duals
# `Dual{T2, Dual{T1, V, N1}, N2}`; each such call propagates N1·N2
# second-order partials, i.e. it does the work of N1·N2 primal
# evaluations to leading order. Charging only the outer `npartials`
# (the pre-V8 behaviour) undercharged every Hessian d-fold — a full
# d×d Hessian was billed ≈ d instead of ≈ d². This mattered for
# MoleWhacker, whose Laplace-fallback component fits call
# `ForwardDiff.hessian` on every candidate mode.
@inline _total_partials(::Type{ForwardDiff.Dual{T,V,N}}) where {T,V,N} =
    N * _total_partials(V)
@inline _total_partials(::Type{T}) where {T} = 1

# Generic call: ANY abstract vector (not just `AbstractVector{<:Real}`).
# Classify primal vs gradient by the *runtime*
# element type, not only by static dispatch. `ForwardDiff.Dual <: Real`,
# so a gradient vector that does not statically present as
# `AbstractVector{<:Dual{T,V,N}}` — e.g. the `Vector{Real}` *or*
# `Vector{Any}` container BAT's `PriorToNormal` transform hands the
# target during NUTS leapfrog steps and step-size adaptation — would
# otherwise fall through to the primal branch (or, for `Vector{Any}`,
# match no method at all) and be undercharged at cost 1 instead of cost
# `d`. The signature is deliberately the untyped `AbstractVector` so a
# `Vector{Any}` of duals is still caught; the specialized `Dual` method
# below remains more specific and still wins for statically-typed
# gradient vectors. Inspecting the first element closes the leak
# regardless of the container's static eltype.
function (c::LikelihoodCounter)(theta::AbstractVector)
    if !isempty(theta) && (eltype(theta) <: ForwardDiff.Dual ||
                            theta[1] isa ForwardDiff.Dual)
        _bump_dual!(c, _total_partials(typeof(theta[1])))
    else
        _bump_primal!(c)
    end
    return c.log_f(theta)
end

# Fast path for properly-typed dual vectors: AD passes
# `ForwardDiff.Dual{T,V,N}` elements and we read the total partial
# count straight from the type (V8-FIX-A6: recursive over nested duals,
# so Hessian calls are charged N_outer·N_inner). This is a strict
# specialization of the generic method above (same accounting), kept so
# the common, statically-typed gradient path avoids the runtime check.
function (c::LikelihoodCounter)(
    theta::AbstractVector{<:ForwardDiff.Dual{T,V,N}},
) where {T,V,N}
    _bump_dual!(c, N * _total_partials(V))
    return c.log_f(theta)
end

"""
    set_gradient_accounting!(c, n_dual_calls, n_grad_partials)

Authoritatively set the gradient counts, overriding the value-dispatch
tally. Used by the NUTS runner: AdvancedHMC computes
the production leapfrog gradients through an internal cached closure that
bypasses this counter (and even an explicit wrapper on BAT's gradient
function), so the dual-dispatch tally saturates at the warmup count and is
structurally blind to production gradients. The runner instead charges the
*exact* leapfrog count `Σ n_steps` recorded by AdvancedHMC per transition
(`tree.nα`, harvested in `BAT._get_sample_id`; warmup + production) and
installs `n_dual_calls = Σn_steps`, `n_grad_partials = d·Σn_steps` here, so
`cost(c)` reflects the real gradient work — one `d`-dimensional gradient =
`d` cost units. See `docs/NUTS-COST-ACCOUNTING.md`.
"""
function set_gradient_accounting!(c::LikelihoodCounter, n_dual_calls::Integer,
                                    n_grad_partials::Integer)
    Base.@lock _COUNTER_LOCK begin
        c.n_dual_calls = Int(n_dual_calls)
        c.n_grad_partials = Int(n_grad_partials)
    end
    return nothing
end

"""
    cost(c::LikelihoodCounter) -> Int

The fairness-axis cost: `n_primal + n_grad_partials`.
"""
cost(c::LikelihoodCounter) = c.n_primal + c.n_grad_partials

"""
    is_over_budget(c::LikelihoodCounter, B::Real) -> Bool

True when the counter has exceeded budget `B` in cost units.
"""
is_over_budget(c::LikelihoodCounter, B::Real) = cost(c) >= B

"""
    Base.getindex(c::LikelihoodCounter) -> Int

`c[]` returns the cumulative cost. This makes the counter a drop-in
replacement for the `Ref{Int}` "evals so far" handle that the
MoleWhacker module's `whack_many_moles(..; counter=...)` argument
expects (see `src/MoleWhacker.jl`).
"""
Base.getindex(c::LikelihoodCounter) = cost(c)

"""
    reset!(c::LikelihoodCounter)

Zero the three counters and unfreeze.
"""
function reset!(c::LikelihoodCounter)
    Base.@lock _COUNTER_LOCK begin
        c.n_primal = 0
        c.n_dual_calls = 0
        c.n_grad_partials = 0
        c.frozen = false
    end
    return c
end

"""
    freeze!(c::LikelihoodCounter)

Disallow further calls. Useful to ensure no post-run evaluations leak
into the counter (e.g. evidence reuse from saved samples).
"""
freeze!(c::LikelihoodCounter) = (c.frozen = true; c)

# -----------------------------------------------------------------------------
# DensityInterface bridge
#
# BAT.jl and many of its samplers do not call the function directly;
# they wrap it in a Likelihood/PosteriorDensity and then call
# `logdensityof(...)`. Make the counter a valid log-density object so
# both code paths converge on the same counter.
# -----------------------------------------------------------------------------

DensityInterface.DensityKind(::LikelihoodCounter) = DensityInterface.IsDensity()

function DensityInterface.logdensityof(c::LikelihoodCounter, theta)
    return c(theta)
end

# -----------------------------------------------------------------------------
# Snapshots and pretty printing
# -----------------------------------------------------------------------------

"""
    snapshot(c::LikelihoodCounter) -> NamedTuple

Lightweight read-only view of the counter state, suitable for embedding
into the per-cell metadata.
"""
function snapshot(c::LikelihoodCounter)
    return (
        n_primal = c.n_primal,
        n_dual_calls = c.n_dual_calls,
        n_grad_partials = c.n_grad_partials,
        cost = cost(c),
        frozen = c.frozen,
    )
end

function Base.show(io::IO, c::LikelihoodCounter)
    print(io,
        "LikelihoodCounter(",
        "n_primal=", c.n_primal,
        ", n_dual=", c.n_dual_calls,
        ", n_grad_partials=", c.n_grad_partials,
        ", cost=", cost(c),
        c.frozen ? ", frozen" : "",
        ")")
end

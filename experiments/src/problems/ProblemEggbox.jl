# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# ProblemEggbox.jl — Mukherjee 2006 eggbox (Protocol §4.7.2)
# =============================================================================
#
# log f(θ) = 5 log(2 + ∏_{i=1}^d cos(θ_i / 2))
#
# on the physical domain `[0, 5π]^d`. Every other configuration in this
# code base reports `θ ∈ [-L, L]^d`; we therefore *shift* the eggbox
# centre to the box centre at `5π/2`, equivalently work in
# `θ ∈ [-L, L]^d` with
#
#   log f(θ) = 5 log(2 + ∏_{i=1}^d cos((θ_i + 5π/2) / 2)).
#
# The multiplicative shift by `5π/2` keeps the eggbox cosine pattern
# centred and makes the cube boundary behave the same as in every
# other problem.

const _EGGBOX_HALF_PERIOD = 5π / 2

"""
    ConfigEggbox(d, L)
"""
struct ConfigEggbox <: ProblemConfig
    d::Int
    L::Float64
end

"""
    make_config_eggbox(; d=2, L=10.0)

`L = 10.0` matches every other problem in the catalogue. The eggbox
centre is `5π / 2 ≈ 7.85`, so `L = 10` covers slightly more than one
period either side of the centre.
"""
function make_config_eggbox(; d::Integer = 2, L::Real = 10.0)
    @assert d >= 1 "ConfigEggbox: d must be ≥ 1"
    @assert L > 0 "ConfigEggbox: L must be positive"
    return ConfigEggbox(Int(d), Float64(L))
end


"""
    build_log_f(cfg::ConfigEggbox) -> Function

Returns the eggbox log-density on `[-L, L]^d`, with the centre
shifted to the box centre.
"""
function build_log_f(cfg::ConfigEggbox)
    d = cfg.d
    function log_f(θ::AbstractVector{<:Real})
        # `s` must start at `one(eltype(θ))` so that under ForwardDiff
        # it never has the Union{Float64,Dual} value-type that breaks
        # BAT.jl's `is_log_zero`.
        s = one(eltype(θ))
        @inbounds for i in 1:d
            s *= cos(0.5 * (θ[i] + _EGGBOX_HALF_PERIOD))
        end
        return 5.0 * log(2.0 + s)
    end
    return log_f
end

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# ProblemShell.jl — Gaussian shell (Protocol §4.5)
# =============================================================================
#
# A thin Gaussian-shell density: the level set of |θ| centred on the
# origin, with radial width w. The geometric factor depends only on
# the radial coordinate r = ‖θ‖:
#
#     log f(θ) = -½ ((r - ρ) / w)^2.
#
# Default parameters give a moderately thin shell: ρ = 4, w = 0.5,
# d = 5  ⇒  w/ρ = 0.125, an honest engineered failure case for
# mixture-based methods (Protocol §13.10).

"""
    ConfigShell(d, ρ, w, L)

Spherical Gaussian shell.

- `d::Int`     dimension (≥ 2)
- `ρ::Float64` shell radius
- `w::Float64` shell width
- `L::Float64` cube half-width (must satisfy L > ρ + several·w)
"""
struct ConfigShell <: ProblemConfig
    d::Int
    ρ::Float64
    w::Float64
    L::Float64
end

"""
    make_config_shell(; d=5, ρ=4.0, w=0.5, L=10.0) -> ConfigShell
"""
function make_config_shell(; d::Int = 5, ρ::Real = 4.0, w::Real = 0.5, L::Real = 10.0)
    @assert d >= 2 "ConfigShell: d must be ≥ 2"
    @assert ρ > 0 "ConfigShell: ρ must be positive"
    @assert w > 0 "ConfigShell: w must be positive"
    @assert L > ρ + 5w "ConfigShell: L should comfortably contain the shell (> ρ + 5w)"
    return ConfigShell(d, Float64(ρ), Float64(w), Float64(L))
end

"""
    build_log_f(cfg::ConfigShell) -> Function
"""
function build_log_f(cfg::ConfigShell)
    ρ = cfg.ρ
    w = cfg.w
    function log_f(theta::AbstractVector)
        r = norm(theta)
        return -0.5 * ((r - ρ) / w)^2
    end
    return log_f
end

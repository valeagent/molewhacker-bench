# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# truth_common.jl — TruthSet container, dispatcher, file I/O
# =============================================================================
#
# The truth set is the "specification" of a target: it is **never**
# consumed by a sampler, only by metric code in `metrics.jl`.

"""
    TruthSet

Per-(problem, d) reference data set:

- `problem::Symbol`               which problem
- `d::Int`                         dimension
- `cfg::ProblemConfig`             the canonical config used to build it
- `samples::Matrix{Float64}`       d × N_ref iid (or unbiased) draws from the
                                   target restricted to the cube
- `mean::Vector{Float64}`          E[θ_j]
- `cov::Matrix{Float64}`           Cov[θ_i, θ_j]
- `quantiles::Matrix{Float64}`     d × 5 matrix; columns are
                                   [p025, p160, p500, p840, p975]
- `logZ::Float64`                  log-evidence under the cube prior
- `extras::Dict{Symbol,Any}`       problem-specific tables (e.g. funnel CDF grid)
"""
struct TruthSet
    problem::Symbol
    d::Int
    cfg::ProblemConfig
    samples::Matrix{Float64}
    mean::Vector{Float64}
    cov::Matrix{Float64}
    quantiles::Matrix{Float64}
    logZ::Float64
    extras::Dict{Symbol,Any}
end

const QUANTILE_LEVELS = (0.025, 0.16, 0.5, 0.84, 0.975)

"""
    quantile_idx_of(p) -> Int

Index of `p` inside `QUANTILE_LEVELS`. Used by `metrics.jl::quantile_errors`.
"""
function quantile_idx_of(p::Real)
    for (i, q) in enumerate(QUANTILE_LEVELS)
        if isapprox(p, q; atol = 1e-12)
            return i
        end
    end
    error("quantile_idx_of: $p not in QUANTILE_LEVELS")
end


# -----------------------------------------------------------------------------
# Generic sample-based moment/quantile estimators (used as a fallback when
# the truth has no closed form for a given column).
# -----------------------------------------------------------------------------

function _empirical_moments(samples::AbstractMatrix{<:Real})
    d, N = size(samples)
    μ = vec(mean(samples; dims = 2))
    centred = samples .- μ
    Σ = (centred * centred') ./ (N - 1)
    return μ, Σ
end

function _empirical_quantiles(samples::AbstractMatrix{<:Real})
    d, N = size(samples)
    Q = Matrix{Float64}(undef, d, 5)
    for j in 1:d
        col = sort(view(samples, j, :))
        for (k, p) in enumerate(QUANTILE_LEVELS)
            r = max(1, min(N, ceil(Int, p * N)))
            Q[j, k] = col[r]
        end
    end
    return Q
end


# -----------------------------------------------------------------------------
# Top-level dispatcher
# -----------------------------------------------------------------------------

"""
    compute_truth(cfg::ProblemConfig; N_ref=1_000_000, seed=20260501) -> TruthSet

Produces the canonical truth artefact for a given (problem, d). The
heavy work — analytical moments, 1D quadrature for `logZ`, and i.i.d.
truth-sample generation — is delegated to per-problem helpers.
"""
function compute_truth(cfg::ProblemConfig; N_ref::Integer = 1_000_000,
                                              seed::Integer = 20260501)
    if cfg isa ConfigMVN
        return compute_truth_mvn(cfg; N_ref = N_ref, seed = seed)
    elseif cfg isa ConfigBanana
        return compute_truth_banana(cfg; N_ref = N_ref, seed = seed)
    elseif cfg isa ConfigFunnel
        return compute_truth_funnel(cfg; N_ref = N_ref, seed = seed)
    elseif cfg isa ConfigMRidges
        return compute_truth_mridges(cfg; N_ref = N_ref, seed = seed)
    elseif cfg isa ConfigShell
        return compute_truth_shell(cfg; N_ref = N_ref, seed = seed)
    elseif cfg isa ConfigMRidgesSpiky
        return compute_truth_mridges_spiky(cfg; N_ref = N_ref, seed = seed)
    elseif cfg isa ConfigEggbox
        return compute_truth_eggbox(cfg; N_ref = N_ref, seed = seed)
    else
        error("compute_truth: unknown config type $(typeof(cfg))")
    end
end

"""
    sample_truth(cfg::ProblemConfig, N::Integer; seed) -> Matrix{Float64}

Draws N i.i.d. samples from the target restricted to the cube.
"""
function sample_truth(cfg::ProblemConfig, N::Integer; seed::Integer = 20260601)
    if cfg isa ConfigMVN
        return sample_truth_mvn(cfg, N; seed = seed)
    elseif cfg isa ConfigBanana
        return sample_truth_banana(cfg, N; seed = seed)
    elseif cfg isa ConfigFunnel
        return sample_truth_funnel(cfg, N; seed = seed)
    elseif cfg isa ConfigMRidges
        return sample_truth_mridges(cfg, N; seed = seed)
    elseif cfg isa ConfigShell
        return sample_truth_shell(cfg, N; seed = seed)
    elseif cfg isa ConfigMRidgesSpiky
        return sample_truth_mridges_spiky(cfg, N; seed = seed)
    elseif cfg isa ConfigEggbox
        return sample_truth_eggbox(cfg, N; seed = seed)
    else
        error("sample_truth: unknown config type $(typeof(cfg))")
    end
end


# -----------------------------------------------------------------------------
# Truth file I/O — one file per (problem, d), JLD2-binary
# -----------------------------------------------------------------------------

"""
    save_truth(path, truth::TruthSet)

Writes the truth to `path` (JLD2). The samples matrix dominates the
file size; for `N_ref = 1e6, d = 5` this is ~40 MB.
"""
function save_truth(path::AbstractString, truth::TruthSet)
    isdir(dirname(path)) || mkpath(dirname(path))
    JLD2.jldopen(path, "w"; iotype = IOStream) do f
        f["problem"] = String(truth.problem)
        f["d"] = truth.d
        f["cfg_type"] = string(typeof(truth.cfg))
        f["cfg_fields"] = _serialise_cfg(truth.cfg)
        f["samples"] = truth.samples
        f["mean"] = truth.mean
        f["cov"] = truth.cov
        f["quantiles"] = truth.quantiles
        f["logZ"] = truth.logZ
        f["extras"] = _serialise_truth_extras(truth.extras)
    end
    return path
end

"""
    load_truth(path) -> TruthSet
"""
function load_truth(path::AbstractString)
    return JLD2.jldopen(path, "r") do f
        problem = Symbol(f["problem"])
        d = Int(f["d"])
        cfg_type = f["cfg_type"]
        cfg_fields = f["cfg_fields"]
        cfg = _deserialise_cfg(cfg_type, cfg_fields)
        samples = Matrix{Float64}(f["samples"])
        μ = Vector{Float64}(f["mean"])
        Σ = Matrix{Float64}(f["cov"])
        Q = Matrix{Float64}(f["quantiles"])
        logZ = Float64(f["logZ"])
        extras = _deserialise_truth_extras(f["extras"])
        return TruthSet(problem, d, cfg, samples, μ, Σ, Q, logZ, extras)
    end
end

function _serialise_cfg(cfg::ProblemConfig)
    out = Dict{String,Any}()
    for fld in fieldnames(typeof(cfg))
        v = getfield(cfg, fld)
        out[String(fld)] = v isa AbstractVector ? Vector(v) : v
    end
    return out
end

function _deserialise_cfg(cfg_type::AbstractString, fields::AbstractDict)
    if endswith(cfg_type, "ConfigMVN")
        return ConfigMVN(Int(fields["d"]), Float64(fields["sigma"]),
                          Float64(fields["rho"]),
                          Vector{Float64}(fields["mu"]), Float64(fields["L"]))
    elseif endswith(cfg_type, "ConfigBanana")
        return ConfigBanana(Int(fields["d"]),
                              Vector{Float64}(fields["sigma"]),
                              Float64(fields["b"]),
                              Float64(fields["L"]))
    elseif endswith(cfg_type, "ConfigFunnel")
        return ConfigFunnel(Int(fields["d"]),
                              Float64(fields["sigma1"]),
                              Float64(fields["L"]))
    elseif endswith(cfg_type, "ConfigMRidges")
        return ConfigMRidges(Int(fields["d"]),
                              Float64(fields["Δ"]),
                              Float64(fields["τ"]),
                              Float64(fields["a"]),
                              Float64(fields["b"]),
                              Float64(fields["c"]),
                              Float64(fields["s0"]),
                              Float64(fields["L"]))
    elseif endswith(cfg_type, "ConfigShell")
        return ConfigShell(Int(fields["d"]),
                            Float64(fields["ρ"]),
                            Float64(fields["w"]),
                            Float64(fields["L"]))
    elseif endswith(cfg_type, "ConfigMRidgesSpiky")
        return ConfigMRidgesSpiky(Int(fields["d"]),
                                    Int(fields["K"]), Int(fields["M"]),
                                    Float64(fields["σ_spike"]),
                                    Float64(fields["σ_perp"]),
                                    Float64(fields["σ_jitter"]),
                                    Float64(fields["Δ_K"]),
                                    Symbol(fields["kernel"]),
                                    Float64(fields["ν"]),
                                    Float64(fields["L"]))
    elseif endswith(cfg_type, "ConfigEggbox")
        return ConfigEggbox(Int(fields["d"]), Float64(fields["L"]))
    else
        error("_deserialise_cfg: unknown cfg type $cfg_type")
    end
end

function _serialise_truth_extras(d::AbstractDict)
    out = Dict{String,Any}()
    for (k, v) in d
        out[String(k)] = v
    end
    return out
end

function _deserialise_truth_extras(raw)
    out = Dict{Symbol,Any}()
    if raw isa AbstractDict
        for (k, v) in raw
            sym = k isa Symbol ? k : Symbol(k)
            out[sym] = v
        end
    end
    return out
end

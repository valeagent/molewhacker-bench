# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# base.jl — shared types and utilities (after counter.jl, before problems)
# =============================================================================

# -----------------------------------------------------------------------------
# Run-sheet constants from §7 of the protocol
# -----------------------------------------------------------------------------

"""
    HEADLINE_DIMENSION

The single dimension at which the full 5×5 cross-product runs.
"""
const HEADLINE_DIMENSION = 5

"""
    BUDGET_GRID

The three budgets B in cost units (counter.cost). Protocol §1, §7.
"""
const BUDGET_GRID = (5e3, 5e4, 5e5)

"""
    SEED_GRID

Twenty seeds fixed for reproducibility. Protocol §7, §13.3 (rev May
2026, V3-S-1: count raised from 5 to 20 to give the paired Wilcoxon
test ≥ 80 % power against `δ = 0.4`).
"""
const SEED_GRID = (11, 23, 41, 67, 97,
                    113, 137, 173, 211, 239,
                    277, 313, 359, 401, 449,
                    491, 541, 587, 631, 677)

"""
    SEED_GRID_V2 — the original five seeds; preserved as a literal subset
to make v2 deliverables a free regression test inside the v3 grid.
"""
const SEED_GRID_V2 = (11, 23, 41, 67, 97)

"""
    PROBLEM_NAMES

Benchmark problems in canonical order: the five baseline targets plus
the two stress targets (`mridges_spiky`, `eggbox`) of the thesis
benchmark suite. An early real-world problem placeholder was dropped
from the design; the thesis does not reference it, so the symbol is
absent from every label dictionary and figure-iteration order.
"""
const PROBLEM_NAMES = (:mvn, :banana, :funnel, :mridges, :shell,
                        :mridges_spiky, :eggbox)

"""
    PROBLEM_NAMES_V2 — the original five problems for backward compatibility.
"""
const PROBLEM_NAMES_V2 = (:mvn, :banana, :funnel, :mridges, :shell)

"""
    ALG_NAMES

The five sampling algorithms in canonical order. Protocol §5.
"""
const ALG_NAMES = (:is, :mh, :nuts, :ns, :mw)

"""
    SCALING_PROBLEMS, SCALING_DIMENSIONS

The scaling subset. The protocol originally swept
`d ∈ {2, 5, 10, 20}` over every problem; the `d = 20` sweep was
dropped from the thesis (June 2026 protocol revision), so the
dimension-scaling study is `d ∈ {2, 5, 10}` for all seven problems
(the headline stays at `d = 5`). This makes every scaling figure square
(no ragged `d`-columns, no N/A) and removes the small-budget NUTS
infeasibility band that `d = 20` introduced. The legacy
`(funnel, shell) × {2, 10}` minimal sweep is preserved as
`SCALING_PROBLEMS_V2 / SCALING_DIMENSIONS_V2`.
"""
const SCALING_PROBLEMS = (:mvn, :banana, :funnel, :mridges, :shell,
                           :mridges_spiky, :eggbox)
const SCALING_DIMENSIONS = (2, 5, 10)
const SCALING_PROBLEMS_V2 = (:funnel, :shell)
const SCALING_DIMENSIONS_V2 = (2, 10)


# -----------------------------------------------------------------------------
# MethodResult — uniform result struct (Protocol §5)
# -----------------------------------------------------------------------------

"""
    MethodResult

Uniform per-cell output struct. Every algorithm produces one of these.

Fields (Protocol §5):
- `algorithm::Symbol`           one of `:is, :mh, :nuts, :ns, :mw`
- `problem::Symbol`             one of `:mvn, :banana, :funnel, :mridges, :shell`
- `d::Int`                      working dimensionality
- `seed::Int`                   the cell seed
- `B::Float64`                  configured budget in cost units
- `Nlike_used::Float64`         actual `cost(counter)` at termination
- `n_primal::Int`               primal-evaluation count
- `n_grad_partials::Int`        partials count (cost = n_primal + n_grad_partials)
- `wall_time_s::Float64`        wall clock seconds
- `samples::Matrix{Float64}`    `d × N_out`; equal-weight if `weights` are uniform
- `weights::Vector{Float64}`    length-`N_out` unnormalised weights (1.0 ⇒ equal-weight)
- `logd::Vector{Float64}`       `log f(samples[:, i])` (no prior, no Z)
- `logZ_estimate::Union{Float64,Missing}`     evidence estimate, when applicable
- `logZ_estimate_se::Union{Float64,Missing}`  Skilling/SNIS/jackknife SE
- `extras::Dict{Symbol,Any}`    algorithm-specific fields (see §5)
"""
struct MethodResult
    algorithm::Symbol
    problem::Symbol
    d::Int
    seed::Int
    B::Float64
    Nlike_used::Float64
    n_primal::Int
    n_grad_partials::Int
    wall_time_s::Float64
    samples::Matrix{Float64}
    weights::Vector{Float64}
    logd::Vector{Float64}
    logZ_estimate::Union{Float64,Missing}
    logZ_estimate_se::Union{Float64,Missing}
    extras::Dict{Symbol,Any}
end

function MethodResult(;
        algorithm::Symbol,
        problem::Symbol,
        d::Int,
        seed::Int,
        B::Real,
        counter::LikelihoodCounter,
        wall_time_s::Real,
        samples::AbstractMatrix,
        weights::AbstractVector{<:Real} = Float64[],
        logd::AbstractVector{<:Real} = Float64[],
        logZ_estimate = missing,
        logZ_estimate_se = missing,
        extras::AbstractDict = Dict{Symbol,Any}(),
    )
    N = size(samples, 2)
    @assert size(samples, 1) == d "MethodResult: samples must be d × N (got size=$(size(samples)))"
    w = isempty(weights) ? fill(1.0, N) : Vector{Float64}(weights)
    @assert length(w) == N "MethodResult: weights length must match #samples"
    ld = isempty(logd) ? fill(NaN, N) : Vector{Float64}(logd)
    @assert length(ld) == N "MethodResult: logd length must match #samples"
    # Expose the dual-call count in extras. It is the
    # number of ForwardDiff gradient evaluations seen by the counter, and
    # it underpins the V6 sanity gate `Neff ≤ n_dual_calls`. The counter
    # is the single source of truth, so we always overwrite any pre-set
    # value a runner may have placed in `extras`.
    extras_out = Dict{Symbol,Any}(extras)
    extras_out[:n_dual_calls] = Int(counter.n_dual_calls)
    return MethodResult(
        algorithm,
        problem,
        Int(d),
        Int(seed),
        Float64(B),
        Float64(cost(counter)),
        Int(counter.n_primal),
        Int(counter.n_grad_partials),
        Float64(wall_time_s),
        Matrix{Float64}(samples),
        w,
        ld,
        logZ_estimate === nothing ? missing : logZ_estimate,
        logZ_estimate_se === nothing ? missing : logZ_estimate_se,
        extras_out,
    )
end

n_samples(mr::MethodResult) = size(mr.samples, 2)


# -----------------------------------------------------------------------------
# Numerical utilities
# -----------------------------------------------------------------------------

"""
    logsumexp(x)

Numerically stable `log(sum(exp.(x)))`. Returns `-Inf` for empty inputs.
"""
function logsumexp(x::AbstractVector{<:Real})
    isempty(x) && return -Inf
    xmax = maximum(x)
    isinf(xmax) && return Float64(xmax)
    s = 0.0
    @inbounds for xi in x
        s += exp(xi - xmax)
    end
    return Float64(xmax) + log(s)
end


"""
    weighted_quantile(x, w, p) -> Float64

Weighted `p`-quantile of vector `x` with non-negative weights `w`.
Implements the inverse-CDF estimator using sorted-data weighted-CDF.
"""
function weighted_quantile(x::AbstractVector{<:Real}, w::AbstractVector{<:Real}, p::Real)
    @assert length(x) == length(w) "weighted_quantile: x, w length mismatch"
    @assert 0.0 <= p <= 1.0 "weighted_quantile: p must be in [0, 1]"
    n = length(x)
    n == 0 && return NaN
    perm = sortperm(x)
    xs = collect(x[perm])
    ws = collect(w[perm])
    W = sum(ws)
    W <= 0 && return NaN
    target = p * W
    csum = 0.0
    @inbounds for i in 1:n
        csum += ws[i]
        if csum >= target
            return Float64(xs[i])
        end
    end
    return Float64(xs[end])
end


"""
    resample_to_equal_weight(samples, weights, N_out; rng) -> Matrix

Stratified-multinomial resampling of weighted samples to `N_out`
equal-weight draws.
"""
function resample_to_equal_weight(
        samples::AbstractMatrix{<:Real},
        weights::AbstractVector{<:Real},
        N_out::Integer;
        rng::AbstractRNG = Random.default_rng(),
    )
    N = size(samples, 2)
    @assert length(weights) == N "resample: weights length mismatch"
    @assert N_out > 0
    if all(weights .== weights[1])
        idx = rand(rng, 1:N, Int(N_out))
    else
        wn = weights ./ sum(weights)
        idx = StatsBase.wsample(rng, 1:N, wn, Int(N_out); replace = true)
    end
    return samples[:, idx]
end


# -----------------------------------------------------------------------------
# Provenance / metadata helpers
# -----------------------------------------------------------------------------

const _SCHEMA_VERSION = 1

function _try_git_sha(; default = "unknown")
    try
        return read(`git rev-parse HEAD`, String) |> strip
    catch
        return default
    end
end

function _try_julia_version()
    return string(VERSION)
end

function _hostname()
    try
        return Sys.gethostname()
    catch
        return "unknown"
    end
end

"""
    provenance_string(; cells_csv_path=nothing, truth_path=nothing)

A single line embeddable in a PDF's `Subject` field. Captures the Git
SHA, Julia version, the CSV/truth digests when available, and an
ISO 8601 timestamp.
"""
function provenance_string(; cells_csv_path::Union{Nothing,AbstractString} = nothing,
                              truth_path::Union{Nothing,AbstractString} = nothing)
    sha = _try_git_sha()
    jl = _try_julia_version()
    cs_hash = (cells_csv_path !== nothing && isfile(cells_csv_path)) ?
        bytes2hex(SHA.sha256(read(cells_csv_path))) : "n/a"
    tr_hash = (truth_path !== nothing && isfile(truth_path)) ?
        bytes2hex(SHA.sha256(read(truth_path))) : "n/a"
    ts = Dates_iso8601()
    return string(
        "git_sha=", sha,
        "  julia=", jl,
        "  cells.csv_sha256=", cs_hash,
        "  truth_sha256=", tr_hash,
        "  rendered_at=", ts,
    )
end

# Lightweight ISO 8601 without depending on Dates being already loaded
function Dates_iso8601()
    t = time()
    secs = floor(Int, t)
    frac_ms = round(Int, (t - secs) * 1000)
    # gmtime
    tm = Libc.TmStruct(secs)
    return @sprintf("%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
        1900 + tm.year, tm.month + 1, tm.mday,
        tm.hour, tm.min, tm.sec, frac_ms)
end


# -----------------------------------------------------------------------------
# Cell directory and naming
# -----------------------------------------------------------------------------

"""
    cell_dir(out_root, problem, alg, d, B, seed) -> String

Canonical directory for one (problem, algorithm, d, B, seed) cell.
"""
function cell_dir(out_root::AbstractString, problem::Symbol, alg::Symbol,
                  d::Integer, B::Real, seed::Integer)
    return joinpath(out_root, "runs",
        @sprintf("%s_%s_d%d_B%s_seed%d", String(problem), String(alg),
                 Int(d), _budget_token(B), Int(seed)))
end

"""
    _budget_token(B) -> String

Render a budget like `5e3, 5e4, 5e5` into the protocol filename token.
"""
function _budget_token(B::Real)
    if B == 5e3
        return "5e3"
    elseif B == 5e4
        return "5e4"
    elseif B == 5e5
        return "5e5"
    else
        return @sprintf("%g", B)
    end
end


# -----------------------------------------------------------------------------
# Per-cell I/O
# -----------------------------------------------------------------------------

"""
    save_method_result(dir, mr) → "result.h5"

Write a `MethodResult` to `<dir>/result.h5` using JLD2 (HDF5-compatible
binary). Schema: each top-level group is one field of `MethodResult`.
"""
function save_method_result(dir::AbstractString, mr::MethodResult)
    isdir(dir) || mkpath(dir)
    path = joinpath(dir, "result.h5")
    extras_kv = _serialise_extras(mr.extras)
    JLD2.jldopen(path, "w"; iotype = IOStream) do f
        f["algorithm"] = String(mr.algorithm)
        f["problem"] = String(mr.problem)
        f["d"] = mr.d
        f["seed"] = mr.seed
        f["B"] = mr.B
        f["Nlike_used"] = mr.Nlike_used
        f["n_primal"] = mr.n_primal
        f["n_grad_partials"] = mr.n_grad_partials
        f["wall_time_s"] = mr.wall_time_s
        f["samples"] = mr.samples
        f["weights"] = mr.weights
        f["logd"] = mr.logd
        f["logZ_estimate"] = mr.logZ_estimate === missing ? NaN : mr.logZ_estimate
        f["logZ_estimate_se"] = mr.logZ_estimate_se === missing ? NaN : mr.logZ_estimate_se
        f["extras"] = extras_kv
    end
    return path
end

"""
    load_method_result(dir) -> MethodResult

Read back a `MethodResult` from `<dir>/result.h5`.
"""
function load_method_result(dir::AbstractString)
    path = isfile(dir) ? dir : joinpath(dir, "result.h5")
    return JLD2.jldopen(path, "r") do f
        algo = Symbol(f["algorithm"])
        prob = Symbol(f["problem"])
        d = Int(f["d"])
        seed = Int(f["seed"])
        B = Float64(f["B"])
        Nl = Float64(f["Nlike_used"])
        np = Int(f["n_primal"])
        ng = Int(f["n_grad_partials"])
        wt = Float64(f["wall_time_s"])
        samples = Matrix{Float64}(f["samples"])
        weights = Vector{Float64}(f["weights"])
        logd = Vector{Float64}(f["logd"])
        lz_raw = f["logZ_estimate"]
        lzse_raw = f["logZ_estimate_se"]
        lz = (lz_raw isa Real && isnan(lz_raw)) ? missing : Float64(lz_raw)
        lzse = (lzse_raw isa Real && isnan(lzse_raw)) ? missing : Float64(lzse_raw)
        extras_raw = f["extras"]
        extras = _deserialise_extras(extras_raw)
        return MethodResult(algo, prob, d, seed, B, Nl, np, ng, wt,
                            samples, weights, logd, lz, lzse, extras)
    end
end

"""
    _serialise_extras(d::Dict{Symbol,Any}) -> Dict{String,Any}

JLD2-friendly dictionary of the extras (Symbols, distributions, and
nested namedtuples are flattened to strings/arrays).
"""
function _serialise_extras(d::AbstractDict)
    out = Dict{String,Any}()
    for (k, v) in d
        out[String(k)] = _flatten_for_jld(v)
    end
    return out
end

_flatten_for_jld(v::Symbol) = String(v)
_flatten_for_jld(v::Number) = v
_flatten_for_jld(v::AbstractString) = String(v)
_flatten_for_jld(v::AbstractVector) = Vector(v)
_flatten_for_jld(v::AbstractMatrix) = Matrix(v)
_flatten_for_jld(v::Tuple) = collect(v)
_flatten_for_jld(v::NamedTuple) = Dict(String(k) => _flatten_for_jld(v[k]) for k in keys(v))
_flatten_for_jld(v::AbstractDict) = Dict(String(k) => _flatten_for_jld(val) for (k, val) in v)
_flatten_for_jld(v) = v   # fall through; JLD2 may still serialize

function _deserialise_extras(raw)
    out = Dict{Symbol,Any}()
    if raw isa AbstractDict
        for (k, v) in raw
            sym = k isa Symbol ? k : Symbol(k)
            out[sym] = v
        end
    end
    return out
end


# -----------------------------------------------------------------------------
# Metadata JSON (per-cell + per-figure)
#
# The protocol prescribes JSON; rather than depending on JSON3, we
# write a tiny encoder/decoder for the dict-of-(string|number|bool|null|
# array|dict) types we actually need.
# -----------------------------------------------------------------------------

function _json_encode(io::IO, x)
    if x === nothing
        print(io, "null")
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Real
        if isnan(x) || isinf(x)
            print(io, "null")
        elseif x isa Integer
            print(io, x)
        else
            print(io, @sprintf("%.17g", x))
        end
    elseif x isa AbstractString
        print(io, '"')
        for c in x
            if c == '"' || c == '\\'
                print(io, '\\', c)
            elseif c == '\n'
                print(io, "\\n")
            elseif c == '\r'
                print(io, "\\r")
            elseif c == '\t'
                print(io, "\\t")
            elseif UInt32(c) < 0x20
                print(io, @sprintf("\\u%04x", UInt32(c)))
            else
                print(io, c)
            end
        end
        print(io, '"')
    elseif x isa Symbol
        _json_encode(io, String(x))
    elseif x isa AbstractDict
        print(io, '{')
        first = true
        for k in sort(collect(keys(x)); by = string)
            if !first
                print(io, ',')
            end
            first = false
            _json_encode(io, string(k))
            print(io, ':')
            _json_encode(io, x[k])
        end
        print(io, '}')
    elseif x isa NamedTuple
        d = Dict{String,Any}(string(k) => x[k] for k in keys(x))
        _json_encode(io, d)
    elseif x isa Tuple
        _json_encode(io, collect(x))
    elseif x isa AbstractVector
        print(io, '[')
        for (i, v) in enumerate(x)
            if i > 1
                print(io, ',')
            end
            _json_encode(io, v)
        end
        print(io, ']')
    else
        _json_encode(io, string(x))
    end
end

"""
    write_metadata_json(dir, meta::AbstractDict)

Writes a `metadata.json` next to `result.h5`.
"""
function write_metadata_json(dir::AbstractString, meta::AbstractDict)
    isdir(dir) || mkpath(dir)
    path = joinpath(dir, "metadata.json")
    open(path, "w") do io
        _json_encode(io, meta)
    end
    return path
end

"""
    read_metadata_json(dir) -> Dict{String,Any}

Light-weight reader. Tolerant: returns `Dict()` if missing or malformed.
"""
function read_metadata_json(dir::AbstractString)
    path = isfile(dir) ? dir : joinpath(dir, "metadata.json")
    isfile(path) || return Dict{String,Any}()
    text = read(path, String)
    return _json_decode(text)
end

# Minimal JSON decoder, only what we need (objects, arrays, numbers,
# strings, true/false/null). Robust enough for our own writer's output.
function _json_decode(s::AbstractString)
    pos = Ref(1)
    text = s
    return _json_parse_value(text, pos)
end

function _json_skip_ws(s, pos::Ref{Int})
    while pos[] <= lastindex(s)
        c = s[pos[]]
        if c == ' ' || c == '\t' || c == '\n' || c == '\r'
            pos[] = nextind(s, pos[])
        else
            break
        end
    end
end

function _json_parse_value(s, pos::Ref{Int})
    _json_skip_ws(s, pos)
    pos[] > lastindex(s) && return nothing
    c = s[pos[]]
    if c == '{'
        return _json_parse_object(s, pos)
    elseif c == '['
        return _json_parse_array(s, pos)
    elseif c == '"'
        return _json_parse_string(s, pos)
    elseif c == 't' || c == 'f'
        return _json_parse_bool(s, pos)
    elseif c == 'n'
        return _json_parse_null(s, pos)
    else
        return _json_parse_number(s, pos)
    end
end

function _json_parse_object(s, pos::Ref{Int})
    out = Dict{String,Any}()
    pos[] = nextind(s, pos[])    # consume '{'
    _json_skip_ws(s, pos)
    if pos[] <= lastindex(s) && s[pos[]] == '}'
        pos[] = nextind(s, pos[])
        return out
    end
    while true
        _json_skip_ws(s, pos)
        key = _json_parse_string(s, pos)
        _json_skip_ws(s, pos)
        pos[] = nextind(s, pos[])    # consume ':'
        out[key] = _json_parse_value(s, pos)
        _json_skip_ws(s, pos)
        c = s[pos[]]
        pos[] = nextind(s, pos[])
        if c == '}'
            break
        end
    end
    return out
end

function _json_parse_array(s, pos::Ref{Int})
    out = Any[]
    pos[] = nextind(s, pos[])
    _json_skip_ws(s, pos)
    if pos[] <= lastindex(s) && s[pos[]] == ']'
        pos[] = nextind(s, pos[])
        return out
    end
    while true
        push!(out, _json_parse_value(s, pos))
        _json_skip_ws(s, pos)
        c = s[pos[]]
        pos[] = nextind(s, pos[])
        if c == ']'
            break
        end
    end
    return out
end

function _json_parse_string(s, pos::Ref{Int})
    pos[] = nextind(s, pos[])    # consume opening quote
    buf = IOBuffer()
    while true
        c = s[pos[]]
        if c == '"'
            pos[] = nextind(s, pos[])
            return String(take!(buf))
        elseif c == '\\'
            pos[] = nextind(s, pos[])
            esc = s[pos[]]
            pos[] = nextind(s, pos[])
            if esc == 'n'
                print(buf, '\n')
            elseif esc == 't'
                print(buf, '\t')
            elseif esc == 'r'
                print(buf, '\r')
            elseif esc == '"'
                print(buf, '"')
            elseif esc == '\\'
                print(buf, '\\')
            elseif esc == 'u'
                # 4 hex digits
                hex = s[pos[]:pos[]+3]
                print(buf, Char(parse(UInt32, hex; base = 16)))
                pos[] += 4
            else
                print(buf, esc)
            end
        else
            print(buf, c)
            pos[] = nextind(s, pos[])
        end
    end
end

function _json_parse_bool(s, pos::Ref{Int})
    if startswith(SubString(s, pos[]), "true")
        pos[] += 4
        return true
    else
        pos[] += 5
        return false
    end
end

function _json_parse_null(s, pos::Ref{Int})
    pos[] += 4
    return nothing
end

function _json_parse_number(s, pos::Ref{Int})
    start = pos[]
    while pos[] <= lastindex(s)
        c = s[pos[]]
        if c == ',' || c == ']' || c == '}' || c == ' ' || c == '\n' || c == '\t' || c == '\r'
            break
        end
        pos[] = nextind(s, pos[])
    end
    txt = SubString(s, start, prevind(s, pos[]))
    return tryparse(Int, txt) !== nothing ? parse(Int, txt) : parse(Float64, txt)
end

"""
    cell_metadata(; problem, algorithm, d, seed, B, started_utc, finished_utc, extra)

Builds the canonical per-cell metadata dict (Protocol §8).
"""
function cell_metadata(;
        problem::Symbol,
        algorithm::Symbol,
        d::Integer,
        seed::Integer,
        B::Real,
        started_utc::AbstractString,
        finished_utc::AbstractString,
        problem_config::AbstractDict = Dict{String,Any}(),
        algorithm_tuning::AbstractDict = Dict{String,Any}(),
        n_threads::Integer = Threads.nthreads(),
        notes::AbstractString = "",
    )
    return Dict{String,Any}(
        "schema_version" => _SCHEMA_VERSION,
        "git_sha" => _try_git_sha(),
        "julia_version" => _try_julia_version(),
        "machine" => Dict{String,Any}(
            "hostname" => _hostname(),
            "n_cpu" => Sys.CPU_THREADS,
            "n_threads_used" => Int(n_threads),
        ),
        "problem" => Dict{String,Any}(
            "name" => String(problem),
            "d" => Int(d),
            "config" => problem_config,
        ),
        "algorithm" => Dict{String,Any}(
            "name" => String(algorithm),
            "tuning" => algorithm_tuning,
        ),
        "budget_B" => Float64(B),
        "seed" => Int(seed),
        "started_utc" => started_utc,
        "finished_utc" => finished_utc,
        "notes" => notes,
    )
end

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# 01_run_cell.jl — run one (algorithm, problem, d, B, seed) cell
# =============================================================================
#
# Usage:
#   julia --project=. -t auto experiments/scripts/01_run_cell.jl \
#       --alg mw --problem mridges --d 5 --B 5e5 --seed 23 \
#       [--out experiments/out] [--force] [--no-metrics]
#
# Writes:
#   <out>/runs/<problem>_<alg>_d<d>_B<token>_seed<seed>/result.h5
#   <out>/runs/<problem>_<alg>_d<d>_B<token>_seed<seed>/metadata.json

using Pkg

const _ROOT = abspath(joinpath(@__DIR__, "..", ".."))
if Base.active_project() != joinpath(_ROOT, "Project.toml")
    Pkg.activate(_ROOT)
end

include(joinpath(_ROOT, "experiments", "src", "ExperimentsBase.jl"))
using .ExperimentsBase

using Printf
using Dates


# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------

function _parse_args(args::Vector{String})
    opts = Dict{String,Any}(
        "alg" => nothing, "problem" => nothing, "d" => HEADLINE_DIMENSION,
        "B" => 5e3, "seed" => 11,
        "out" => joinpath(_ROOT, "experiments", "out"),
        "truth_dir" => joinpath(_ROOT, "experiments", "out", "truth"),
        "force" => false, "no_metrics" => false,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--alg";       opts["alg"] = Symbol(args[i+1]); i += 2
        elseif a == "--problem"; opts["problem"] = Symbol(args[i+1]); i += 2
        elseif a == "--d";       opts["d"] = parse(Int, args[i+1]); i += 2
        elseif a == "--B";       opts["B"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--seed";    opts["seed"] = parse(Int, args[i+1]); i += 2
        elseif a == "--out";     opts["out"] = args[i+1]; i += 2
        elseif a == "--truth_dir"; opts["truth_dir"] = args[i+1]; i += 2
        elseif a == "--force";   opts["force"] = true; i += 1
        elseif a == "--no-metrics"; opts["no_metrics"] = true; i += 1
        elseif a == "--help" || a == "-h"
            println(stdout, _help()); exit(0)
        else
            @warn "Unknown arg $a — ignored"; i += 1
        end
    end
    @assert opts["alg"] !== nothing "missing --alg"
    @assert opts["problem"] !== nothing "missing --problem"
    @assert opts["alg"] in ALG_NAMES "unknown algorithm $(opts["alg"]); allowed: $ALG_NAMES"
    @assert opts["problem"] in PROBLEM_NAMES "unknown problem $(opts["problem"]); allowed: $PROBLEM_NAMES"
    return opts
end

_help() = """
01_run_cell.jl — run a single benchmark cell.

Required:
  --alg     {is, mh, nuts, ns, mw}
  --problem {mvn, banana, funnel, mridges, shell, mridges_spiky, eggbox}
Optional:
  --d N            dimension (default 5)
  --B 5e5          budget (default 5e3)
  --seed N         seed (default 11)
  --out PATH       run-output root (default experiments/out)
  --truth_dir PATH truth root (default experiments/out/truth)
  --force          overwrite existing result
  --no-metrics     skip the per-cell metric computation
"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

function _make_cfg(problem::Symbol, d::Int)
    if problem === :mvn;     return make_config_mvn(d = d)
    elseif problem === :banana; return make_config_banana(d = d)
    elseif problem === :funnel; return make_config_funnel(d = d)
    elseif problem === :mridges; return make_config_mridges(d = d)
    elseif problem === :shell;   return make_config_shell(d = d)
    # V8-FIX-A7: the CLI accepted the V3 problem names but had no
    # constructors for them; single-cell reruns errored out (or silently
    # kept stale results). Defaults mirror 04_run_all.jl.
    elseif problem === :mridges_spiky
        return make_config_mridges_spiky(d = d, M = 4, σ_spike = 0.1,
                                          kernel = :gaussian)
    elseif problem === :eggbox; return make_config_eggbox(d = d)
    end
    error("unknown problem $problem")
end

function _truth_path(truth_root, problem::Symbol, d::Int)
    return joinpath(truth_root, @sprintf("%s_d%d.h5", String(problem), d))
end


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

function main(args = ARGS)
    opts = _parse_args(collect(String, args))
    alg, problem, d = opts["alg"], opts["problem"], opts["d"]
    B, seed = opts["B"], opts["seed"]
    cell_root = cell_dir(opts["out"], problem, alg, d, B, seed)
    isdir(cell_root) || mkpath(cell_root)
    if isfile(joinpath(cell_root, "result.h5")) && !opts["force"]
        @info "Skipping (already exists)" cell = cell_root
        return 0
    end

    cfg = _make_cfg(problem, d)
    started_utc = string(Dates.now(UTC))
    @info "Running cell" alg problem d B seed cell_root

    log_f = build_log_f(cfg)
    counter = LikelihoodCounter(log_f)
    mr = nothing
    notes = ""
    try
        mr = run_algorithm(alg, cfg, B, seed; counter = counter)
    catch err
        notes = "Sampler raised: $(typeof(err)) — see stack trace in stderr"
        @error "Sampler raised — writing empty result" alg problem d B seed exception = (err, catch_backtrace())
    end
    finished_utc = string(Dates.now(UTC))

    if mr === nothing
        # Defensive empty result so downstream pipeline survives
        mr = MethodResult(
            algorithm = alg, problem = problem, d = d, seed = Int(seed),
            B = Float64(B), counter = counter, wall_time_s = 0.0,
            samples = zeros(d, 1), weights = [1.0], logd = [NaN],
            extras = Dict{Symbol,Any}(:stop_reason => :error),
        )
    end

    save_method_result(cell_root, mr)
    overshoot = (mr.Nlike_used - B) / max(B, 1.0)
    if overshoot > 0.10 && alg != :is
        notes = strip(string(notes,
            (isempty(notes) ? "" : "; "),
            @sprintf("Overshoot %.1f%%", 100overshoot)))
    end

    meta = cell_metadata(
        problem = problem, algorithm = alg, d = d, seed = Int(seed), B = Float64(B),
        started_utc = started_utc, finished_utc = finished_utc,
        problem_config = Dict{String,Any}(
            String(fld) => let v = getfield(cfg, fld)
                v isa AbstractVector ? collect(v) : v
            end
            for fld in fieldnames(typeof(cfg))
        ),
        algorithm_tuning = Dict{String,Any}(String(k) => _safe_meta(v) for (k, v) in mr.extras),
        notes = notes,
    )
    meta["counter"] = Dict{String,Any}(
        "n_primal" => mr.n_primal,
        "n_grad_partials" => mr.n_grad_partials,
        "Nlike_used" => mr.Nlike_used,
        "wall_time_s" => mr.wall_time_s,
    )
    write_metadata_json(cell_root, meta)
    @info "Cell done" Nlike_used = mr.Nlike_used wall_time_s = mr.wall_time_s

    # ------------------------------------------------------------------
    # Optional per-cell metrics row (also recomputed in 02_aggregate.jl)
    # ------------------------------------------------------------------
    if !opts["no_metrics"]
        truth_path = _truth_path(opts["truth_dir"], problem, d)
        if !isfile(truth_path)
            @warn "Truth file missing — metrics skipped" path = truth_path
        else
            truth = load_truth(truth_path)
            row = compute_metrics_for_cell(mr, truth)
            row[:terminated_by] = String(get(mr.extras, :stop_reason, :unknown))
            row[:notes] = notes
            open(joinpath(cell_root, "metrics.json"), "w") do io
                ExperimentsBase._json_encode(io, Dict(String(k) => v for (k, v) in row))
            end
            @info "Metrics" W1 = row[:W1_marginal_avg] eta = row[:eta_Nlike] dlogZ = row[:dlogZ]
        end
    end
    return 0
end

# Make a JSON-friendly value out of arbitrary extras (skip mixtures etc.)
function _safe_meta(v)
    if v isa Symbol
        return String(v)
    elseif v isa Number || v isa AbstractString || v isa Bool
        return v
    elseif v isa AbstractVector{<:Real}
        return collect(v)
    elseif v isa AbstractVector
        return [_safe_meta(x) for x in v]
    elseif v isa AbstractDict
        return Dict(String(k) => _safe_meta(val) for (k, val) in v)
    elseif v isa NamedTuple
        return Dict(String(k) => _safe_meta(v[k]) for k in keys(v))
    else
        return string(typeof(v))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end

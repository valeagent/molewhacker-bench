# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# 00_generate_truth.jl — produce one truth file per (problem, d)
# =============================================================================
#
# Usage (from the MoleWhacker repository root):
#
#     julia --project=. -t auto experiments/scripts/00_generate_truth.jl \
#         [--problems mvn,banana,funnel,mridges,shell] \
#         [--dims 5] [--N_ref 1000000] [--out experiments/out/truth]
#
# Writes:
#     experiments/out/truth/<problem>_d<d>.h5      (JLD2 binary)
#     experiments/out/truth/<problem>_d<d>.json    (lightweight metadata)
#
# Each call to compute_truth(cfg) is deterministic given the truth seed
# (default 20260501), so re-runs give bit-identical files.

using Pkg

# Make sure we are running with the parent project active.
const _ROOT = abspath(joinpath(@__DIR__, "..", ".."))
if Base.active_project() != joinpath(_ROOT, "Project.toml")
    Pkg.activate(_ROOT)
end

include(joinpath(_ROOT, "experiments", "src", "ExperimentsBase.jl"))
using .ExperimentsBase

using Printf
using Random
using Statistics
using LinearAlgebra
using Dates


# -----------------------------------------------------------------------------
# CLI parsing (minimal; no ArgParse dependency)
# -----------------------------------------------------------------------------

function _parse_args(args::Vector{String})
    opts = Dict{String,Any}(
        "problems" => collect(string.(PROBLEM_NAMES)),
        "dims"     => [HEADLINE_DIMENSION],
        "N_ref"    => 1_000_000,
        "out"      => joinpath(_ROOT, "experiments", "out", "truth"),
        "include_scaling" => true,
        "force"    => false,
        "seed"     => 20260501,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--problems"
            opts["problems"] = String.(split(args[i+1], ','))
            i += 2
        elseif a == "--dims"
            opts["dims"] = [parse(Int, x) for x in split(args[i+1], ',')]
            i += 2
        elseif a == "--N_ref"
            opts["N_ref"] = parse(Int, args[i+1])
            i += 2
        elseif a == "--out"
            opts["out"] = args[i+1]
            i += 2
        elseif a == "--seed"
            opts["seed"] = parse(Int, args[i+1])
            i += 2
        elseif a == "--no-scaling"
            opts["include_scaling"] = false
            i += 1
        elseif a == "--force"
            opts["force"] = true
            i += 1
        elseif a == "--help" || a == "-h"
            println(stdout, _help())
            exit(0)
        else
            @warn "Unknown argument $a — ignored"
            i += 1
        end
    end
    return opts
end

function _help()
    return """
    00_generate_truth.jl

    Options:
      --problems mvn,banana,funnel,mridges,shell   default: all five
      --dims 5,10                                   default: 5
      --N_ref 1000000                               number of truth samples
      --out PATH                                    default: experiments/out/truth
      --no-scaling                                  skip the (funnel, shell) × {2, 10} subset
      --force                                       overwrite existing truth files
      --seed N                                      truth seed (default 20260501)
    """
end


# -----------------------------------------------------------------------------
# Build cfg from problem symbol + d
# -----------------------------------------------------------------------------

function _make_cfg(problem::Symbol, d::Int)
    if problem === :mvn
        return make_config_mvn(d = d)
    elseif problem === :banana
        return make_config_banana(d = d)
    elseif problem === :funnel
        return make_config_funnel(d = d)
    elseif problem === :mridges
        return make_config_mridges(d = d)
    elseif problem === :shell
        return make_config_shell(d = d)
    elseif problem === :mridges_spiky
        return make_config_mridges_spiky(d = d)
    elseif problem === :eggbox
        return make_config_eggbox(d = d)
    else
        error("unknown problem $problem")
    end
end


# -----------------------------------------------------------------------------
# Plan: every (problem, d) we should generate. Headline d=5 for all problems
# plus {2, 10} for {funnel, shell} when --no-scaling is not set.
# -----------------------------------------------------------------------------

function _plan(opts)
    out = Tuple{Symbol,Int}[]
    for p in Symbol.(opts["problems"])
        for d in opts["dims"]
            push!(out, (p, d))
        end
        if opts["include_scaling"] && p in SCALING_PROBLEMS
            for d in SCALING_DIMENSIONS
                if (p, d) ∉ out
                    push!(out, (p, d))
                end
            end
        end
    end
    return out
end


# -----------------------------------------------------------------------------
# Truth-file metadata sidecar
# -----------------------------------------------------------------------------

function _write_truth_metadata(meta_path, truth, cfg, N_ref, seed, t_seconds)
    meta = Dict{String,Any}(
        "problem"          => String(truth.problem),
        "d"                => truth.d,
        "N_ref"            => N_ref,
        "seed"             => seed,
        "elapsed_s"        => Float64(t_seconds),
        "logZ"             => truth.logZ,
        "mean_norm"        => sqrt(sum(abs2, truth.mean)),
        "trace_cov"        => tr(truth.cov),
        "max_abs_coord"    => maximum(abs, truth.samples),
        "schema_version"   => 1,
        "generated_utc"    => string(Dates.now(UTC)),
        "config"           => Dict{String,Any}(
            String(fld) => let v = getfield(cfg, fld)
                v isa AbstractVector ? collect(v) : v
            end
            for fld in fieldnames(typeof(cfg))
        ),
    )
    write_metadata_json(dirname(meta_path), Dict{String,Any}("truth" => meta))
    return meta_path
end


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

function main(args = ARGS)
    opts = _parse_args(collect(String, args))
    out_dir = opts["out"]
    isdir(out_dir) || mkpath(out_dir)
    plan = _plan(opts)
    @info "Truth generation plan" cells = length(plan) out = out_dir

    for (problem, d) in plan
        cfg = _make_cfg(problem, d)
        cfg === nothing && continue
        truth_path = joinpath(out_dir, @sprintf("%s_d%d.h5", String(problem), d))
        meta_path = joinpath(out_dir, @sprintf("%s_d%d.metadata.json", String(problem), d))
        if isfile(truth_path) && !opts["force"]
            @info "Skipping (already exists)" file = truth_path
            continue
        end
        @info "Computing truth" problem = problem d = d N_ref = opts["N_ref"]
        t0 = time()
        truth = compute_truth(cfg; N_ref = opts["N_ref"], seed = opts["seed"])
        elapsed = time() - t0
        save_truth(truth_path, truth)
        _write_truth_metadata(meta_path, truth, cfg, opts["N_ref"], opts["seed"], elapsed)
        @info "Truth saved" path = truth_path elapsed_s = round(elapsed; digits = 2)
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end

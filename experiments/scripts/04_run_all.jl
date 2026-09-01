# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# 04_run_all.jl — orchestrate the entire experimental run from scratch
# =============================================================================
#
# Calls 00_generate_truth.jl, then iterates the run-sheet of Protocol §7
# (5 algorithms × 5 problems × 1 dim × 3 budgets × 5 seeds = 375 cells
# at d=5, plus the scaling subset for funnel/shell at d∈{2,10}).
# Finally calls 02_aggregate.jl and 03_plots.jl.
#
# Intended invocation (from the MoleWhacker repository root):
#
#   julia --project=. -t auto experiments/scripts/04_run_all.jl \
#       [--out experiments/out] \
#       [--algs is,mh,nuts,ns,mw] \
#       [--problems mvn,banana,funnel,mridges,shell] \
#       [--dims 5] [--scaling] \
#       [--budgets 5e3,5e4,5e5] \
#       [--seeds 11,23,41,67,97] \
#       [--skip_truth] [--skip_runs] [--skip_aggregate] [--skip_plots] \
#       [--force] [--dryrun]
#
# Within each (problem, d, B, seed) tuple the five algorithms run in
# sequence (Protocol §7) so they share the wall-time machine state.

using Pkg

const _ROOT = abspath(joinpath(@__DIR__, "..", ".."))
if Base.active_project() != joinpath(_ROOT, "Project.toml")
    Pkg.activate(_ROOT)
end

include(joinpath(_ROOT, "experiments", "src", "ExperimentsBase.jl"))
using .ExperimentsBase
using Printf, Dates


function _parse_args(args::Vector{String})
    opts = Dict{String,Any}(
        "out" => joinpath(_ROOT, "experiments", "out"),
        "algs" => collect(string.(ALG_NAMES)),
        "problems" => collect(string.(PROBLEM_NAMES)),
        "dims" => [HEADLINE_DIMENSION],
        "budgets" => collect(Float64, BUDGET_GRID),
        "seeds" => collect(Int, SEED_GRID),
        "scaling" => false,
        "skip_truth" => false,
        "skip_runs" => false,
        "skip_aggregate" => false,
        "skip_tests" => false,
        "skip_plots" => false,
        "skip_warmup" => false,
        "force" => false,
        "dryrun" => false,
        "stop_on_error" => false,
        "with_mmd" => false,
        "with_cd" => false,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--out"; opts["out"] = args[i+1]; i += 2
        elseif a == "--algs"; opts["algs"] = String.(split(args[i+1], ',')); i += 2
        elseif a == "--problems"; opts["problems"] = String.(split(args[i+1], ',')); i += 2
        elseif a == "--dims"; opts["dims"] = [parse(Int, x) for x in split(args[i+1], ',')]; i += 2
        elseif a == "--budgets"; opts["budgets"] = [parse(Float64, x) for x in split(args[i+1], ',')]; i += 2
        elseif a == "--seeds"; opts["seeds"] = [parse(Int, x) for x in split(args[i+1], ',')]; i += 2
        elseif a == "--scaling"; opts["scaling"] = true; i += 1
        elseif a == "--skip_truth"; opts["skip_truth"] = true; i += 1
        elseif a == "--skip_runs"; opts["skip_runs"] = true; i += 1
        elseif a == "--skip_aggregate"; opts["skip_aggregate"] = true; i += 1
        elseif a == "--skip_tests"; opts["skip_tests"] = true; i += 1
        elseif a == "--skip_plots"; opts["skip_plots"] = true; i += 1
        elseif a == "--skip_warmup"; opts["skip_warmup"] = true; i += 1
        elseif a == "--force"; opts["force"] = true; i += 1
        elseif a == "--dryrun"; opts["dryrun"] = true; i += 1
        elseif a == "--stop_on_error"; opts["stop_on_error"] = true; i += 1
        elseif a == "--with-mmd" || a == "--with_mmd"; opts["with_mmd"] = true; i += 1
        elseif a == "--with-cd"  || a == "--with_cd"; opts["with_cd"]  = true; i += 1
        elseif a == "--help" || a == "-h"
            println(_help()); exit(0)
        else
            @warn "Unknown arg $a — ignored"; i += 1
        end
    end
    return opts
end

_help() = """
04_run_all.jl — full pipeline orchestrator.

Stages (each can be skipped):
  --skip_truth       skip 00_generate_truth.jl
  --skip_runs        skip per-cell sampler runs
  --skip_aggregate   skip 02_aggregate.jl
  --skip_plots       skip 03_plots.jl

Selection:
  --algs is,mh,nuts,ns,mw           (default: all five)
  --problems mvn,banana,funnel,mridges,shell
  --dims 5                           (default: 5)
  --scaling                          add (funnel,shell) × {2,10} subset
  --budgets 5e3,5e4,5e5
  --seeds 11,23,41,67,97

Other:
  --force            re-run cells / overwrite truth files even if present
  --dryrun           list the planned cells; do not run anything
  --stop_on_error    abort the run on the first sampler error
  --with-mmd         compute the mmd_rbf metric (slow; off by default since v3-lite)
  --with-cd          emit Friedman/Nemenyi tests + CD diagram (off by default since v3-lite)
"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

function _problem_d_pairs(opts)
    pairs = Tuple{Symbol,Int}[]
    for p in Symbol.(opts["problems"])
        for d in opts["dims"]
            push!(pairs, (p, d))
        end
        if opts["scaling"] && p in SCALING_PROBLEMS
            for d in SCALING_DIMENSIONS
                if (p, d) ∉ pairs
                    push!(pairs, (p, d))
                end
            end
        end
    end
    return pairs
end


function _stage_runs(opts)
    @info "Stage 1/4: per-cell sampler runs"
    pairs = _problem_d_pairs(opts)
    cells = Tuple{Symbol,Int,Float64,Int}[]
    # Outer loop: (problem, d, B, seed); inner loop: algorithms
    for (p, d) in pairs
        for B in opts["budgets"]
            for seed in opts["seeds"]
                push!(cells, (p, d, B, seed))
            end
        end
    end
    @info "Run grid" n_cells = length(cells) n_algs = length(opts["algs"])
    if opts["dryrun"]
        for (p, d, B, seed) in cells
            for alg in opts["algs"]
                println("DRY: ", alg, " ", p, " d=", d, " B=", B, " seed=", seed)
            end
        end
        return
    end

    if !opts["skip_warmup"]
        _jit_warmup(opts)
    end

    n_total = length(cells) * length(opts["algs"])
    n_done = 0
    n_err = 0
    truth_root = joinpath(opts["out"], "truth")
    truth_cache = Dict{Tuple{Symbol,Int},TruthSet}()
    for (p, d, B, seed) in cells
        for alg in opts["algs"]
            n_done += 1
            @info "Cell $n_done/$n_total" alg p d B seed
            try
                _run_one_cell(Symbol(alg), p, Int(d), Float64(B), Int(seed),
                                opts, truth_cache)
            catch err
                n_err += 1
                @error "Cell failed" alg p d B seed exception = (err, catch_backtrace())
                opts["stop_on_error"] && rethrow(err)
            end
        end
    end
    @info "Stage 1/4 finished" n_total n_done n_err
end


# -----------------------------------------------------------------------------
# JIT warm-up loop (Protocol §7.1, fix P0-5).
# -----------------------------------------------------------------------------
#
# The first invocation of every BAT.jl + AdvancedHMC + NestedSamplers
# call pays a 4–6 s compile cost. Without warm-up the wall-time of
# first-seed cells is dominated by JIT, making the Pareto / wall-time
# plots untrustworthy. We run every algorithm once on a tiny throw-away
# `mvn` cell at B = 1e3 before the timed loop. The results are
# discarded.
# -----------------------------------------------------------------------------
function _jit_warmup(opts)
    @info "JIT warm-up — running every algorithm once on a throw-away mvn cell"
    cfg_warm = make_config_mvn(d = HEADLINE_DIMENSION)
    for alg in opts["algs"]
        try
            t0 = time_ns()
            counter = LikelihoodCounter(build_log_f(cfg_warm))
            mr = run_algorithm(Symbol(alg), cfg_warm, 1.0e3, 11; counter = counter)
            dt = (time_ns() - t0) / 1e9
            @info "Warmed up $alg" wall_time_s = dt Nlike_used = mr.Nlike_used
        catch err
            @warn "Warmup of $alg raised — continuing" exception = (err, catch_backtrace())
        end
    end
    @info "JIT warm-up complete"
end


# In-process per-cell runner. Uses the ExperimentsBase symbols already
# in scope (via `using .ExperimentsBase` at the top of this script).
function _run_one_cell(alg::Symbol, problem::Symbol, d::Int, B::Float64,
                          seed::Int, opts::AbstractDict, truth_cache::AbstractDict)
    cell_root = cell_dir(opts["out"], problem, alg, d, B, seed)
    isdir(cell_root) || mkpath(cell_root)
    if isfile(joinpath(cell_root, "result.h5")) && !opts["force"]
        @info "Skipping (already exists)" cell = cell_root
        return
    end
    cfg = if problem === :mvn;             make_config_mvn(d = d)
        elseif problem === :banana;          make_config_banana(d = d)
        elseif problem === :funnel;          make_config_funnel(d = d)
        elseif problem === :mridges;         make_config_mridges(d = d)
        elseif problem === :shell;           make_config_shell(d = d)
        elseif problem === :mridges_spiky
            spiky = get(opts, "spiky_setting",
                          (kernel = :gaussian, M = 4, σ_spike = 0.1))
            make_config_mridges_spiky(d = d,
                                       M = spiky.M,
                                       σ_spike = spiky.σ_spike,
                                       kernel = spiky.kernel)
        elseif problem === :eggbox;          make_config_eggbox(d = d)
        else error("unknown problem $problem")
        end
    started_utc = string(Dates.now(UTC))
    log_f = build_log_f(cfg)
    counter = LikelihoodCounter(log_f)
    notes = ""
    mr = nothing
    try
        mr = run_algorithm(alg, cfg, B, seed; counter = counter)
    catch err
        notes = "Sampler raised: $(typeof(err))"
        @error "Sampler raised — writing empty result" alg problem d B seed exception = (err, catch_backtrace())
    end
    finished_utc = string(Dates.now(UTC))
    if mr === nothing
        mr = MethodResult(
            algorithm = alg, problem = problem, d = d, seed = seed,
            B = B, counter = counter, wall_time_s = 0.0,
            samples = zeros(d, 1), weights = [1.0], logd = [NaN],
            extras = Dict{Symbol,Any}(:stop_reason => :error),
        )
    end
    save_method_result(cell_root, mr)
    overshoot = (mr.Nlike_used - B) / max(B, 1.0)
    if overshoot > 0.10 && alg !== :is
        notes = strip(string(notes,
            (isempty(notes) ? "" : "; "),
            @sprintf("Overshoot %.1f%%", 100overshoot)))
    end
    meta = cell_metadata(
        problem = problem, algorithm = alg, d = d, seed = seed, B = B,
        started_utc = started_utc, finished_utc = finished_utc,
        problem_config = Dict{String,Any}(
            String(fld) => let v = getfield(cfg, fld)
                v isa AbstractVector ? collect(v) : v
            end
            for fld in fieldnames(typeof(cfg))
        ),
        algorithm_tuning = Dict{String,Any}(),
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
    # Compute per-cell metrics and persist
    truth_root = joinpath(opts["out"], "truth")
    key = (problem, d)
    truth = if haskey(truth_cache, key)
        truth_cache[key]
    else
        tp = joinpath(truth_root, @sprintf("%s_d%d.h5", String(problem), d))
        isfile(tp) ? (truth_cache[key] = load_truth(tp)) : nothing
    end
    if truth !== nothing
        try
            row = compute_metrics_for_cell(mr, truth;
                                                with_mmd = get(opts, "with_mmd", false))
            # If the in-process runner had its own `notes` (e.g. from
            # an exception), prepend them — `compute_metrics_for_cell`
            # adds BUDGET-VIOLATION / RHAT-FAIL but doesn't see runner
            # exceptions.
            if !isempty(notes)
                existing = String(get(row, :notes, ""))
                row[:notes] = isempty(existing) ? notes : (notes * ";" * existing)
            end
            open(joinpath(cell_root, "metrics.json"), "w") do io
                ExperimentsBase._json_encode(io, Dict(String(k) => v for (k, v) in row))
            end
        catch err
            @warn "Per-cell metrics failed" exception = (err, catch_backtrace())
        end
    end
    return mr
end


function _stage_aggregate(opts)
    @info "Stage 2/4: aggregation"
    if opts["dryrun"]
        println("would run 02_aggregate.jl"); return
    end
    args = String[
        "--out", opts["out"],
        "--truth_dir", joinpath(opts["out"], "truth"),
    ]
    opts["with_mmd"] && push!(args, "--with-mmd")
    _run_subscript(joinpath(@__DIR__, "02_aggregate.jl"), args)
end


function _stage_tests(opts)
    @info "Stage 2b/4: statistical tests (V3)"
    if opts["dryrun"]
        println("would run 02b_tests.jl"); return
    end
    csv_path = joinpath(opts["out"], "tables", "cells.csv")
    isfile(csv_path) || begin
        @warn "Skipping tests stage (cells.csv missing)" csv_path
        return
    end
    args = String[
        "--csv", csv_path,
        "--out_dir", joinpath(opts["out"], "tables"),
    ]
    opts["with_cd"] && push!(args, "--with-cd")
    _run_subscript(joinpath(@__DIR__, "02b_tests.jl"), args)
end


function _stage_plots(opts)
    @info "Stage 3/4: figure generation"
    if opts["dryrun"]
        println("would run 03_plots.jl"); return
    end
    args = String[
        "--out", opts["out"],
        "--cells", joinpath(opts["out"], "tables", "cells.csv"),
        "--truth_dir", joinpath(opts["out"], "truth"),
        "--runs_dir", joinpath(opts["out"], "runs"),
        "--figs_dir", joinpath(opts["out"], "figs"),
    ]
    opts["with_cd"] && push!(args, "--with-cd")
    _run_subscript(joinpath(@__DIR__, "03_plots.jl"), args)
end


# Spawn a fresh `julia` for stages 2 and 3 — they don't share state
# with stage 1 and isolating them avoids module-redefinition warnings
# in this orchestrator process.
function _run_subscript(path::AbstractString, args::Vector{String})
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -t auto $path $(args...)`
    @info "Running sub-script" path
    proc = run(pipeline(cmd; stdout = stdout, stderr = stderr); wait = true)
    success(proc) || @warn "Sub-script returned non-zero exit code" path
    return proc
end


# (Stage 0 / truth generation also gets a fresh process so the truth
# script can use its own argument parser cleanly.)
function _stage_truth(opts)
    @info "Stage 0/4: truth generation"
    args = String[
        "--out", joinpath(opts["out"], "truth"),
        "--problems", join(opts["problems"], ','),
        "--dims", join(opts["dims"], ','),
    ]
    opts["scaling"] || push!(args, "--no-scaling")
    opts["force"] && push!(args, "--force")
    if opts["dryrun"]
        println("would run 00_generate_truth.jl ", join(args, ' '))
        return
    end
    _run_subscript(joinpath(@__DIR__, "00_generate_truth.jl"), args)
end


function main(args = ARGS)
    opts = _parse_args(collect(String, args))
    started = now()
    @info "==================== 04_run_all.jl ====================" started = string(started)
    opts["skip_truth"]     || _stage_truth(opts)
    opts["skip_runs"]      || _stage_runs(opts)
    opts["skip_aggregate"] || _stage_aggregate(opts)
    opts["skip_tests"]     || _stage_tests(opts)
    opts["skip_plots"]     || _stage_plots(opts)
    @info "==================== Done ====================" elapsed = string(now() - started)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end

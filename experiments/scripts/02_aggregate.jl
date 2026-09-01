# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# 02_aggregate.jl — collect every cell into experiments/out/tables/cells.csv
# =============================================================================
#
# Walks experiments/out/runs/, recomputes metrics from the saved
# MethodResult + truth, and writes one row per cell into cells.csv with
# the schema of Protocol §8.
#
# Usage:
#   julia --project=. -t auto experiments/scripts/02_aggregate.jl \
#       [--out experiments/out] [--truth_dir experiments/out/truth]

using Pkg

const _ROOT = abspath(joinpath(@__DIR__, "..", ".."))
if Base.active_project() != joinpath(_ROOT, "Project.toml")
    Pkg.activate(_ROOT)
end

include(joinpath(_ROOT, "experiments", "src", "ExperimentsBase.jl"))
using .ExperimentsBase
using DataFrames, CSV, Printf, Dates, SHA


function _parse_args(args::Vector{String})
    opts = Dict{String,Any}(
        "out" => joinpath(_ROOT, "experiments", "out"),
        "truth_dir" => joinpath(_ROOT, "experiments", "out", "truth"),
        "csv" => nothing,
        "with_mmd" => false,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--out"; opts["out"] = args[i+1]; i += 2
        elseif a == "--truth_dir"; opts["truth_dir"] = args[i+1]; i += 2
        elseif a == "--csv"; opts["csv"] = args[i+1]; i += 2
        elseif a == "--with-mmd" || a == "--with_mmd"; opts["with_mmd"] = true; i += 1
        elseif a == "--help" || a == "-h"
            println("02_aggregate.jl [--out PATH] [--truth_dir PATH] [--csv PATH] [--with-mmd]")
            println()
            println("  --with-mmd   compute the mmd_rbf metric per cell (slow; off by default")
            println("               since v3-lite. Headline figures use W1 + dlogZ + mode_recovery.)")
            exit(0)
        else
            @warn "Unknown arg $a — ignored"; i += 1
        end
    end
    return opts
end


function main(args = ARGS)
    opts = _parse_args(collect(String, args))
    runs_dir = joinpath(opts["out"], "runs")
    isdir(runs_dir) || error("No runs directory at $runs_dir")
    csv_path = opts["csv"] === nothing ?
        joinpath(opts["out"], "tables", "cells.csv") : opts["csv"]
    isdir(dirname(csv_path)) || mkpath(dirname(csv_path))

    cells = readdir(runs_dir; join = true)
    cells = filter(isdir, cells)
    nthr = Threads.nthreads()
    @info "Aggregating cells" n_cells = length(cells) csv_path = csv_path with_mmd = opts["with_mmd"] threads = nthr

    # Pre-load every truth file we need so the per-thread loop is read-only
    # against `truth_cache` (no locking required). Truth keys are parsed
    # from cell directory names so we don't need to open every result.h5.
    # Cell name pattern: `<problem>_<alg>_d<d>_B<budget>_seed<seed>`, where
    # `<problem>` may itself contain underscores (e.g. `mridges_spiky`).
    function _parse_cell_key(name::AbstractString)
        m = match(r"^(.*)_([a-z]+)_d(\d+)_B[\w.]+_seed\d+$", name)
        m === nothing && return nothing
        return (Symbol(m.captures[1]), parse(Int, m.captures[3]))
    end

    needed_keys = Set{Tuple{Symbol,Int}}()
    for cell in cells
        isfile(joinpath(cell, "result.h5")) || continue
        k = _parse_cell_key(basename(cell))
        k === nothing && continue
        push!(needed_keys, k)
    end
    truth_cache = Dict{Tuple{Symbol,Int},TruthSet}()
    for key in needed_keys
        tp = joinpath(opts["truth_dir"],
                      @sprintf("%s_d%d.h5", String(key[1]), key[2]))
        if !isfile(tp)
            @warn "Truth file missing — cells with this key will be skipped" path = tp
            continue
        end
        truth_cache[key] = load_truth(tp)
    end
    @info "Truth cache warm" loaded = length(truth_cache)

    # Per-thread sigma caches (mmd_rbf bandwidth); merged later if needed.
    per_thread_rows = [Vector{Dict{Symbol,Any}}() for _ in 1:nthr]
    per_thread_sigma = [Dict{Any,Float64}() for _ in 1:nthr]

    progress_lock = ReentrantLock()
    n_done = Threads.Atomic{Int}(0)
    Threads.@threads :static for cell in cells
        tid = Threads.threadid()
        rh5 = joinpath(cell, "result.h5")
        isfile(rh5) || continue
        local mr
        try
            mr = load_method_result(cell)
        catch err
            @warn "Could not load result.h5" cell exception = err
            continue
        end
        key = (mr.problem, mr.d)
        haskey(truth_cache, key) || continue
        truth = truth_cache[key]
        row = compute_metrics_for_cell(mr, truth;
                                            sigma_cache = per_thread_sigma[tid],
                                            with_mmd = opts["with_mmd"])
        push!(per_thread_rows[tid], row)
        n = Threads.atomic_add!(n_done, 1) + 1
        if n % 100 == 0
            lock(progress_lock) do
                @info "Aggregator progress" n_done = n n_total = length(cells)
            end
        end
    end
    rows = reduce(vcat, per_thread_rows)

    if isempty(rows)
        @warn "No cells aggregated; cells.csv not written"
        return 1
    end

    # Stable column order. Rhat_max is added per Protocol §9.3a; the
    # `notes` column carries `BUDGET-VIOLATION` (§6.1) and / or
    # `RHAT-FAIL` (§9.3a) flags, set in metrics.jl.
    cols = [
        :problem, :algorithm, :d, :B, :seed,
        :Nlike_used, :wall_time_s, :n_primal, :n_grad_partials, :n_dual_calls,
        :Neff, :eta_Nlike,
        :W1_marginal_avg, :SWD,
        :dlogZ,
        :QE_p025, :QE_p160, :QE_p500, :QE_p840, :QE_p975,
        :KL_cube,
        :mode_recovery,    # Protocol §9.7 (V3)
        :mmd_rbf,          # Protocol §9.8 (V3)
        :Rhat_max,
        :terminated_by, :notes,
    ]
    df = DataFrame()
    for c in cols
        df[!, c] = [get(r, c, missing) for r in rows]
    end
    sort!(df, [:problem, :algorithm, :d, :B, :seed])
    CSV.write(csv_path, df)
    @info "cells.csv written" csv = csv_path n_rows = nrow(df)
    # SHA-256 sidecar for figure provenance
    open(csv_path * ".sha256", "w") do io
        write(io, bytes2hex(SHA.sha256(read(csv_path))) * "  " * basename(csv_path) * "\n")
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end

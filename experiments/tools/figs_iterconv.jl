# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# =============================================================================
# figs_iterconv.jl — two jobs in one pass over all MW runs:
#
#   (1) KL_cube recompute: recompute the KL_cube column of cells.csv with
#       the corrected kl_cube_mw (mixture draws mapped from the PriorToNormal
#       z-space to user space with the exact box-prior transform+Jacobian;
#       an earlier implementation evaluated the user-space target at z-space
#       points, making those values meaningless). cells.csv is backed up first.
#
#   (2) Iteration-convergence figures: for every (problem, d, B) cell of
#       the grid, render eta(t) = Neff(t)/NL(t) for all available seeds
#       (fig_mw_itercurves) from the recorded iter_log — no re-running.
#       Also write out/tables/mw_iteration_summary.csv with the stop
#       iteration and stop reason per cell.
# =============================================================================

using Pkg
const _ROOT = abspath(joinpath(@__DIR__, "..", ".."))
Base.active_project() != joinpath(_ROOT, "Project.toml") && Pkg.activate(_ROOT)
include(joinpath(_ROOT, "experiments", "src", "ExperimentsBase.jl"))
using .ExperimentsBase
using DataFrames, CSV, Printf, Statistics

const OUT = joinpath(_ROOT, "experiments", "out")
const FIGS = joinpath(OUT, "figs")
const BUDGETS = (5e3, 5e4, 5e5)
const COMBOS = [
    (:mvn,           (2, 5, 10)),
    (:banana,        (2, 5, 10)),
    (:funnel,        (2, 5, 10)),
    (:mridges,       (2, 5, 10)),
    (:mridges_spiky, (5,)),
    (:eggbox,        (2,)),
    (:shell,         (5,)),
]

function main()
    cells_path = joinpath(OUT, "tables", "cells.csv")
    df = CSV.read(cells_path, DataFrame)
    backup = joinpath(OUT, "tables", "cells_pre_klcube_recompute.csv")
    isfile(backup) || CSV.write(backup, df)
    @info "cells.csv loaded" rows = nrow(df) backup = backup

    truth_cache = Dict{Tuple{Symbol,Int},Any}()
    function get_truth(prob::Symbol, d::Int)
        get!(truth_cache, (prob, d)) do
            load_truth(joinpath(OUT, "truth", @sprintf("%s_d%d.h5", String(prob), d)))
        end
    end

    n_patched = 0
    n_nan = 0
    summary_rows = NamedTuple[]

    for (prob, dims) in COMBOS, d in dims, B in BUDGETS
        runs = MethodResult[]
        for seed in SEED_GRID
            dir = cell_dir(OUT, prob, :mw, d, B, seed)
            isdir(dir) || continue
            isfile(joinpath(dir, "result.h5")) || continue
            mr = try
                load_method_result(dir)
            catch err
                @warn "load failed" prob d B seed err
                continue
            end
            push!(runs, mr)
        end
        isempty(runs) && continue

        # --- (1) KL_cube recompute + cells.csv patch --------------------
        truth = try
            get_truth(prob, d)
        catch err
            @warn "truth load failed — KL skipped" prob d err
            nothing
        end
        if truth !== nothing
            for mr in runs
                kl = kl_cube_mw(mr, truth, truth.cfg)
                mask = (df.problem .== String(prob)) .& (df.algorithm .== "mw") .&
                       (df.d .== d) .& (df.B .== Float64(B)) .& (df.seed .== mr.seed)
                idx = findall(mask)
                length(idx) == 1 || (@warn "row match != 1" prob d B mr.seed n = length(idx); continue)
                df[idx[1], :KL_cube] = kl
                n_patched += 1
                isnan(kl) && (n_nan += 1)
            end
        end

        # --- (2) iteration figure + stop summary ------------------------
        try
            fig = fig_mw_itercurves(runs; problem = prob, d = d, B = B,
                                    class = :narrow)
            save_pdf(fig, fig_filename(family = :iterconv, problem = prob,
                       d = d, B = B, alg = :mw, extra = "eta"); dir = FIGS)
        catch err
            @warn "iterconv figure failed" prob d B err
        end

        t_stops = Int[]
        eta_final = Float64[]
        n_tmax = 0; n_budget = 0; n_other = 0
        for mr in runs
            il = get(mr.extras, :iter_log, NamedTuple[])
            isempty(il) && continue
            push!(t_stops, Int(last(il).iter))
            cum = Float64(last(il).cum_cost)
            cum > 0 && push!(eta_final, Float64(last(il).ess) / cum)
            stop = Symbol(get(mr.extras, :stop_reason, :unknown))
            stop === :T_max ? (n_tmax += 1) :
                stop === :budget ? (n_budget += 1) : (n_other += 1)
        end
        isempty(t_stops) && continue
        push!(summary_rows, (
            problem = String(prob), d = d, B = Float64(B),
            n_runs = length(runs),
            n_stop_T_max = n_tmax, n_stop_budget = n_budget,
            n_stop_other = n_other,
            t_stop_median = median(t_stops),
            t_stop_min = minimum(t_stops), t_stop_max = maximum(t_stops),
            eta_final_median = isempty(eta_final) ? NaN : median(eta_final),
        ))
    end

    CSV.write(cells_path, df)
    @info "KL_cube column patched" n_patched n_nan

    sum_path = joinpath(OUT, "tables", "mw_iteration_summary.csv")
    CSV.write(sum_path, DataFrame(summary_rows))
    @info "stop summary written" sum_path

    println("\n=== MW stop-iteration summary ===")
    @printf("%-14s %-3s %-6s %6s %6s %7s %8s %10s\n",
        "problem", "d", "B", "Tmax", "budget", "t_med", "t_range", "eta_med")
    for r in summary_rows
        @printf("%-14s %-3d %-6.0g %6d %6d %7.1f %3d-%-4d %10.3g\n",
            r.problem, r.d, r.B, r.n_stop_T_max, r.n_stop_budget,
            r.t_stop_median, r.t_stop_min, r.t_stop_max, r.eta_final_median)
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end

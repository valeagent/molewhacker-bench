# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# figs_viz_recovery_scaling.jl — targeted regeneration of the
# problem-identity and curve figures embedded by the thesis
# =============================================================================
#
# Regenerates exactly these thesis figure families:
#
#   1. viz__<p>__d2__surface       — z-axis label no longer clipped
#   2. viz__<p>__d5__overview      — stray minor ticks on marginal panels
#   3. tri__*                      — fonts scaled for half-width embedding;
#                                    d = 10 restricted to a coordinate subset
#   4. recovery__mridges(_spiky)   — rendered at half-column width
#   5. scaling__mridges__B5e5      — rendered at half-column width
#
# Output goes to experiments/out/figs (same filenames as 03_plots.jl).
#
# Usage:
#   julia --project=. -t 1 experiments/tools/figs_viz_recovery_scaling.jl

using Pkg
const _ROOT = abspath(joinpath(@__DIR__, "..", ".."))
if Base.active_project() != joinpath(_ROOT, "Project.toml")
    Pkg.activate(_ROOT)
end

include(joinpath(_ROOT, "experiments", "src", "ExperimentsBase.jl"))
using .ExperimentsBase
using DataFrames, CSV, Printf, Random

const OUT       = joinpath(_ROOT, "experiments", "out")
const FIGS_DIR  = joinpath(OUT, "figs")
const TRUTH_DIR = joinpath(OUT, "truth")
const RUNS_DIR  = joinpath(OUT, "runs")
const CELLS_CSV = joinpath(OUT, "tables", "cells.csv")

_make_cfg(p::Symbol, d::Int) = p === :mvn ? make_config_mvn(d = d) :
                                p === :banana ? make_config_banana(d = d) :
                                p === :funnel ? make_config_funnel(d = d) :
                                p === :mridges ? make_config_mridges(d = d) :
                                p === :shell ? make_config_shell(d = d) :
                                p === :mridges_spiky ? make_config_mridges_spiky(d = d) :
                                p === :eggbox ? make_config_eggbox(d = d) :
                                nothing

function _load_truth_file(problem::Symbol, d::Int)
    path = joinpath(TRUTH_DIR, @sprintf("%s_d%d.h5", String(problem), d))
    isfile(path) || return nothing
    return load_truth(path)
end

function _load_run(problem::Symbol, alg::Symbol, d::Int, B::Real, seed::Int)
    cell = cell_dir(joinpath(RUNS_DIR, ".."), problem, alg, d, B, seed)
    cell = joinpath(RUNS_DIR, basename(cell))
    isdir(cell) || return nothing
    isfile(joinpath(cell, "result.h5")) || return nothing
    return load_method_result(cell)
end

function _median_seed_run(problem::Symbol, alg::Symbol, d::Int, B::Real)
    candidates = MethodResult[]
    for seed in SEED_GRID
        mr = _load_run(problem, alg, d, B, seed)
        mr === nothing && continue
        push!(candidates, mr)
    end
    isempty(candidates) && return nothing
    return candidates[ceil(Int, length(candidates) / 2)]
end

Random.seed!(20260601)

# ---------------------------------------------------------------------------
# 1 + 2 — viz surfaces (d = 2) and overviews (d = 5)
# ---------------------------------------------------------------------------
surface_alias = Dict{Symbol,Symbol}(:mridges_spiky => :mspiky)
for prob in PROBLEM_NAMES
    cfg2 = _make_cfg(prob, 2)
    if cfg2 !== nothing
        try
            fig = fig_viz_surface_3d(cfg2)
            slug = String(get(surface_alias, prob, prob))
            save_pdf(fig, "viz__" * slug * "__d2__surface"; dir = FIGS_DIR)
            println("surface  $prob  OK")
        catch err
            @warn "surface failed" prob err
        end
    end
    cfg5 = _make_cfg(prob, 5)
    truth5 = _load_truth_file(prob, 5)
    if cfg5 !== nothing && truth5 !== nothing
        try
            fig = fig_viz_overview(cfg5, truth5)
            save_pdf(fig, fig_filename(family = :viz, problem = prob,
                      d = 5, extra = "overview"); dir = FIGS_DIR)
            println("overview $prob  OK")
        catch err
            @warn "overview failed" prob err
        end
    end
end

# ---------------------------------------------------------------------------
# 3 — the thirteen triangle plots the thesis embeds
# ---------------------------------------------------------------------------
tri_jobs = [
    (:mvn,     :mw,   5), (:mvn,     :nuts, 5),
    (:banana,  :mw,   5), (:banana,  :nuts, 5),
    (:funnel,  :mw,   5), (:funnel,  :nuts, 5),
    (:mridges, :mw,   5), (:mridges, :nuts, 5), (:mridges, :ns, 5),
    (:shell,   :mw,   5),
    (:mridges, :mw,   2), (:mridges, :mw,  10),
    (:eggbox,  :mw,   2),
]
for (prob, alg, d) in tri_jobs
    truth = _load_truth_file(prob, d)
    if truth === nothing
        @warn "truth missing for tri" prob d
        continue
    end
    mr = _median_seed_run(prob, alg, d, 5e5)
    if mr === nothing || size(mr.samples, 2) < 4
        @warn "run missing for tri" prob alg d
        continue
    end
    try
        fig = fig_tri(mr, truth)
        save_pdf(fig, fig_filename(family = :tri, problem = prob, d = d,
                   B = 5e5, alg = alg); dir = FIGS_DIR)
        println("tri      $prob $alg d=$d  OK")
    catch err
        @warn "tri failed" prob alg d err
    end
end

# ---------------------------------------------------------------------------
# 4 + 5 — recovery and scaling line plots at half-column width
# ---------------------------------------------------------------------------
df = DataFrame(CSV.File(CELLS_CSV))
for prob in (:mridges, :mridges_spiky)
    try
        fig = fig_recovery(df, prob; class = :narrow)
        save_pdf(fig, fig_filename(family = :recovery, problem = prob,
                  d = HEADLINE_DIMENSION, B = :all, alg = :all,
                  extra = "all"); dir = FIGS_DIR)
        println("recovery $prob  OK")
    catch err
        @warn "recovery failed" prob err
    end
end
try
    fig = fig_scaling(df, :mridges; B = 5e5, class = :narrow)
    save_pdf(fig, fig_filename(family = :scaling, problem = :mridges,
              d = nothing, B = 5e5, alg = :all,
              extra = "W1-vs-d"); dir = FIGS_DIR)
    println("scaling  mridges  OK")
catch err
    @warn "scaling failed" err
end

println("\nReplot complete.")

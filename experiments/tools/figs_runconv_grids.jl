# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# =============================================================================
# figs_runconv_grids.jl — appendix atlas: one per-problem 2x3 grid of the
# five single-algorithm running-efficiency panels (all seeds + median),
# headline settings. Chain-prefix ESS dominates the runtime (~1-2 h).
# =============================================================================

include(joinpath(@__DIR__, "..", "scripts", "03_plots.jl"))

const FIGS = joinpath(_ROOT, "experiments", "out", "figs")
const OUT = joinpath(_ROOT, "experiments", "out")
const HEADLINE = [(:mvn, 5), (:banana, 5), (:funnel, 5), (:mridges, 5),
                  (:shell, 5), (:mridges_spiky, 5), (:eggbox, 2)]
const ALGS = (:is, :mh, :nuts, :ns, :mw)
const B = 5e5

function _cell_runs(prob::Symbol, alg::Symbol, d::Int, Bb::Real)
    runs = MethodResult[]
    for seed in SEED_GRID
        dir = cell_dir(OUT, prob, alg, d, Bb, seed)
        isdir(dir) || continue
        isfile(joinpath(dir, "result.h5")) || continue
        mr = try
            load_method_result(dir)
        catch
            continue
        end
        size(mr.samples, 2) > 1 || continue
        push!(runs, mr)
    end
    return runs
end

for (prob, d) in HEADLINE
    name = fig_filename(family = :runconv, problem = prob, d = d, B = B,
                        alg = :all, extra = "grid")
    if isfile(joinpath(FIGS, name * ".pdf"))
        @info "exists — skipping" name
        continue
    end
    runs_by_alg = Pair{Symbol,Vector{MethodResult}}[]
    for alg in ALGS
        runs = _cell_runs(prob, alg, d, B)
        isempty(runs) && continue
        push!(runs_by_alg, alg => runs)
    end
    isempty(runs_by_alg) && continue
    t0 = time()
    fig = fig_runconv_grid(runs_by_alg; problem = prob, d = d, B = B)
    save_pdf(fig, name; dir = FIGS)
    @info "grid done" prob round(time() - t0; digits = 1)
end
@info "runconv grids done"

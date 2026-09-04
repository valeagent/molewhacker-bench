# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# figs_runconv_d10.jl — running-efficiency grids (all seeds + median, five
# single-algorithm panels) at d = 10 for the four scaling targets at the
# largest budget (thesis Appendix B). Identical construction to the d = 5
# atlas of figs_runconv_grids.jl; chain-prefix ESS dominates the runtime.
#
# Usage:  julia --project=. -t auto experiments/tools/figs_runconv_d10.jl
# =============================================================================

include(joinpath(@__DIR__, "..", "scripts", "03_plots.jl"))   # helpers; main() guarded

const FIGS = joinpath(_ROOT, "experiments", "out", "figs")
const OUT = joinpath(_ROOT, "experiments", "out")
const JOBS = [(:mvn, 10), (:banana, 10), (:funnel, 10), (:mridges, 10)]
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

function main()
    for (prob, d) in JOBS
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
    @info "runconv d=10 grids done" FIGS
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end

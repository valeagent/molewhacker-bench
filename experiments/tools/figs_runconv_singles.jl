# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# =============================================================================
# figs_runconv_singles.jl — running-efficiency atlas: one panel per
# (algorithm, problem) at the headline setting (d = 5, eggbox d = 2;
# B = 5e5), all seeds as light curves + across-seed median, x = samples
# produced, y = eta = Neff/NL. 35 panels total, saved incrementally as
#   runconv__<problem>__d<d>__B5e5__<alg>__eta.pdf
# Chain-prefix ESS is expensive; expect ~1-2 h wall time.
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

for (prob, d) in HEADLINE, alg in ALGS
    name = fig_filename(family = :runconv, problem = prob, d = d, B = B,
                        alg = alg, extra = "eta")
    if isfile(joinpath(FIGS, name * ".pdf"))
        @info "exists — skipping" name
        continue
    end
    runs = _cell_runs(prob, alg, d, B)
    if isempty(runs)
        @warn "no runs" prob alg d
        continue
    end
    t0 = time()
    fig = fig_runconv_panel(runs; algorithm = alg, problem = prob,
                            d = d, B = B, class = :narrow)
    save_pdf(fig, name; dir = FIGS)
    @info "panel done" prob alg round(time() - t0; digits = 1)
end
@info "runconv atlas done"

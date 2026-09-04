# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# figs_heatmap_angles.jl — the three summary heatmaps (eta, dlogZ, W1) at
# the four grid angles not shown in the thesis body (Appendix B):
#   (d = 5, B = 5e3), (d = 5, B = 5e4)  — budget pairs, all seven targets
#   (d = 2, B = 5e5), (d = 10, B = 5e5) — dimension pairs, restricted to
#                                          the targets with data there
#
# Usage:  julia --project=. experiments/tools/figs_heatmap_angles.jl
# =============================================================================

include(joinpath(@__DIR__, "..", "scripts", "03_plots.jl"))   # helpers; main() guarded
using DataFrames, CSV

const OUT  = joinpath(_ROOT, "experiments", "out")
const FIGS = joinpath(OUT, "figs")

const SCALING = Symbol[:mvn, :banana, :funnel, :mridges]
const ALLPROB = collect(ExperimentsBase.HEATMAP_PROBLEM_ORDER)

function main()
    df = CSV.read(joinpath(OUT, "tables", "cells.csv"), DataFrame)
    drop_v5_excluded_rows!(df)
    angles = [
        (5,  5e3, ALLPROB),
        (5,  5e4, ALLPROB),
        (2,  5e5, vcat(SCALING, :eggbox)),
        (10, 5e5, SCALING),
    ]
    for (metric, transform, tag) in [(:eta_Nlike,       :log10,          "eta"),
                                     (:dlogZ,           :negative_log10, "dlogz"),
                                     (:W1_marginal_avg, :negative_log10, "W1")]
        for (dd, BB, probs) in angles
            fig = fig_summary_heatmap(df, metric; transform = transform,
                    d = dd, B = BB, problems = probs)
            save_pdf(fig, fig_filename(family = :summary, problem = :all,
                      d = dd, B = BB, extra = "heatmap-" * tag); dir = FIGS)
        end
    end
    @info "appendix-angle heatmaps done (12 files)" FIGS
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end

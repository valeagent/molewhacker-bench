# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# figs_calibration.jl — quantile-calibration heatmaps: median absolute
# quantile error (averaged over coordinates, per-cell median over
# admissible seeds) at the five canonical levels of the protocol
# (2.5 %, 16 %, median, 84 %, 97.5 %), at the headline setting
# (d = 5, eggbox at d = 2, B = 5e5). The median and 97.5 % maps are
# embedded in the thesis results chapter; the remaining three levels
# complete the set in the results appendix.
#
# Usage:  julia --project=. experiments/tools/figs_calibration.jl
# =============================================================================

include(joinpath(@__DIR__, "..", "scripts", "03_plots.jl"))   # helpers; main() guarded
using DataFrames, CSV

const OUT  = joinpath(_ROOT, "experiments", "out")
const FIGS = joinpath(OUT, "figs")

function main()
    df = CSV.read(joinpath(OUT, "tables", "cells.csv"), DataFrame)
    drop_v5_excluded_rows!(df)
    for (metric, tag) in [(:QE_p025, "qe025"), (:QE_p160, "qe160"),
                          (:QE_p500, "qe500"), (:QE_p840, "qe840"),
                          (:QE_p975, "qe975")]
        fig = fig_summary_heatmap(df, metric; transform = :negative_log10,
                d = 5, B = 5e5)
        save_pdf(fig, fig_filename(family = :summary, problem = :all,
                  d = 5, B = 5e5, extra = "heatmap-" * tag); dir = FIGS)
    end
    @info "calibration heatmaps done (5 files)" FIGS
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# figs_headline.jl — regenerate ONLY the Chapter 8 headline figures
# (glyph consistency, dlogZ column set + exact medians,
# effect-size Wilcoxon redesign, 4-panel dim grid). Everything else in
# out/figs is untouched; run 03_plots.jl for a full rebuild.
# =============================================================================

using Pkg

const _ROOT = abspath(joinpath(@__DIR__, "..", ".."))
if Base.active_project() != joinpath(_ROOT, "Project.toml")
    Pkg.activate(_ROOT)
end

include(joinpath(_ROOT, "experiments", "src", "ExperimentsBase.jl"))
using .ExperimentsBase
using DataFrames, CSV

function main()
    cells = joinpath(_ROOT, "experiments", "out", "tables", "cells.csv")
    isfile(cells) || error("cells.csv not found at $cells")
    df = CSV.read(cells, DataFrame)
    drop_v5_excluded_rows!(df)
    figs_dir = joinpath(_ROOT, "experiments", "out", "figs")

    d, B = 5, 5e5

    # Headline heatmaps (Figs. 8.1, 8.2, 8.4).
    for (metric, transform, tag) in [(:eta_Nlike,       :log10,          "eta"),
                                     (:dlogZ,           :negative_log10, "dlogz"),
                                     (:W1_marginal_avg, :negative_log10, "W1")]
        fig = fig_summary_heatmap(df, metric; transform = transform, d = d, B = B)
        save_pdf(fig, fig_filename(family = :summary, problem = :all, d = d,
                   B = B, extra = "heatmap-" * tag); dir = figs_dir)
    end

    # Effect-size / significance heatmap (Fig. 8.5). Stars use the
    # audited full-family Holm adjustment from headline_tests.csv so
    # the figure agrees with the p_adj values quoted in the text.
    tests_csv = joinpath(_ROOT, "experiments", "out", "tables", "headline_tests.csv")
    tests = isfile(tests_csv) ? CSV.read(tests_csv, DataFrame) : nothing
    fig = fig_tests_heatmap(df; d = d, B = B, tests = tests)
    save_pdf(fig, fig_filename(family = :tests, problem = :all, d = d,
              B = B, alg = :mw, extra = "wilcoxon"); dir = figs_dir)

    # Dimensional grid at the central budget (Fig. 8.6b).
    fig = fig_dim_grid(df; B = 5e4)
    save_pdf(fig, fig_filename(family = :dim, problem = :all, d = nothing,
              B = 5e4, alg = :all, extra = "all"); dir = figs_dir)

    @info "headline figures regenerated" figs_dir
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end

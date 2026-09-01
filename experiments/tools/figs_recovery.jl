# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# Targeted follow-up to `figs_viz_recovery_scaling.jl`: regenerate only the
# two recovery figures (y-label hyphen fix). Run from the repository root:
#   julia --project=. -t 1 experiments/tools/figs_recovery.jl

using Pkg
const _ROOT = abspath(joinpath(@__DIR__, "..", ".."))
if Base.active_project() != joinpath(_ROOT, "Project.toml")
    Pkg.activate(_ROOT)
end

include(joinpath(_ROOT, "experiments", "src", "ExperimentsBase.jl"))
using .ExperimentsBase
using DataFrames, CSV

const OUT      = joinpath(_ROOT, "experiments", "out")
const FIGS_DIR = joinpath(OUT, "figs")
df = DataFrame(CSV.File(joinpath(OUT, "tables", "cells.csv")))
for prob in (:mridges, :mridges_spiky)
    fig = fig_recovery(df, prob; class = :narrow)
    save_pdf(fig, fig_filename(family = :recovery, problem = prob,
              d = HEADLINE_DIMENSION, B = :all, alg = :all,
              extra = "all"); dir = FIGS_DIR)
    println("recovery $prob  OK")
end

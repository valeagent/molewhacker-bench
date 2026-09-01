# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# figs_runconv_shell.jl — regenerate the shell/MW running-efficiency
# panel (degenerate-cell annotation fix).
include(joinpath(@__DIR__, "..", "scripts", "03_plots.jl"))
const OUT = joinpath(_ROOT, "experiments", "out")
runs = MethodResult[]
for seed in SEED_GRID
    dir = cell_dir(OUT, :shell, :mw, 5, 5e5, seed)
    isdir(dir) && isfile(joinpath(dir, "result.h5")) || continue
    mr = load_method_result(dir)
    size(mr.samples, 2) > 1 && push!(runs, mr)
end
fig = fig_runconv_panel(runs; algorithm = :mw, problem = :shell,
                        d = 5, B = 5e5, class = :narrow)
save_pdf(fig, fig_filename(family = :runconv, problem = :shell, d = 5,
           B = 5e5, alg = :mw, extra = "eta");
         dir = joinpath(OUT, "figs"))
@info "shell panel done"

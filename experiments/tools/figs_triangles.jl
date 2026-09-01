# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# figs_triangles.jl — regenerate ONLY the fifteen triangle plots embedded in
# thesis Appendix B (visible legend swatches,
# smaller tick labels, tighter panel gaps). Uses the same representative-seed
# selection as 03_plots.jl, so footer values are unchanged.
# =============================================================================

include(joinpath(@__DIR__, "..", "scripts", "03_plots.jl"))   # helpers only; main() is guarded

function replot_tri()
    out = joinpath(_ROOT, "experiments", "out")
    truth_dir = joinpath(out, "truth")
    runs_dir  = joinpath(out, "runs")
    figs_dir  = joinpath(out, "figs")

    jobs = [
        (:mvn,           :mw,   5), (:mvn,    :nuts, 5),
        (:banana,        :mw,   5), (:banana, :nuts, 5),
        (:funnel,        :mw,   5), (:funnel, :nuts, 5),
        (:mridges,       :nuts, 5), (:mridges, :ns,  5), (:mridges, :mw, 5),
        (:shell,         :mw,   5),
        (:mridges_spiky, :mw,   5), (:mridges_spiky, :mh, 5),
        (:mridges,       :mw,   2), (:mridges, :mw, 10),
        (:eggbox,        :mw,   2),
    ]
    for (prob, alg, d) in jobs
        truth = _load_truth(truth_dir, prob, d)
        mr = _median_seed_run(runs_dir, prob, alg, d, 5e5)
        if truth === nothing || mr === nothing
            @warn "Missing truth or run — skipped" prob alg d
            continue
        end
        fig = fig_tri(mr, truth)
        save_pdf(fig, fig_filename(family = :tri, problem = prob, d = d,
                   B = 5e5, alg = alg); dir = figs_dir)
    end
    @info "triangle plots regenerated" figs_dir
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(replot_tri())
end

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# 03_plots.jl — generate every figure of Catalogues A–D from cells.csv
# =============================================================================
#
# Inputs:
#   - experiments/out/tables/cells.csv              (from 02_aggregate.jl)
#   - experiments/out/truth/<problem>_d<d>.h5       (from 00_generate_truth.jl)
#   - experiments/out/runs/                          (used for triangle / diag)
#
# Output:
#   - experiments/out/figs/*.pdf                    (vector)
#   - experiments/out/figs/png/*.png                (raster mirror)
#   - experiments/out/figs/*.tex                    (LaTeX summary tables)
#
# Usage:
#   julia --project=. -t 1 experiments/scripts/03_plots.jl \
#       [--out experiments/out] [--cells path/to/cells.csv]
#       [--catalogues A,B,C,D]    [--quick]

using Pkg

const _ROOT = abspath(joinpath(@__DIR__, "..", ".."))
if Base.active_project() != joinpath(_ROOT, "Project.toml")
    Pkg.activate(_ROOT)
end

include(joinpath(_ROOT, "experiments", "src", "ExperimentsBase.jl"))
using .ExperimentsBase
using DataFrames, CSV, Printf, Random, Dates


function _parse_args(args::Vector{String})
    opts = Dict{String,Any}(
        "out" => joinpath(_ROOT, "experiments", "out"),
        "cells" => nothing,
        "truth_dir" => nothing,
        "runs_dir" => nothing,
        "figs_dir" => nothing,
        "catalogues" => "A,B,C,D,V3",
        "quick" => false,
        "with_cd" => false,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--out"; opts["out"] = args[i+1]; i += 2
        elseif a == "--cells"; opts["cells"] = args[i+1]; i += 2
        elseif a == "--truth_dir"; opts["truth_dir"] = args[i+1]; i += 2
        elseif a == "--runs_dir"; opts["runs_dir"] = args[i+1]; i += 2
        elseif a == "--figs_dir"; opts["figs_dir"] = args[i+1]; i += 2
        elseif a == "--catalogues"; opts["catalogues"] = args[i+1]; i += 2
        elseif a == "--quick"; opts["quick"] = true; i += 1
        elseif a == "--with-cd" || a == "--with_cd"; opts["with_cd"] = true; i += 1
        elseif a == "--help" || a == "-h"
            println("03_plots.jl [--out PATH] [--cells PATH] [--catalogues A,B,C,D,V3] [--quick] [--with-cd]")
            println()
            println("  --quick     skip the heaviest appendix figures: Catalogue B.5 corner plots")
            println("              (fig_tri) and the entire Catalogue C per-algorithm diagnostics.")
            println("              Headline figures (A, B.1-4 + B.6, D, V3) are still rendered.")
            println("  --with-cd   render the Friedman/Nemenyi critical-difference diagram")
            println("              (off by default since v3-lite; needs friedman_nemenyi.csv).")
            exit(0)
        else
            @warn "Unknown arg $a — ignored"; i += 1
        end
    end
    opts["cells"] === nothing && (opts["cells"] = joinpath(opts["out"], "tables", "cells.csv"))
    opts["truth_dir"] === nothing && (opts["truth_dir"] = joinpath(opts["out"], "truth"))
    opts["runs_dir"] === nothing && (opts["runs_dir"] = joinpath(opts["out"], "runs"))
    opts["figs_dir"] === nothing && (opts["figs_dir"] = joinpath(opts["out"], "figs"))
    return opts
end


# -----------------------------------------------------------------------------
# Configurations + truth-file helpers
# -----------------------------------------------------------------------------

_make_cfg(p::Symbol, d::Int) = p === :mvn ? make_config_mvn(d = d) :
                                p === :banana ? make_config_banana(d = d) :
                                p === :funnel ? make_config_funnel(d = d) :
                                p === :mridges ? make_config_mridges(d = d) :
                                p === :shell ? make_config_shell(d = d) :
                                p === :mridges_spiky ? make_config_mridges_spiky(d = d) :
                                p === :eggbox ? make_config_eggbox(d = d) :
                                nothing  # caller skips unknown problems

function _load_truth(truth_dir, problem::Symbol, d::Int)
    path = joinpath(truth_dir, @sprintf("%s_d%d.h5", String(problem), d))
    isfile(path) || return nothing
    return load_truth(path)
end

function _load_run(runs_dir, problem::Symbol, alg::Symbol, d::Int, B::Real, seed::Int)
    cell = cell_dir(joinpath(runs_dir, ".."), problem, alg, d, B, seed)
    # cell_dir already prepends "runs/", so step out one then back in
    cell = joinpath(runs_dir, basename(cell))
    isdir(cell) || return nothing
    isfile(joinpath(cell, "result.h5")) || return nothing
    return load_method_result(cell)
end

function _median_seed_run(runs_dir, problem::Symbol, alg::Symbol, d::Int, B::Real)
    candidates = MethodResult[]
    for seed in SEED_GRID
        mr = _load_run(runs_dir, problem, alg, d, B, seed)
        mr === nothing && continue
        push!(candidates, mr)
    end
    isempty(candidates) && return nothing
    # Pick the seed with median W1 if truth available; otherwise the first.
    return candidates[ceil(Int, length(candidates) / 2)]
end


# -----------------------------------------------------------------------------
# Catalogue dispatchers
# -----------------------------------------------------------------------------

function _do_catalogue_A(opts, df, dims_in_df)
    Random.seed!(20260601)
    set_pub_theme!()
    truth_dir = opts["truth_dir"]
    figs_dir = opts["figs_dir"]
    isdir(figs_dir) || mkpath(figs_dir)

    # ---------- A.1 — V5 §3a: 3-D surface render of every target at d = 2 ----
    # Filename convention: `viz__<problem>__d2__surface.pdf`. The figure
    # catalogue spells `mridges_spiky` as `mspiky` for this filename only;
    # honour that exactly so the thesis side does not have to rename anything.
    surface_alias = Dict{Symbol,Symbol}(:mridges_spiky => :mspiky)
    for prob in PROBLEM_NAMES
        cfg2 = _make_cfg(prob, 2)
        if cfg2 === nothing
            @info "Skipping V5 surface — no d=2 config implemented" problem = prob
            continue
        end
        try
            fig = fig_viz_surface_3d(cfg2)
            slug = String(get(surface_alias, prob, prob))
            save_pdf(fig, "viz__" * slug * "__d2__surface"; dir = figs_dir)
        catch err
            @warn "V5 surface failed" problem = prob exception = (err, catch_backtrace())
        end
    end

    # ---------- A.2 — per-problem composites at d = 5 (V5 §3b) ---------------
    truths = TruthSet[]
    cfgs = ProblemConfig[]
    for prob in PROBLEM_NAMES
        d = HEADLINE_DIMENSION
        cfg = _make_cfg(prob, d)
        if cfg === nothing
            @info "Skipping Catalogue A — no config implemented" problem = prob
            continue
        end
        truth = _load_truth(truth_dir, prob, d)
        if truth === nothing
            @warn "Truth missing — skipping Catalogue A for $prob d=$d"
            continue
        end
        push!(truths, truth); push!(cfgs, cfg)

        # V5 §3b — 2-D contour overview + marginals at d = 5.
        fig_overview = fig_viz_overview(cfg, truth)
        save_pdf(fig_overview, fig_filename(family = :viz, problem = prob,
                  d = d, extra = "overview"); dir = figs_dir)
        # Auxiliary V4 panels (kept for the appendix / supplementary repo).
        fig_density = fig_viz_density(cfg; truth = truth)
        save_pdf(fig_density, fig_filename(family = :viz, problem = prob,
                  d = d, extra = "density-1-2"); dir = figs_dir)
        fig_scatter = fig_viz_samples_truth(truth)
        save_png(fig_scatter, fig_filename(family = :viz, problem = prob,
                  d = d, extra = "samples-truth"); dir = figs_dir)
        fig_marginals = fig_viz_marginals(truth)
        save_pdf(fig_marginals, fig_filename(family = :viz, problem = prob,
                  d = d, extra = "marginals"); dir = figs_dir)
    end

    # Cross-benchmark gallery
    if length(cfgs) >= 2
        fig_gallery = fig_viz_gallery(cfgs, truths)
        save_pdf(fig_gallery, fig_filename(family = :viz, problem = :all,
                  d = HEADLINE_DIMENSION, extra = "gallery"); dir = figs_dir)
    end
end


function _do_catalogue_B(opts, df, dims_in_df)
    set_pub_theme!()
    figs_dir = opts["figs_dir"]
    isdir(figs_dir) || mkpath(figs_dir)
    runs_dir = opts["runs_dir"]
    truth_dir = opts["truth_dir"]
    for prob in PROBLEM_ORDER
        for d in dims_in_df
            df_pp = df[df.problem .== String(prob) .&& df.d .== d, :]
            isempty(df_pp) && continue

            # B.1 — convergence
            fig = fig_conv(df, prob, d)
            save_pdf(fig, fig_filename(family = :conv, problem = prob, d = d,
                       B = :all, alg = :all); dir = figs_dir)
            # B.2 — efficiency
            fig = fig_eff(df, prob, d)
            save_pdf(fig, fig_filename(family = :eff, problem = prob, d = d,
                       B = :all, alg = :all); dir = figs_dir)
            # B.3 — log-evidence
            fig = fig_logz(df, prob, d)
            save_pdf(fig, fig_filename(family = :logz, problem = prob, d = d,
                       B = :all, alg = :all); dir = figs_dir)
            # B.4 — quantile errors at largest budget
            fig = fig_qe(df, prob, d)
            save_pdf(fig, fig_filename(family = :qe, problem = prob, d = d,
                       B = 5e5, alg = :all); dir = figs_dir)
            # B.6 — Pareto per problem
            fig = fig_pareto_per_problem(df, prob, d)
            save_pdf(fig, fig_filename(family = :pareto, problem = prob, d = d,
                       B = 5e5, alg = :all); dir = figs_dir)
            # B.5 — corner plot for `mw` (body) + others (appendix)
            #       In --quick mode skip the appendix corner plots (very expensive).
            truth = _load_truth(truth_dir, prob, d)
            truth === nothing && continue
            quick = get(opts, "quick", false)
            algs_to_render = quick ? (:mw,) : ALG_ORDER
            for alg in algs_to_render
                mr = _median_seed_run(runs_dir, prob, alg, d, 5e5)
                mr === nothing && continue
                if size(mr.samples, 2) < 4
                    continue
                end
                fig = fig_tri(mr, truth)
                save_pdf(fig, fig_filename(family = :tri, problem = prob, d = d,
                           B = 5e5, alg = alg); dir = figs_dir)
            end
        end
    end
end


function _do_catalogue_C(opts, df, dims_in_df)
    set_pub_theme!()
    figs_dir = opts["figs_dir"]
    isdir(figs_dir) || mkpath(figs_dir)
    runs_dir = opts["runs_dir"]
    truth_dir = opts["truth_dir"]

    for prob in PROBLEM_ORDER
        for d in dims_in_df
            truth = _load_truth(truth_dir, prob, d)
            truth === nothing && continue

            # C.1 — MW iteration log
            mr_mw = _median_seed_run(runs_dir, prob, :mw, d, 5e5)
            if mr_mw !== nothing
                try
                    fig = fig_iter_mw(mr_mw)
                    save_pdf(fig, fig_filename(family = :iter, problem = prob,
                              d = d, B = 5e5, alg = :mw, extra = "neff-kl");
                              dir = figs_dir)
                catch err
                    @warn "iter_mw failed" prob d exception = err
                end
                # C.2 — mole map
                try
                    fig = fig_diag_mw_moles(mr_mw, truth)
                    save_pdf(fig, fig_filename(family = :diag, problem = prob,
                              d = d, B = 5e5, alg = :mw, extra = "moles");
                              dir = figs_dir)
                catch err
                    @warn "mw_moles failed" prob d exception = err
                end
            end

            # C.3 — NUTS divergence map (P2-11: skip if no divergent points stored)
            mr_nuts = _median_seed_run(runs_dir, prob, :nuts, d, 5e5)
            if mr_nuts !== nothing && size(mr_nuts.samples, 2) >= 2
                try
                    fig = fig_diag_nuts_div(mr_nuts, truth)
                    if fig !== nothing
                        save_pdf(fig, fig_filename(family = :diag, problem = prob,
                                  d = d, B = 5e5, alg = :nuts, extra = "divergences");
                                  dir = figs_dir)
                    else
                        @info "Skipping NUTS divergence figure (no divergent transitions stored)" prob d
                    end
                catch err
                    @warn "nuts_div failed" prob d exception = err
                end
            end

            # C.4 — MH trace + ACF
            mr_mh = _median_seed_run(runs_dir, prob, :mh, d, 5e5)
            if mr_mh !== nothing && size(mr_mh.samples, 2) >= 50
                try
                    fig = fig_diag_mh_trace(mr_mh)
                    save_pdf(fig, fig_filename(family = :diag, problem = prob,
                              d = d, B = 5e5, alg = :mh, extra = "trace");
                              dir = figs_dir)
                catch err
                    @warn "mh_trace failed" prob d exception = err
                end
            end

            # C.5 — NS live-points
            mr_ns = _median_seed_run(runs_dir, prob, :ns, d, 5e5)
            if mr_ns !== nothing
                try
                    fig = fig_diag_ns_live(mr_ns)
                    save_pdf(fig, fig_filename(family = :diag, problem = prob,
                              d = d, B = 5e5, alg = :ns, extra = "livepoints");
                              dir = figs_dir)
                catch err
                    @warn "ns_live failed" prob d exception = err
                end
            end

            # C.6 — IS weights
            mr_is = _median_seed_run(runs_dir, prob, :is, d, 5e5)
            if mr_is !== nothing
                try
                    fig = fig_diag_is_weights(mr_is)
                    save_pdf(fig, fig_filename(family = :diag, problem = prob,
                              d = d, B = 5e5, alg = :is, extra = "weights");
                              dir = figs_dir)
                catch err
                    @warn "is_weights failed" prob d exception = err
                end
            end
        end
    end
end


function _do_catalogue_D(opts, df, dims_in_df)
    set_pub_theme!()
    figs_dir = opts["figs_dir"]
    isdir(figs_dir) || mkpath(figs_dir)

    # D.1 — heatmap summaries (one per metric, per dim, per budget present)
    # V5 §2: four metrics — eta, W1, dlogZ, mode_recovery. Mode-recovery
    # is a raw rate in [0, 1] so we use the identity transform (no log).
    budgets_present = sort(unique(df.B))
    headline_B = maximum(budgets_present)
    metric_specs = [
        (:W1_marginal_avg, :negative_log10, "W1"),
        (:eta_Nlike,       :log10,          "eta"),
        (:dlogZ,           :negative_log10, "dlogz"),
    ]
    if hasproperty(df, :mode_recovery)
        push!(metric_specs, (:mode_recovery, :identity, "recovery"))
    end
    for d in dims_in_df
        for B in budgets_present
            for (metric, transform, tag) in metric_specs
                try
                    fig = fig_summary_heatmap(df, metric;
                                                  transform = transform,
                                                  d = d, B = B)
                    save_pdf(fig, fig_filename(family = :summary, problem = :all, d = d,
                               B = B, extra = "heatmap-" * tag); dir = figs_dir)
                    tex_path = joinpath(figs_dir,
                        fig_filename(family = :summary, problem = :all, d = d,
                                      B = B, extra = "heatmap-" * tag) * ".tex")
                    write_summary_table(df, metric, tex_path; d = d, B = B)
                catch err
                    @warn "heatmap failed" metric d B exception = err
                end
            end
        end
    end

    # D.2 — scaling plots (only meaningful for the 2 scaling problems if multi-d data exist)
    multi_d = unique(df.d)
    if length(multi_d) >= 2
        for prob in SCALING_PROBLEMS
            sub = df[df.problem .== String(prob), :]
            length(unique(sub.d)) >= 2 || continue
            try
                # V8-FIX-C3: the plot shows a single budget, so the
                # filename says so (the old `Ball` token claimed all
                # budgets while fig_scaling hard-coded B = 5e5).
                B_scal = 5e5
                # Rendered at half-column width because the
                # thesis embeds it as a subfigure next to the dim grid.
                fig = fig_scaling(df, prob; B = B_scal, class = :narrow)
                save_pdf(fig, fig_filename(family = :scaling, problem = prob,
                          d = nothing, B = B_scal, alg = :all,
                          extra = "W1-vs-d"); dir = figs_dir)
            catch err
                @warn "scaling fig failed" prob exception = err
            end
        end
    end

    # D.3 — grand Pareto plot (one per dim, at the largest budget present)
    for d in dims_in_df
        try
            fig = fig_pareto_grand(df, d; B = headline_B)
            save_pdf(fig, fig_filename(family = :pareto, problem = :all, d = d,
                      B = headline_B, alg = :all, extra = "time-vs-eta"); dir = figs_dir)
        catch err
            @warn "pareto grand failed" d exception = err
        end
    end
end


# -----------------------------------------------------------------------------
# V3 figures (Protocol §10.17). Run only if the relevant data exists.
# -----------------------------------------------------------------------------
function _do_catalogue_V3(opts, df, dims_in_df)
    set_pub_theme!()
    figs_dir = opts["figs_dir"]
    isdir(figs_dir) || mkpath(figs_dir)

    budgets_present = sort(unique(df.B))
    headline_B = maximum(budgets_present)

    # V3-F1 — hardness sweep (only if mridges_spiky cells exist).
    if any(df.problem .== "mridges_spiky")
        for B in budgets_present
            try
                fig = fig_hardness_mridges_spiky(df; B = B)
                save_pdf(fig, fig_filename(family = :hardness,
                                              problem = :mridges_spiky,
                                              d = HEADLINE_DIMENSION,
                                              B = B, alg = :all,
                                              extra = "all"); dir = figs_dir)
            catch err
                @warn "hardness fig failed" B exception = err
            end
        end
    end

    # V3-F2 — recovery curves for every multimodal problem with the metric.
    if hasproperty(df, :mode_recovery)
        for prob in (:mridges, :mridges_spiky, :eggbox)
            any(df.problem .== String(prob)) || continue
            try
                # Half-column width (side-by-side pair in
                # the thesis results section).
                fig = fig_recovery(df, prob; class = :narrow)
                save_pdf(fig, fig_filename(family = :recovery, problem = prob,
                          d = HEADLINE_DIMENSION, B = :all, alg = :all,
                          extra = "all"); dir = figs_dir)
            catch err
                @warn "recovery fig failed" prob exception = err
            end
        end
    end

    # V3-F4 — dimensional grid (only if multiple dims exist).
    if length(dims_in_df) >= 2
        for B in budgets_present
            try
                fig = fig_dim_grid(df; B = B)
                save_pdf(fig, fig_filename(family = :dim, problem = :all,
                          d = nothing, B = B, alg = :all,
                          extra = "all"); dir = figs_dir)
            catch err
                @warn "dim_grid fig failed" B exception = err
            end
        end
    end

    # V3-F11 — effect-size heatmap (Cliff's δ + Holm-adjusted Wilcoxon
    # stars). Prefer the audited full-family p_adj from
    # headline_tests.csv so the figure matches the tabulated tests.
    tests_csv = joinpath(opts["out"], "tables", "headline_tests.csv")
    tests_df = isfile(tests_csv) ? CSV.read(tests_csv, DataFrame) : nothing
    for d in dims_in_df, B in budgets_present
        try
            fig = fig_tests_heatmap(df; d = d, B = B, tests = tests_df)
            save_pdf(fig, fig_filename(family = :tests, problem = :all,
                      d = d, B = B, alg = :mw, extra = "wilcoxon");
                     dir = figs_dir)
        catch err
            @warn "tests heatmap failed" d B exception = err
        end
    end

    # V3-CD — Friedman/Nemenyi critical-difference diagram.
    # Off by default since v3-lite; opt-in via --with-cd. The headline
    # tests (Wilcoxon + Cliff's δ + Holm) already cover the per-cell
    # MW-vs-comparator claim and are typically what reviewers ask for.
    if !get(opts, "with_cd", false)
        @info "Skipping CD diagram (use --with-cd to enable)"
        return nothing
    end
    try
        # Build score matrix from cells.csv (W1, lower = better).
        rows = NamedTuple[]
        for problem in unique(df.problem),
            d in unique(df[df.problem .== problem, :d]),
            B in unique(df[df.problem .== problem .&& df.d .== d, :B])
            sub = df[df.problem .== problem .&& df.d .== d .&& df.B .== Float64(B), :]
            seeds = unique(sub.seed)
            for s in seeds
                row = fill(NaN, length(ALG_ORDER))
                ok = true
                for (j, alg) in enumerate(ALG_ORDER)
                    sa = sub[sub.algorithm .== String(alg) .&& sub.seed .== s, :]
                    if isempty(sa)
                        ok = false; break
                    end
                    notes_str = String(coalesce(sa.notes[1], ""))
                    if occursin("RHAT-FAIL", notes_str) ||
                       occursin("FAILED-SANITY", notes_str) ||
                       occursin("BUDGET-VIOLATION", notes_str)
                        ok = false; break
                    end
                    v = sa[1, :W1_marginal_avg]
                    if v === missing || !isfinite(Float64(v))
                        ok = false; break
                    end
                    row[j] = Float64(v)
                end
                ok && push!(rows, NamedTuple{Tuple(Symbol("a$i") for i in 1:length(ALG_ORDER))}(row))
            end
        end
        if length(rows) >= 2
            score_mat = Matrix{Float64}(undef, length(rows), length(ALG_ORDER))
            for (i, r) in enumerate(rows), (j, _) in enumerate(ALG_ORDER)
                score_mat[i, j] = r[j]
            end
            fig = fig_cd_diagram(score_mat,
                                    [ALG_LABEL[a] for a in ALG_ORDER])
            save_pdf(fig, fig_filename(family = :tests, problem = :all,
                      d = HEADLINE_DIMENSION, B = headline_B, alg = :all,
                      extra = "cd-diagram"); dir = figs_dir)
        else
            @info "Skipping CD diagram (insufficient complete blocks)"
        end
    catch err
        @warn "CD diagram failed" exception = err
    end
    return nothing
end


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

function main(args = ARGS)
    opts = _parse_args(collect(String, args))
    isfile(opts["cells"]) || error("cells.csv not found at $(opts["cells"])")
    df = CSV.read(opts["cells"], DataFrame)
    @info "Loaded cells.csv" rows = nrow(df) cols = ncol(df)

    # V5 §2 — drop the two stray `eggbox / d=5 / seed=11` cells from
    # every downstream computation (in-memory only; the on-disk
    # cells.csv is left untouched so the audit trail still shows
    # them).
    n_dropped = drop_v5_excluded_rows!(df)
    if n_dropped > 0
        @info "Dropped V5-excluded rows from in-memory dataframe" n = n_dropped
    end

    catalogues = Set(uppercase.(strip.(split(opts["catalogues"], ","))))
    dims_in_df = sort(unique(df.d))

    println("Provenance: ", provenance_string(cells_csv_path = opts["cells"]))

    if "A" in catalogues
        @info "Catalogue A — benchmark visualisations"
        _do_catalogue_A(opts, df, dims_in_df)
    end
    if "B" in catalogues
        @info "Catalogue B — cross-algorithm comparison"
        _do_catalogue_B(opts, df, dims_in_df)
    end
    if "C" in catalogues
        if get(opts, "quick", false)
            @info "Catalogue C — skipped (--quick mode)"
        else
            @info "Catalogue C — algorithm-specific diagnostics"
            _do_catalogue_C(opts, df, dims_in_df)
        end
    end
    if "D" in catalogues
        @info "Catalogue D — aggregated summaries"
        _do_catalogue_D(opts, df, dims_in_df)
    end
    if "V3" in catalogues || "ALL" in catalogues
        @info "Catalogue V3 — hardness, recovery, dim sweep, statistical tests"
        _do_catalogue_V3(opts, df, dims_in_df)
    end

    @info "All requested catalogues rendered" out = opts["figs_dir"]
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end

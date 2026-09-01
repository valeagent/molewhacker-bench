# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# 02b_tests.jl — emit headline_tests.csv and friedman_nemenyi.csv (V3-S-3/-S-4)
# =============================================================================
#
# Per Protocol §11.4 (rev May 2026), runs:
#
# 1. **Pairwise paired Wilcoxon signed-rank tests** "MW < comparator"
#    with Cliff's δ effect size and Holm-Bonferroni step-down
#    correction (§11.4.1) for every (problem, d, B) cell on
#    `W1_marginal_avg`, `SWD`, `mmd_rbf` (when present), and
#    `mode_recovery` (when present); writes
#    `experiments/out/tables/headline_tests.csv`.
#
# 2. **Friedman χ² + Nemenyi critical difference** across the
#    (problem, d, B) blocks for every algorithm (§11.4.2); writes
#    `experiments/out/tables/friedman_nemenyi.csv` with one row per
#    metric (and a CD-diagram is rendered separately by `03_plots.jl`).

using Pkg

const _ROOT = abspath(joinpath(@__DIR__, "..", ".."))
if Base.active_project() != joinpath(_ROOT, "Project.toml")
    Pkg.activate(_ROOT)
end

include(joinpath(_ROOT, "experiments", "src", "ExperimentsBase.jl"))
using .ExperimentsBase
using DataFrames, CSV, Printf, SHA, Statistics, StatsBase


function _parse_args(args::Vector{String})
    opts = Dict{String,Any}(
        "csv" => joinpath(_ROOT, "experiments", "out", "tables", "cells.csv"),
        "out_dir" => joinpath(_ROOT, "experiments", "out", "tables"),
        "with_cd" => false,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--csv"; opts["csv"] = args[i+1]; i += 2
        elseif a == "--out_dir"; opts["out_dir"] = args[i+1]; i += 2
        elseif a == "--with-cd" || a == "--with_cd"; opts["with_cd"] = true; i += 1
        elseif a == "--help" || a == "-h"
            println("02b_tests.jl [--csv PATH] [--out_dir PATH] [--with-cd]")
            println()
            println("  Always emits headline_tests.csv (Wilcoxon + Cliff's δ + Holm-Bonferroni).")
            println("  --with-cd   additionally emits friedman_nemenyi.csv for the CD diagram.")
            println("              (Off by default since v3-lite; the headline tests already")
            println("               cover the per-cell MW-vs-comparator claim.)")
            exit(0)
        else
            @warn "Unknown arg $a — ignored"; i += 1
        end
    end
    return opts
end

# True iff `notes` flags a real failure (after V2-FIX-1 the
# `BUDGET-VIOLATION` token only appears for non-exempt overshoots).
function _has_real_failure(notes_str::AbstractString)
    occursin("RHAT-FAIL", notes_str) ||
        occursin("FAILED-SANITY", notes_str) ||
        occursin("BUDGET-VIOLATION", notes_str)
end

# Pull the paired (mw, comparator) values for a metric on a single
# cell key.
function _paired_values(df::DataFrame, prob::AbstractString, d::Integer,
                            B::Real, alg::AbstractString, metric::Symbol)
    mw = df[df.problem .== prob .&& df.d .== d .&& df.B .== Float64(B) .&&
                df.algorithm .== "mw", :]
    cm = df[df.problem .== prob .&& df.d .== d .&& df.B .== Float64(B) .&&
                df.algorithm .== alg, :]
    common_seeds = sort!(collect(intersect(Set(mw.seed), Set(cm.seed))))
    a = Float64[]
    b = Float64[]
    for s in common_seeds
        mw_row = mw[mw.seed .== s, :]
        cm_row = cm[cm.seed .== s, :]
        isempty(mw_row) && continue
        isempty(cm_row) && continue
        if hasproperty(mw_row, :notes) && _has_real_failure(String(coalesce(mw_row.notes[1], "")))
            continue
        end
        if hasproperty(cm_row, :notes) && _has_real_failure(String(coalesce(cm_row.notes[1], "")))
            continue
        end
        mv = mw_row[1, metric]; cv = cm_row[1, metric]
        (mv === missing || cv === missing) && continue
        m = Float64(mv); c = Float64(cv)
        (isfinite(m) && isfinite(c)) || continue
        push!(a, m)
        push!(b, c)
    end
    return a, b
end

function _maybe_metric_in_df(df::DataFrame, m::Symbol)
    return hasproperty(df, m) && any(isfinite, Vector{Float64}(df[!, m]))
end


function run_headline_tests(df::DataFrame; tail::Symbol = :left)
    rows = NamedTuple[]
    metrics = [:W1_marginal_avg, :SWD]
    if _maybe_metric_in_df(df, :mmd_rbf)
        push!(metrics, :mmd_rbf)
    end
    if _maybe_metric_in_df(df, :mode_recovery)
        # For mode_recovery, larger is better; the test alternative
        # becomes "MW > comparator" → tail = :right.
        push!(metrics, :mode_recovery)
    end
    keys = unique(eachrow(select(df, [:problem, :d, :B])))
    comparators = [a for a in ALG_ORDER if a != :mw]
    raw_p = Float64[]
    raw_idx = Int[]
    bookkeeping = Vector{NamedTuple}(undef, 0)
    for (problem, d, B) in unique([(r.problem, r.d, r.B) for r in eachrow(df)])
        for metric in metrics
            metric_tail = metric === :mode_recovery ? :right : tail
            for alg in comparators
                a, b = _paired_values(df, String(problem), Int(d),
                                          Float64(B), String(alg), metric)
                n = length(a)
                if n < 3
                    continue
                end
                _, p, n_used = wilcoxon_signed_rank(a, b; tail = metric_tail)
                δ = cliffs_delta(a, b)
                push!(rows, (
                    problem = problem, d = d, B = B,
                    metric = String(metric),
                    comparator = String(alg),
                    n_pairs = n_used,
                    median_mw = median(a),
                    median_cmp = median(b),
                    diff_median = median(a .- b),
                    cliffs_delta = δ,
                    p_raw = p,
                    p_adj = NaN,
                    tail = String(metric_tail),
                ))
                push!(raw_p, p)
                push!(raw_idx, length(rows))
            end
        end
    end
    if !isempty(raw_p)
        adj = holm_bonferroni(raw_p)
        for (k, idx) in enumerate(raw_idx)
            r = rows[idx]
            rows[idx] = merge(r, (p_adj = adj[k],))
        end
    end
    return DataFrame(rows)
end

function run_friedman_nemenyi(df::DataFrame)
    metrics = [:W1_marginal_avg, :SWD]
    if _maybe_metric_in_df(df, :mmd_rbf)
        push!(metrics, :mmd_rbf)
    end
    if _maybe_metric_in_df(df, :mode_recovery)
        push!(metrics, :mode_recovery)
    end
    rows = NamedTuple[]
    for metric in metrics
        # V8-FIX-B1: blocks are (problem, d, B) CELLS with the per-algorithm
        # SEED-MEDIAN score — not (problem, d, B, seed) tuples. Twenty seeds
        # of the same cell are repeated measurements of one configuration,
        # not independent datasets in Demšar's framework; treating them as
        # blocks inflated n_blocks ~20× and shrank the Nemenyi critical
        # difference by ≈ √20 ≈ 4.5×. A block is dropped when any algorithm
        # has no surviving finite seed value (flagged/missing); the dropped
        # count is DISCLOSED in the output because requiring all algorithms
        # finite conditions the test on "cells where every algorithm runs"
        # (e.g. budgets where NUTS is feasible) — a selection the reader
        # must see.
        score_rows = Matrix{Float64}(undef, 0, length(ALG_ORDER))
        n_dropped = 0
        for problem in unique(df.problem),
            d in unique(df[df.problem .== problem, :d]),
            B in unique(df[df.problem .== problem .&& df.d .== d, :B])
            sub = df[df.problem .== problem .&& df.d .== d .&& df.B .== Float64(B), :]
            row = fill(NaN, length(ALG_ORDER))
            ok = true
            for (j, alg) in enumerate(ALG_ORDER)
                sa = sub[sub.algorithm .== String(alg), :]
                vals = Float64[]
                for r in eachrow(sa)
                    if hasproperty(sa, :notes) &&
                       _has_real_failure(String(coalesce(r.notes, "")))
                        continue
                    end
                    v = r[metric]
                    (v === missing || !isfinite(Float64(v))) && continue
                    push!(vals, Float64(v))
                end
                if isempty(vals)
                    ok = false; break
                end
                row[j] = median(vals)
            end
            if !ok
                n_dropped += 1
                continue
            end
            # For "higher is better" metrics, negate so Friedman
            # ranks correctly (lower is better).
            if metric === :mode_recovery
                row .= -row
            end
            score_rows = vcat(score_rows, row')
        end
        n_blocks = size(score_rows, 1)
        if n_blocks < 2
            push!(rows, (
                metric = String(metric),
                k = length(ALG_ORDER),
                n_blocks = n_blocks,
                n_blocks_dropped = n_dropped,
                chi2 = NaN, df = length(ALG_ORDER) - 1, p_value = NaN,
                CD_005 = NaN, CD_001 = NaN,
                avg_ranks = "", alg_order = join([String(a) for a in ALG_ORDER], ","),
            ))
            continue
        end
        χ2, dof, p = friedman_chi2(score_rows)
        cd05 = nemenyi_critical_difference(length(ALG_ORDER), n_blocks; α = 0.05)
        cd01 = nemenyi_critical_difference(length(ALG_ORDER), n_blocks; α = 0.01)
        # Average rank per algorithm (lower = better, on the (negated
        # if needed) scores).
        ranks_mat = Matrix{Float64}(undef, n_blocks, length(ALG_ORDER))
        for i in 1:n_blocks
            row = collect(Float64, score_rows[i, :])
            perm = sortperm(row)
            rk = Vector{Float64}(undef, length(ALG_ORDER))
            idx = 1
            while idx <= length(ALG_ORDER)
                j = idx
                while j < length(ALG_ORDER) && row[perm[j + 1]] == row[perm[idx]]
                    j += 1
                end
                avg = (idx + j) / 2
                for s in idx:j
                    rk[perm[s]] = avg
                end
                idx = j + 1
            end
            ranks_mat[i, :] = rk
        end
        avg_ranks = vec(StatsBase.mean(ranks_mat; dims = 1))
        push!(rows, (
            metric = String(metric),
            k = length(ALG_ORDER),
            n_blocks = n_blocks,
            n_blocks_dropped = n_dropped,
            chi2 = χ2, df = dof, p_value = p,
            CD_005 = cd05, CD_001 = cd01,
            avg_ranks = join([@sprintf("%.3f", r) for r in avg_ranks], ","),
            alg_order = join([String(a) for a in ALG_ORDER], ","),
        ))
    end
    return DataFrame(rows)
end


function main(args = ARGS)
    opts = _parse_args(collect(String, args))
    csv_path = opts["csv"]
    isfile(csv_path) || error("cells.csv not found at $csv_path")
    df = CSV.read(csv_path, DataFrame)
    @info "Loaded cells.csv" path = csv_path n_rows = nrow(df)

    headline_path = joinpath(opts["out_dir"], "headline_tests.csv")
    friedman_path = joinpath(opts["out_dir"], "friedman_nemenyi.csv")
    isdir(dirname(headline_path)) || mkpath(dirname(headline_path))

    headline = run_headline_tests(df)
    CSV.write(headline_path, headline)
    @info "headline_tests.csv written" path = headline_path n_rows = nrow(headline)

    if opts["with_cd"]
        fried = run_friedman_nemenyi(df)
        CSV.write(friedman_path, fried)
        @info "friedman_nemenyi.csv written" path = friedman_path n_rows = nrow(fried)
    else
        @info "Skipping Friedman/Nemenyi (use --with-cd to enable)"
    end

    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end

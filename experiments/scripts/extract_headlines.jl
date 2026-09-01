# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# extract_headlines.jl — emit median-per-cell summary for DELIVERABLE.md.
using Pkg
const _ROOT = abspath(joinpath(@__DIR__, "..", ".."))
if Base.active_project() != joinpath(_ROOT, "Project.toml")
    Pkg.activate(_ROOT)
end
using DataFrames, CSV, Statistics, Printf

const _CSV = joinpath(_ROOT, "experiments", "out", "tables", "cells.csv")
df = CSV.read(_CSV, DataFrame)

# The headline panel is fixed at d=5 (the dimension-scaling
# study spans d∈{2,5,10}, but the headline stays at d=5). Restrict to it so the
# per-(problem,algorithm,B) medians below are NOT silently averaged across
# dimensions — scaling rows live in cells.csv for the dim__*/scaling__* figures.
const _HEADLINE_D = 5
df = df[df.d .== _HEADLINE_D, :]

# V8-FIX-A8: apply the SAME cell-exclusion rule as the statistical tests
# (02b_tests.jl `_has_real_failure`), so the headline medians and the
# significance tests are computed on identical cell populations. Also
# drop `budget_infeasible` placeholder rows entirely — their metric
# columns are NaN by design and their cost columns document the probe
# cost, not real spending, so they must not enter any median.
_has_real_failure(notes_str::AbstractString) =
    occursin("RHAT-FAIL", notes_str) ||
    occursin("FAILED-SANITY", notes_str) ||
    occursin("BUDGET-VIOLATION", notes_str)

n_before = nrow(df)
if hasproperty(df, :terminated_by)
    df = df[[String(coalesce(t, "")) != "budget_infeasible"
             for t in df.terminated_by], :]
end
n_infeasible = n_before - nrow(df)
if hasproperty(df, :notes)
    df = df[[!_has_real_failure(String(coalesce(n, ""))) for n in df.notes], :]
end
n_flagged = n_before - n_infeasible - nrow(df)
println("# Cell exclusion (V8-FIX-A8): dropped $(n_infeasible) budget-infeasible " *
        "and $(n_flagged) flagged (RHAT-FAIL / BUDGET-VIOLATION / FAILED-SANITY) " *
        "of $(n_before) d=$(_HEADLINE_D) rows\n")

# Median per (problem, algorithm, B) across seeds (at d=5)
key_metrics = [:Nlike_used, :W1_marginal_avg, :SWD, :dlogZ, :Neff,
               :eta_Nlike, :Rhat_max]

agg = combine(groupby(df, [:problem, :algorithm, :B])) do sub
    out = NamedTuple()
    for m in key_metrics
        col = sub[!, m]
        vals = filter(isfinite, [Float64(x) for x in skipmissing(col)])
        med = isempty(vals) ? NaN : median(vals)
        out = merge(out, (; m => med))
    end
    DataFrame([out])
end

# Print summary tables in markdown
println("# Median across seeds at d=$(_HEADLINE_D) (headline panel)\n")
for B in unique(df.B)
    println("## Budget B = $(Int(B))\n")
    println("| Problem | Algorithm | Nlike_used | W1 | dlogZ | η | Neff | Rhat_max |")
    println("|---------|-----------|-----------|-------|--------|--------|--------|---------|")
    for r in eachrow(filter(row -> row.B == B, agg))
        @printf("| %s | %s | %.0f | %.4f | %.4f | %.4g | %.1f | %s |\n",
                r.problem, r.algorithm, r.Nlike_used, r.W1_marginal_avg,
                r.dlogZ, r.eta_Nlike, r.Neff,
                isnan(r.Rhat_max) ? "—" : @sprintf("%.3f", r.Rhat_max))
    end
    println()
end

# Save the aggregated medians as CSV for figures / supplementary
out_path = joinpath(_ROOT, "experiments", "out", "tables", "headline_medians.csv")
CSV.write(out_path, agg)
println("Wrote $(out_path)")

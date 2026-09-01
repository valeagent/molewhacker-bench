# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
using Pkg
const _ROOT = abspath(joinpath(@__DIR__, "..", ".."))
Base.active_project() == joinpath(_ROOT, "Project.toml") || Pkg.activate(_ROOT)
using CSV, DataFrames, Statistics, Printf

df = CSV.read(joinpath(_ROOT, "experiments", "out", "tables", "cells.csv"), DataFrame)
println("rows=", nrow(df), " cols=", ncol(df))
println("budgets=", sort(unique(df.B)))
println("algs=", unique(df.algorithm))
println("problems=", unique(df.problem))

# Replace "NaN" string with actual NaN if needed
for col in (:W1_marginal_avg, :Neff, :Nlike_used, :dlogZ, :SWD, :KL_cube)
    if eltype(df[!, col]) <: AbstractString
        df[!, col] = parse.(Float64, df[!, col])
    end
end

for B in sort(unique(df.B))
    println()
    println("============== B = ", B, " ==============")
    sub = df[df.B .== B, :]
    g = combine(groupby(sub, [:problem, :algorithm])) do d
        kl_vals = "KL_cube" in names(d) ?
                    filter(x -> x !== missing && !(x isa Number ? isnan(x) : false),
                           collect(skipmissing(d.KL_cube))) : Float64[]
        (median_W1 = median(filter(!isnan, d.W1_marginal_avg)),
         median_Neff = median(filter(!isnan, d.Neff)),
         median_used = median(filter(!isnan, d.Nlike_used)),
         median_KL = isempty(kl_vals) ? NaN : median(Float64.(kl_vals)))
    end
    sort!(g, [:problem, :algorithm])
    for (problem, sub_p) in pairs(groupby(g, :problem))
        println("\n  ", problem.problem, ":")
        for row in eachrow(sub_p)
            @printf("    %-5s  W1=%-6.3g  Neff=%-7.0f  used=%-8.0f  KL=%s\n",
                    row.algorithm, row.median_W1, row.median_Neff,
                    row.median_used,
                    isnan(row.median_KL) ? "n/a" : @sprintf("%.3g", row.median_KL))
        end
    end
end

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# 06_funnel_logz_check.jl — V3 dlogZ persistence diagnostic (Protocol §11.5)
# =============================================================================
#
# Early runs flagged a persistent dlogZ offset for IS, NS, and MW on
# the funnel target. The protocol asks us to verify
# whether the offset is (a) a numerical bias in the truth's logZ
# (1-D quadrature truncation), (b) sample-size noise, or (c) an
# algorithmic bias.
#
# This script:
#   1. Computes the funnel truth's logZ at *three* quadrature
#      resolutions: the v2 default (`n_grid = 4096`), an 8× refined
#      grid (`n_grid = 32 768`), and an adaptive `QuadGK` baseline.
#      The three numbers should agree to better than 1e-6 if the
#      offset is not a quadrature artefact.
#   2. Loads every existing funnel result.h5 (IS, NS, MW × seeds × B)
#      from `experiments/out/runs` and recomputes `dlogZ` against
#      the `QuadGK` reference; writes the per-cell offset and the
#      per-(alg, B) median to `funnel_logz_check.csv`.
#
# Usage:
#   julia --project=. experiments/scripts/06_funnel_logz_check.jl \
#       [--out experiments/out] [--d 5]

using Pkg

const _ROOT = abspath(joinpath(@__DIR__, "..", ".."))
if Base.active_project() != joinpath(_ROOT, "Project.toml")
    Pkg.activate(_ROOT)
end

include(joinpath(_ROOT, "experiments", "src", "ExperimentsBase.jl"))
using .ExperimentsBase
using DataFrames, CSV, Printf, Statistics, QuadGK


function _parse_args(args::Vector{String})
    opts = Dict{String,Any}(
        "out" => joinpath(_ROOT, "experiments", "out"),
        "d" => HEADLINE_DIMENSION,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--out"; opts["out"] = args[i+1]; i += 2
        elseif a == "--d"; opts["d"] = parse(Int, args[i+1]); i += 2
        elseif a == "--help" || a == "-h"
            println("06_funnel_logz_check.jl [--out PATH] [--d N]"); exit(0)
        else
            @warn "Unknown arg $a — ignored"; i += 1
        end
    end
    return opts
end


function _funnel_logZ_quadrature(cfg::ConfigFunnel; n_grid::Integer = 4096)
    L = cfg.L
    d = cfg.d
    grid = collect(range(-L, L; length = n_grid))
    log_p1 = [ExperimentsBase._log_marginal_theta1(t, cfg) for t in grid]
    log_max = maximum(log_p1)
    p = exp.(log_p1 .- log_max)
    Z1 = 0.0
    @inbounds for i in 2:n_grid
        Z1 += 0.5 * (p[i] + p[i - 1]) * (grid[i] - grid[i - 1])
    end
    return log_max + log(Z1) - d * log(2 * L)
end


function _funnel_logZ_quadgk(cfg::ConfigFunnel; reltol::Real = 1e-12)
    d = cfg.d
    L = cfg.L
    # Stable integrand with peak normalisation.
    L_grid = collect(range(-L, L; length = 8192))
    log_p1_grid = [ExperimentsBase._log_marginal_theta1(t, cfg) for t in L_grid]
    log_max = maximum(log_p1_grid)
    integrand(t) = exp(ExperimentsBase._log_marginal_theta1(t, cfg) - log_max)
    Z1, _ = QuadGK.quadgk(integrand, -L, L; rtol = reltol, atol = 0.0)
    return log_max + log(Z1) - d * log(2 * L)
end


function _per_cell_offsets(opts, ref_logZ)
    runs_dir = joinpath(opts["out"], "runs")
    isdir(runs_dir) || error("No runs directory at $runs_dir")
    out_rows = NamedTuple[]
    for cell in readdir(runs_dir; join = true)
        isdir(cell) || continue
        rh5 = joinpath(cell, "result.h5")
        isfile(rh5) || continue
        local mr
        try
            mr = load_method_result(cell)
        catch err
            @warn "Skipping unreadable cell" cell exception = err
            continue
        end
        mr.problem === :funnel || continue
        mr.algorithm in (:is, :ns, :mw) || continue
        if mr.logZ_estimate === missing || mr.logZ_estimate === nothing
            continue
        end
        push!(out_rows, (
            algorithm = String(mr.algorithm),
            d = mr.d,
            B = mr.B,
            seed = mr.seed,
            logZ_alg = Float64(mr.logZ_estimate),
            logZ_truth = ref_logZ,
            offset = Float64(mr.logZ_estimate) - ref_logZ,
        ))
    end
    return DataFrame(out_rows)
end


function main(args = ARGS)
    opts = _parse_args(collect(String, args))
    cfg = make_config_funnel(d = opts["d"])
    @info "Funnel logZ — three quadrature resolutions" d = cfg.d L = cfg.L
    z_4096   = _funnel_logZ_quadrature(cfg; n_grid = 4096)
    z_32768  = _funnel_logZ_quadrature(cfg; n_grid = 32_768)
    z_quadgk = _funnel_logZ_quadgk(cfg)
    println(@sprintf("logZ (n_grid =  4096) = %.10f", z_4096))
    println(@sprintf("logZ (n_grid = 32768) = %.10f", z_32768))
    println(@sprintf("logZ (QuadGK rtol=1e-12) = %.10f", z_quadgk))
    println(@sprintf("Δ(4096 → 32768)   = %+.3e", z_32768 - z_4096))
    println(@sprintf("Δ(4096 → QuadGK)  = %+.3e", z_quadgk - z_4096))

    @info "Per-cell dlogZ offsets vs. QuadGK reference"
    rows = _per_cell_offsets(opts, z_quadgk)
    if isempty(rows)
        @warn "No funnel cells found" runs_dir = joinpath(opts["out"], "runs")
        return 1
    end
    by_grp = combine(groupby(rows, [:algorithm, :B])) do s
        DataFrame(
            n = nrow(s),
            median_offset = median(s.offset),
            mean_offset = mean(s.offset),
            std_offset = std(s.offset),
        )
    end
    sort!(by_grp, [:algorithm, :B])
    println("\nPer-(algorithm, B) median dlogZ offset:")
    println(by_grp)

    out_csv = joinpath(opts["out"], "tables", "funnel_logz_check.csv")
    isdir(dirname(out_csv)) || mkpath(dirname(out_csv))
    CSV.write(out_csv, rows)
    @info "Per-cell offset table written" path = out_csv
    summary_csv = joinpath(opts["out"], "tables", "funnel_logz_summary.csv")
    CSV.write(summary_csv, by_grp)
    @info "Summary written" path = summary_csv

    # Append the three quadrature values for record-keeping.
    open(joinpath(opts["out"], "tables", "funnel_logz_quadrature.txt"), "w") do io
        println(io, "# Funnel logZ — quadrature resolution audit (Protocol §11.5)")
        println(io, "d = $(cfg.d)  L = $(cfg.L)")
        println(io, @sprintf("logZ (n_grid =  4096)    = %.12f", z_4096))
        println(io, @sprintf("logZ (n_grid = 32768)    = %.12f", z_32768))
        println(io, @sprintf("logZ (QuadGK rtol=1e-12) = %.12f", z_quadgk))
        println(io, @sprintf("Δ(4096 -> 32768)         = %+.6e", z_32768 - z_4096))
        println(io, @sprintf("Δ(4096 -> QuadGK)        = %+.6e", z_quadgk - z_4096))
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# primary_fresh_view.jl — canonical PRIMARY results view (revision of
# 27 September 2026).
#
# The thesis reports MoleWhacker (MW) through the independent draw from the
# frozen final mixture that uses the remaining budget ("fresh" / final
# sample, estimator fresh_v1_2026-09-25). This tool joins the EXISTING saved
# outputs into one canonical dataset from which every primary table and
# summary figure is regenerated. It performs no sampling and no target
# evaluation; all inputs are read-only.
#
# Inputs (read-only)
#   experiments/out/tables/cells.csv                          comparators + MW adaptation records
#   experiments/out/revision_2026-09-25/cells_fresh.csv       795 MW fresh-draw records
#   experiments/out/revision_2026-09-25/partition_mass.csv    native partition-mass errors
#   experiments/out/revision_2026-09-25/fresh_runs/<cell>.jld2 (existence check only)
#
# Outputs (experiments/out/primary_fresh_2026-09-27/)
#   primary_cells.csv         cells.csv schema (same columns, same order) + provenance columns
#   primary_manifest.csv      one row per MW run: eligibility, provenance, native values
#   primary_cell_medians.csv  per (problem, d, B, algorithm) medians and MW win flags
#   primary_checks.json       assertions (the script errors out if any is violated)
#
# Conventions
#   * cells.csv rows pass through ExperimentsBase.drop_v5_excluded_rows!
#     first (the stray eggbox/d=5/seed=11 rows, if present); the count is
#     recorded in primary_checks.json.
#   * Comparator rows (is, mh, nuts, ns) are copied verbatim, estimator = "output".
#   * MW rows: one per cells_fresh.csv row, keyed (problem, d, B, seed).
#     Eligible iff status == full_remaining_budget && weight_status == ok &&
#     N_fresh > 0 && all fresh metrics finite. Eligible rows carry the fresh
#     values (Nlike_used = C_total, Neff = Neff_fresh, eta = eta_cost_fresh,
#     W1/SWD/dlogZ/QE = fresh; KL_cube, mode_recovery, mmd_rbf, Rhat_max = NaN
#     because they are not fresh metrics; terminated_by = "final_sample").
#     Unavailable rows carry NaN in every metric column, Nlike_used = C_old,
#     notes = "NO-FINAL-SAMPLE: <status>". Population values are never used.
#   * wall_time_s of an MW row is the measured ADAPTATION wall time of the
#     archived run (identical to wall_adapt_s); the fresh phases are in
#     wall_fresh_draw_eval_s and wall_metrics_s and were measured in a
#     different execution context. No end-to-end sum is formed here.
#
# Usage:  julia --project=. -t 8 experiments/tools/primary_fresh_view.jl
# =============================================================================

using Pkg
const _ROOT = abspath(joinpath(@__DIR__, "..", ".."))
if Base.active_project() != joinpath(_ROOT, "Project.toml")
    Pkg.activate(_ROOT)
end
include(joinpath(_ROOT, "experiments", "src", "ExperimentsBase.jl"))
using .ExperimentsBase
using DataFrames, CSV, Statistics, Printf, Dates

const OUT      = joinpath(_ROOT, "experiments", "out")
const REV      = joinpath(OUT, "revision_2026-09-25")
const PRIMARY  = joinpath(OUT, "primary_fresh_2026-09-27")
const ESTIMATOR = "fresh_v1_2026-09-25"
const ALGS     = ["is", "mh", "nuts", "ns", "mw"]
const PROBS    = ["mvn", "banana", "funnel", "mridges", "shell", "mridges_spiky", "eggbox"]
const CELLS_COLS = ["problem", "algorithm", "d", "B", "seed", "Nlike_used", "wall_time_s",
                    "n_primal", "n_grad_partials", "n_dual_calls", "Neff", "eta_Nlike",
                    "W1_marginal_avg", "SWD", "dlogZ", "QE_p025", "QE_p160", "QE_p500",
                    "QE_p840", "QE_p975", "KL_cube", "mode_recovery", "mmd_rbf", "Rhat_max",
                    "terminated_by", "notes"]
const METRIC_COLS = ["Neff", "eta_Nlike", "W1_marginal_avg", "SWD", "dlogZ", "QE_p025",
                     "QE_p160", "QE_p500", "QE_p840", "QE_p975", "KL_cube", "mode_recovery",
                     "mmd_rbf", "Rhat_max"]
const EXTRA_COLS = ["estimator", "eligibility_status", "eligibility_reason", "C_adaptation",
                    "N_fresh", "C_total", "eta_proposal", "pareto_k", "max_w", "wall_adapt_s",
                    "wall_fresh_draw_eval_s", "wall_metrics_s", "source_sha256"]
const FRESH_METRICS = ["Neff_fresh", "eta_cost_fresh", "eta_proposal_fresh", "W1_marginal_avg_fresh",
                       "SWD_fresh", "dlogZ_fresh", "QE_p025_fresh", "QE_p160_fresh", "QE_p500_fresh",
                       "QE_p840_fresh", "QE_p975_fresh", "logZ_fresh", "logZ_fresh_se"]

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
_f(x) = x === missing ? NaN : Float64(x)
_s(x) = x === missing ? "" : String(string(x))
_i(x) = x === missing ? 0 : Int(round(Float64(x)))
_key(problem, d, B, seed) = (String(problem), Int(d), Float64(B), Int(seed))
_med(v::AbstractVector{<:Real}) = (w = filter(isfinite, Vector{Float64}(v)); isempty(w) ? NaN : median(w))
_admissible(note::AbstractString) = !(occursin("RHAT-FAIL", note) || occursin("BUDGET-VIOLATION", note) ||
                                      occursin("budget-infeasible", note))
_sig5(x::Real) = isfinite(x) && x != 0 ? round(x; sigdigits = 5) : x

# Minimal JSON writer (no JSON package in Project.toml; NaN/Inf -> null).
function _json(io::IO, x; indent::Int = 0)
    pad = repeat("  ", indent)
    if x === nothing || x === missing
        print(io, "null")
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Integer
        print(io, x)
    elseif x isa Real
        isfinite(x) ? print(io, repr(Float64(x))) : print(io, "null")
    elseif x isa AbstractString || x isa Symbol
        s = String(string(x))
        s = replace(s, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n", "\r" => "\\r", "\t" => "\\t")
        print(io, "\"", s, "\"")
    elseif x isa AbstractDict
        ks = sort(collect(keys(x)); by = string)   # deterministic key order
        print(io, "{\n")
        for (n, k) in enumerate(ks)
            print(io, pad, "  \"", String(string(k)), "\": ")
            _json(io, x[k]; indent = indent + 1)
            print(io, n < length(ks) ? ",\n" : "\n")
        end
        print(io, pad, "}")
    elseif x isa AbstractVector || x isa Tuple
        if isempty(x)
            print(io, "[]")
        else
            print(io, "[\n")
            for (n, v) in enumerate(x)
                print(io, pad, "  ")
                _json(io, v; indent = indent + 1)
                print(io, n < length(x) ? ",\n" : "\n")
            end
            print(io, pad, "]")
        end
    else
        _json(io, string(x); indent = indent)
    end
end

# Ordered check registry: every assertion is recorded, the script fails at
# the end if any check is violated (after writing the JSON).
const CHECKS = Vector{Dict{String,Any}}()
function check!(name::AbstractString, passed::Bool; expected = nothing, actual = nothing, note = "")
    push!(CHECKS, Dict{String,Any}("check" => String(name), "pass" => passed,
        "expected" => expected, "actual" => actual, "note" => note))
    passed || @error "PRIMARY CHECK FAILED" name expected actual
    return passed
end

# -----------------------------------------------------------------------------
function main()
    isdir(PRIMARY) || mkpath(PRIMARY)
    t0 = time()

    # ---- inputs -----------------------------------------------------------
    cells = CSV.read(joinpath(OUT, "tables", "cells.csv"), DataFrame)
    n_cells_raw = nrow(cells)
    n_dropped = drop_v5_excluded_rows!(cells)
    @info "cells.csv loaded" rows = n_cells_raw dropped_by_drop_v5_excluded_rows = n_dropped
    cells.notes = [_s(x) for x in cells.notes]
    cells.terminated_by = [_s(x) for x in cells.terminated_by]
    @assert names(cells) == CELLS_COLS "cells.csv schema differs from the expected column list"

    fresh = CSV.read(joinpath(REV, "cells_fresh.csv"), DataFrame)
    part  = CSV.read(joinpath(REV, "partition_mass.csv"), DataFrame)
    part.admissible = [x === missing ? false : (x isa Bool ? x : lowercase(String(x)) == "true") for x in part.admissible]

    # ---- MW adaptation records from cells.csv (keyed) ------------------------
    mw_old = cells[cells.algorithm .== "mw", :]
    old_by_key = Dict{Tuple{String,Int,Float64,Int},DataFrameRow}()
    for r in eachrow(mw_old)
        old_by_key[_key(r.problem, r.d, r.B, r.seed)] = r
    end
    fresh_keys = Set(_key(r.problem, r.d, r.B, r.seed) for r in eachrow(fresh))
    check!("mw_keys_cells_equal_cells_fresh", Set(keys(old_by_key)) == fresh_keys;
        expected = "identical (problem,d,B,seed) key sets", actual = "$(length(old_by_key)) vs $(length(fresh_keys))")
    check!("n_mw_runs", nrow(fresh) == 795; expected = 795, actual = nrow(fresh))

    # Native fresh partition TV per run (estimator == fresh, admissible).
    tv_fresh = Dict{Tuple{String,Int,Float64,Int},Float64}()
    for r in eachrow(part[(part.algorithm .== "mw") .& (part.estimator .== "fresh") .& part.admissible, :])
        tv_fresh[_key(r.problem, r.d, r.B, r.seed)] = _f(r.TV_partition)
    end

    # ---- build MW primary rows + manifest -------------------------------------
    rows_mw  = Vector{Dict{String,Any}}()
    manifest = Vector{Dict{String,Any}}()
    n_eligible = 0
    n_unavail  = 0
    n_ctotal_eq_B = 0
    reason_counts = Dict{String,Int}()
    for r in eachrow(fresh)
        k = _key(r.problem, r.d, r.B, r.seed)
        old = old_by_key[k]
        status = _s(r.status); wstatus = _s(r.weight_status)
        C_old = _f(r.C_old); N_fresh = _i(r.N_fresh); C_total = _f(r.C_total)
        B = Float64(r.B)
        @assert C_old == Float64(old.Nlike_used) "C_old differs from archived Nlike_used for $(k)"
        metrics_finite = all(isfinite(_f(r[Symbol(c)])) for c in FRESH_METRICS)
        eligible = status == "full_remaining_budget" && wstatus == "ok" && N_fresh > 0 && metrics_finite
        reason = if eligible
            "full_remaining_budget; weight_status=ok"
        elseif status != "full_remaining_budget"
            status
        elseif wstatus != "ok"
            "weight_status=" * (isempty(wstatus) ? "missing" : wstatus)
        elseif N_fresh <= 0
            "N_fresh=0"
        else
            "nonfinite_fresh_metric"
        end
        reason_counts[reason] = get(reason_counts, reason, 0) + 1
        if eligible
            n_eligible += 1
            @assert C_old + N_fresh == C_total "C_old + N_fresh != C_total for $(k)"
            @assert C_total <= B "C_total > B for $(k)"
            C_total == B && (n_ctotal_eq_B += 1)
        else
            n_unavail += 1
        end
        payload = joinpath("experiments", "out", "revision_2026-09-25", "fresh_runs",
            @sprintf("%s_mw_d%d_B%s_seed%d.jld2", k[1], k[2], ExperimentsBase._budget_token(k[3]), k[4]))
        isfile(joinpath(_ROOT, payload)) || error("payload missing: $payload")

        row = Dict{String,Any}()
        row["problem"] = k[1]; row["algorithm"] = "mw"; row["d"] = k[2]; row["B"] = B; row["seed"] = k[4]
        row["wall_time_s"] = Float64(old.wall_time_s)          # adaptation phase (harness)
        row["n_grad_partials"] = Int(old.n_grad_partials)
        row["n_dual_calls"] = Int(old.n_dual_calls)
        row["KL_cube"] = NaN; row["mode_recovery"] = NaN; row["mmd_rbf"] = NaN; row["Rhat_max"] = NaN
        row["estimator"] = ESTIMATOR
        row["C_adaptation"] = C_old; row["N_fresh"] = N_fresh; row["C_total"] = C_total
        row["wall_adapt_s"] = Float64(old.wall_time_s)
        row["source_sha256"] = _s(r.source_sha256)
        if eligible
            row["Nlike_used"] = C_total
            row["n_primal"] = Int(old.n_primal) + N_fresh
            row["Neff"] = _f(r.Neff_fresh); row["eta_Nlike"] = _f(r.eta_cost_fresh)
            row["W1_marginal_avg"] = _f(r.W1_marginal_avg_fresh); row["SWD"] = _f(r.SWD_fresh)
            row["dlogZ"] = _f(r.dlogZ_fresh)
            for q in ("p025", "p160", "p500", "p840", "p975")
                row["QE_" * q] = _f(r[Symbol("QE_" * q * "_fresh")])
            end
            row["terminated_by"] = "final_sample"; row["notes"] = ""
            row["eligibility_status"] = "eligible"; row["eligibility_reason"] = reason
            row["eta_proposal"] = _f(r.eta_proposal_fresh); row["pareto_k"] = _f(r.pareto_k_fresh)
            row["max_w"] = _f(r.max_w_fresh)
            row["wall_fresh_draw_eval_s"] = _f(r.wall_fresh_draw_eval_s)
            row["wall_metrics_s"] = _f(r.wall_metrics_s)
        else
            row["Nlike_used"] = C_old
            row["n_primal"] = Int(old.n_primal)
            for c in ("Neff", "eta_Nlike", "W1_marginal_avg", "SWD", "dlogZ", "QE_p025", "QE_p160",
                      "QE_p500", "QE_p840", "QE_p975")
                row[c] = NaN
            end
            row["terminated_by"] = ""; row["notes"] = "NO-FINAL-SAMPLE: " * status
            row["eligibility_status"] = "unavailable"; row["eligibility_reason"] = reason
            row["eta_proposal"] = NaN; row["pareto_k"] = NaN; row["max_w"] = NaN
            row["wall_fresh_draw_eval_s"] = missing; row["wall_metrics_s"] = missing
        end
        push!(rows_mw, row)

        m = Dict{String,Any}()
        m["problem"] = k[1]; m["d"] = k[2]; m["B"] = B; m["seed"] = k[4]
        m["estimator"] = ESTIMATOR
        m["eligibility_status"] = row["eligibility_status"]; m["eligibility_reason"] = reason
        m["source_file"] = _s(r.source_file); m["source_sha256"] = _s(r.source_sha256)
        m["proposal_sha256"] = _s(r.proposal_sha256); m["rng_seed"] = _s(r.rng_seed)
        m["C_adaptation"] = C_old; m["N_fresh"] = N_fresh; m["C_total"] = C_total
        m["Neff_native"] = eligible ? _f(r.Neff_fresh) : NaN
        m["eta_total"] = eligible ? _f(r.eta_cost_fresh) : NaN
        m["eta_proposal"] = eligible ? _f(r.eta_proposal_fresh) : NaN
        m["logZ"] = eligible ? _f(r.logZ_fresh) : NaN
        m["logZ_se"] = eligible ? _f(r.logZ_fresh_se) : NaN
        m["abs_dlogZ"] = eligible ? abs(_f(r.dlogZ_fresh)) : NaN
        m["W1"] = eligible ? _f(r.W1_marginal_avg_fresh) : NaN
        m["SWD"] = eligible ? _f(r.SWD_fresh) : NaN
        for (q, qq) in zip(("p025", "p160", "p500", "p840", "p975"), ("QE025", "QE160", "QE500", "QE840", "QE975"))
            m[qq] = eligible ? _f(r[Symbol("QE_" * q * "_fresh")]) : NaN
        end
        m["TV_partition_fresh"] = eligible ? get(tv_fresh, k, NaN) : NaN
        m["pareto_k"] = eligible ? _f(r.pareto_k_fresh) : NaN
        m["max_w"] = eligible ? _f(r.max_w_fresh) : NaN
        m["wall_adapt_s"] = Float64(old.wall_time_s)
        m["wall_fresh_draw_eval_s"] = eligible ? _f(r.wall_fresh_draw_eval_s) : NaN
        m["wall_metrics_s"] = eligible ? _f(r.wall_metrics_s) : NaN
        m["payload_path"] = payload
        push!(manifest, m)
    end

    # ---- assemble primary_cells.csv ----------------------------------------
    comp = cells[cells.algorithm .!= "mw", :]
    prim = DataFrame()
    for c in CELLS_COLS
        cs = Symbol(c)
        comp_col = comp[!, cs]
        mw_col = [row[c] for row in rows_mw]
        if c in ("problem", "algorithm", "terminated_by", "notes")
            prim[!, cs] = vcat(String.(string.(comp_col)), String.(mw_col))
        elseif c in ("d", "seed", "n_primal", "n_grad_partials", "n_dual_calls")
            prim[!, cs] = vcat(Int.(comp_col), Int.(mw_col))
        else
            prim[!, cs] = vcat(Float64.(_f.(comp_col)), Float64.(mw_col))
        end
    end
    n_comp = nrow(comp)
    comp_reason(note) = occursin("RHAT-FAIL", note) ? "RHAT-FAIL" :
                        occursin("BUDGET-VIOLATION", note) ? "BUDGET-VIOLATION" :
                        occursin("budget-infeasible", note) ? "budget-infeasible" : "admissible"
    prim[!, :estimator] = vcat(fill("output", n_comp), [row["estimator"] for row in rows_mw])
    prim[!, :eligibility_status] = vcat([_admissible(n) ? "admissible" : "excluded" for n in comp.notes],
                                        [row["eligibility_status"] for row in rows_mw])
    prim[!, :eligibility_reason] = vcat([comp_reason(n) for n in comp.notes],
                                        [row["eligibility_reason"] for row in rows_mw])
    for c in ("C_adaptation", "C_total", "eta_proposal", "pareto_k", "max_w", "wall_adapt_s",
              "wall_fresh_draw_eval_s", "wall_metrics_s")
        prim[!, Symbol(c)] = Vector{Union{Missing,Float64}}(vcat(fill(missing, n_comp), [row[c] for row in rows_mw]))
    end
    prim[!, :N_fresh] = Vector{Union{Missing,Int}}(vcat(fill(missing, n_comp), [row["N_fresh"] for row in rows_mw]))
    prim[!, :source_sha256] = Vector{Union{Missing,String}}(vcat(fill(missing, n_comp), [row["source_sha256"] for row in rows_mw]))
    select!(prim, vcat(Symbol.(CELLS_COLS), Symbol.(EXTRA_COLS)))
    sort!(prim, [:problem, :d, :B, :algorithm, :seed])
    @assert names(prim) == vcat(CELLS_COLS, EXTRA_COLS)
    CSV.write(joinpath(PRIMARY, "primary_cells.csv"), prim)

    man = DataFrame([c => [m[c] for m in manifest] for c in
        ["problem", "d", "B", "seed", "estimator", "eligibility_status", "eligibility_reason", "source_file",
         "source_sha256", "proposal_sha256", "rng_seed", "C_adaptation", "N_fresh", "C_total", "Neff_native",
         "eta_total", "eta_proposal", "logZ", "logZ_se", "abs_dlogZ", "W1", "SWD", "QE025", "QE160", "QE500",
         "QE840", "QE975", "TV_partition_fresh", "pareto_k", "max_w", "wall_adapt_s", "wall_fresh_draw_eval_s",
         "wall_metrics_s", "payload_path"]]...)
    sort!(man, [:problem, :d, :B, :seed])
    CSV.write(joinpath(PRIMARY, "primary_manifest.csv"), man)

    # ---- counts ---------------------------------------------------------------
    mwp = prim[prim.algorithm .== "mw", :]
    elig = mwp[mwp.eligibility_status .== "eligible", :]
    check!("n_mw_rows_in_primary_cells", nrow(mwp) == 795; expected = 795, actual = nrow(mwp))
    check!("n_eligible", n_eligible == 489; expected = 489, actual = n_eligible)
    check!("n_unavailable", n_unavail == 306; expected = 306, actual = n_unavail)
    by_B = Dict{String,Any}()
    for (B, exp_e) in ((5e3, 0), (5e4, 224), (5e5, 265))
        ne = count((mwp.B .== B) .& (mwp.eligibility_status .== "eligible"))
        na = count(mwp.B .== B)
        by_B[@sprintf("B=%g", B)] = Dict("eligible" => ne, "attempted" => na)
        check!(@sprintf("eligible_at_B=%g", B), ne == exp_e && na == 265; expected = "$exp_e/265", actual = "$ne/$na")
    end
    check!("C_total_equals_B_for_all_eligible", n_ctotal_eq_B == n_eligible;
        expected = n_eligible, actual = n_ctotal_eq_B,
        note = "charged cost C_total = C_adapt + N_fresh equals the nominal budget B for every available run")
    n_failed_sanity = count(occursin.("FAILED-SANITY", cells.notes))

    # ---- per-cell medians and MW win flags --------------------------------------
    cell_rows = Vector{Dict{String,Any}}()
    wins = Dict("eta" => 0, "W1" => 0, "absdlogZ" => 0)
    n_elig_cells = 0; n_elig_cells_B = Dict(5e3 => 0, 5e4 => 0, 5e5 => 0)
    headline = Dict{String,Any}()
    headline_wins = Dict("eta" => 0, "W1" => 0, "absdlogZ" => 0)
    headline_W1_mw_wins = String[]; headline_W1_comp_best = Dict{String,String}()
    grid = unique(prim[:, [:problem, :d, :B]])
    sort!(grid, [:problem, :d, :B])
    for g in eachrow(grid)
        cell = prim[(prim.problem .== g.problem) .& (prim.d .== g.d) .& (prim.B .== g.B), :]
        med_by_alg = Dict{String,Dict{String,Any}}()
        for alg in ALGS
            s = cell[cell.algorithm .== alg, :]
            isempty(s) && continue
            keep = alg == "mw" ? (s.eligibility_status .== "eligible") : [_admissible(n) for n in s.notes]
            sk = s[keep, :]
            est = alg == "mw" ? "fresh" : "output"
            pp = part[(part.problem .== g.problem) .& (part.algorithm .== alg) .& (part.d .== g.d) .&
                      (part.B .== g.B) .& part.admissible .& (part.estimator .== est), :]
            m = Dict{String,Any}(
                "problem" => g.problem, "d" => g.d, "B" => g.B, "algorithm" => alg,
                "estimator" => alg == "mw" ? ESTIMATOR : "output",
                "n_attempted" => nrow(s), "n_eligible" => nrow(sk),
                "eta_Nlike" => _med(sk.eta_Nlike), "W1_marginal_avg" => _med(sk.W1_marginal_avg),
                "SWD" => _med(sk.SWD),
                "abs_dlogZ" => alg in ("mh", "nuts") ? NaN : _med(abs.(sk.dlogZ)),
                "QE_p025" => _med(sk.QE_p025), "QE_p160" => _med(sk.QE_p160), "QE_p500" => _med(sk.QE_p500),
                "QE_p840" => _med(sk.QE_p840), "QE_p975" => _med(sk.QE_p975),
                "TV_partition" => _med(pp.TV_partition), "n_TV_partition" => nrow(pp),
                "TV_partition_estimator" => nrow(pp) > 0 ? est : "",
                "n_reason_RHAT_FAIL" => count(occursin.("RHAT-FAIL", s.notes)),
                "n_reason_BUDGET_VIOLATION" => count(occursin.("BUDGET-VIOLATION", s.notes)),
                "n_reason_budget_infeasible" => count(occursin.("budget-infeasible", s.notes)),
                "n_reason_no_final_sample" => alg == "mw" ? count(s.eligibility_status .== "unavailable") : 0,
                "best_eta_comparator" => "", "best_eta_median" => NaN, "win_eta" => missing,
                "best_W1_comparator" => "", "best_W1_median" => NaN, "win_W1" => missing,
                "best_absdlogZ_comparator" => "", "best_absdlogZ_median" => NaN, "win_absdlogZ" => missing)
            med_by_alg[alg] = m
        end
        if haskey(med_by_alg, "mw")
            m = med_by_alg["mw"]
            if m["n_eligible"] > 0
                n_elig_cells += 1
                n_elig_cells_B[g.B] += 1
            end
            for (name, col, maximize, comps) in (("eta", "eta_Nlike", true, ["is", "mh", "nuts", "ns"]),
                                                  ("W1", "W1_marginal_avg", false, ["is", "mh", "nuts", "ns"]),
                                                  ("absdlogZ", "abs_dlogZ", false, ["is", "ns"]))
                cands = [(a, med_by_alg[a][col]) for a in comps if haskey(med_by_alg, a) && isfinite(med_by_alg[a][col])]
                if !isempty(cands)
                    best = maximize ? argmax(x -> x[2], cands) : argmin(x -> x[2], cands)
                    m["best_" * name * "_comparator"] = best[1]
                    m["best_" * name * "_median"] = best[2]
                    mm = m[col]
                    if isfinite(mm)
                        win = maximize ? (mm > best[2]) : (mm < best[2])
                        m["win_" * name] = win
                        win && (wins[name] += 1)
                    end
                end
            end
            # headline setting: B = 5e5, d = 5 (eggbox d = 2)
            if g.B == 5e5 && g.d == (g.problem == "eggbox" ? 2 : 5) && m["n_eligible"] > 0
                headline[g.problem] = Dict(
                    "n_eligible" => m["n_eligible"], "n_attempted" => m["n_attempted"],
                    "eta" => m["eta_Nlike"], "W1" => m["W1_marginal_avg"], "abs_dlogZ" => m["abs_dlogZ"],
                    "SWD" => m["SWD"], "QE500" => m["QE_p500"], "QE975" => m["QE_p975"],
                    "TV_partition_fresh" => m["TV_partition"],
                    "best_eta" => (m["best_eta_comparator"], m["best_eta_median"]),
                    "best_W1" => (m["best_W1_comparator"], m["best_W1_median"]),
                    "best_absdlogZ" => (m["best_absdlogZ_comparator"], m["best_absdlogZ_median"]),
                    "win_eta" => m["win_eta"], "win_W1" => m["win_W1"], "win_absdlogZ" => m["win_absdlogZ"])
                for name in ("eta", "W1", "absdlogZ")
                    m["win_" * name] === true && (headline_wins[name] += 1)
                end
                m["win_W1"] === true && push!(headline_W1_mw_wins, g.problem)
                headline_W1_comp_best[g.problem] = m["best_W1_comparator"]
            end
        end
        for alg in ALGS
            haskey(med_by_alg, alg) && push!(cell_rows, med_by_alg[alg])
        end
    end
    med_cols = ["problem", "d", "B", "algorithm", "estimator", "n_attempted", "n_eligible", "eta_Nlike",
                "W1_marginal_avg", "SWD", "abs_dlogZ", "QE_p025", "QE_p160", "QE_p500", "QE_p840", "QE_p975",
                "TV_partition", "n_TV_partition", "TV_partition_estimator", "n_reason_RHAT_FAIL",
                "n_reason_BUDGET_VIOLATION", "n_reason_budget_infeasible", "n_reason_no_final_sample",
                "best_eta_comparator", "best_eta_median", "win_eta", "best_W1_comparator", "best_W1_median",
                "win_W1", "best_absdlogZ_comparator", "best_absdlogZ_median", "win_absdlogZ"]
    meds = DataFrame([c => [r[c] for r in cell_rows] for c in med_cols]...)
    CSV.write(joinpath(PRIMARY, "primary_cell_medians.csv"), meds)

    check!("n_eligible_cells", n_elig_cells == 28; expected = 28, actual = n_elig_cells)
    check!("n_eligible_cells_B=5e4", n_elig_cells_B[5e4] == 13; expected = 13, actual = n_elig_cells_B[5e4])
    check!("n_eligible_cells_B=5e5", n_elig_cells_B[5e5] == 15; expected = 15, actual = n_elig_cells_B[5e5])
    check!("n_eligible_cells_B=5e3", n_elig_cells_B[5e3] == 0; expected = 0, actual = n_elig_cells_B[5e3])
    fun2 = meds[(meds.problem .== "funnel") .& (meds.d .== 2) .& (meds.B .== 5e4) .& (meds.algorithm .== "mw"), :]
    check!("funnel_d2_B5e4_eligible_19_of_20", nrow(fun2) == 1 && fun2.n_eligible[1] == 19 && fun2.n_attempted[1] == 20;
        expected = "19/20", actual = nrow(fun2) == 1 ? "$(fun2.n_eligible[1])/$(fun2.n_attempted[1])" : "row missing")
    for dd in (5, 10)
        fu = meds[(meds.problem .== "funnel") .& (meds.d .== dd) .& (meds.B .== 5e4) .& (meds.algorithm .== "mw"), :]
        check!("funnel_d$(dd)_B5e4_eligible_0", nrow(fu) == 1 && fu.n_eligible[1] == 0;
            expected = 0, actual = nrow(fu) == 1 ? fu.n_eligible[1] : "row missing")
    end
    check!("wins_eta_over_28_cells", wins["eta"] == 25; expected = 25, actual = wins["eta"])
    check!("wins_W1_over_28_cells", wins["W1"] == 22; expected = 22, actual = wins["W1"])
    check!("wins_absdlogZ_over_28_cells", wins["absdlogZ"] == 23; expected = 23, actual = wins["absdlogZ"])
    check!("headline_wins_eta", headline_wins["eta"] == 6; expected = 6, actual = headline_wins["eta"])
    check!("headline_wins_W1", headline_wins["W1"] == 4; expected = 4, actual = headline_wins["W1"])
    check!("headline_wins_absdlogZ", headline_wins["absdlogZ"] == 5; expected = 5, actual = headline_wins["absdlogZ"])
    check!("headline_W1_mw_wins_set", Set(headline_W1_mw_wins) == Set(["mvn", "banana", "funnel", "mridges_spiky"]);
        expected = ["mvn", "banana", "funnel", "mridges_spiky"], actual = sort(headline_W1_mw_wins))
    check!("headline_W1_mh_wins_set",
        all(get(headline_W1_comp_best, p, "") == "mh" && haskey(headline, p) && headline[p]["win_W1"] === false
            for p in ("mridges", "shell", "eggbox"));
        expected = "mh is the best comparator and beats MW on mridges, shell, eggbox",
        actual = Dict(p => (get(headline_W1_comp_best, p, ""), haskey(headline, p) ? headline[p]["win_W1"] : nothing)
                      for p in ("mridges", "shell", "eggbox")))

    expected_headline = Dict(
        "mvn"           => (0.834939,   0.0160621, 0.000137981),
        "banana"        => (0.515983,   0.0210984, 0.000802904),
        "funnel"        => (0.443535,   0.0371858, 0.000674538),
        "mridges"       => (0.243189,   0.153439,  0.128768),
        "shell"         => (7.76273e-6, 1.01386,   0.467018),
        "mridges_spiky" => (0.750680,   0.0131615, 0.000640840),
        "eggbox"        => (0.715551,   0.0853019, 0.000633061))
    for (p, (e_eta, e_w1, e_dz)) in expected_headline
        h = get(headline, p, nothing)
        ok = h !== nothing && isapprox(h["eta"], e_eta; rtol = 1e-5) && isapprox(h["W1"], e_w1; rtol = 1e-5) &&
             isapprox(h["abs_dlogZ"], e_dz; rtol = 1e-5)
        check!("headline_medians_5sig_" * p, ok; expected = (e_eta, e_w1, e_dz),
            actual = h === nothing ? nothing : (_sig5(h["eta"]), _sig5(h["W1"]), _sig5(h["abs_dlogZ"])))
    end
    tv_mr = get(get(headline, "mridges", Dict()), "TV_partition_fresh", NaN)
    tv_sp = get(get(headline, "mridges_spiky", Dict()), "TV_partition_fresh", NaN)
    check!("headline_TV_partition_fresh_mridges_approx_0.122", isfinite(tv_mr) && abs(tv_mr - 0.122) < 0.001;
        expected = "~0.122", actual = tv_mr)
    check!("headline_TV_partition_fresh_spiky_approx_0.001", isfinite(tv_sp) && abs(tv_sp - 0.001) < 0.001;
        expected = "~0.001", actual = tv_sp)

    # ---- equality of the 489 eligible rows with cells_fresh.csv (after CSV round trip) ----
    prim_rt = CSV.read(joinpath(PRIMARY, "primary_cells.csv"), DataFrame)
    prim_rt = prim_rt[(prim_rt.algorithm .== "mw") .& (coalesce.(prim_rt.eligibility_status, "") .== "eligible"), :]
    fresh_by_key = Dict(_key(r.problem, r.d, r.B, r.seed) => r for r in eachrow(fresh))
    pairs = (("Neff", "Neff_fresh"), ("eta_Nlike", "eta_cost_fresh"), ("W1_marginal_avg", "W1_marginal_avg_fresh"),
             ("SWD", "SWD_fresh"), ("dlogZ", "dlogZ_fresh"), ("QE_p025", "QE_p025_fresh"), ("QE_p160", "QE_p160_fresh"),
             ("QE_p500", "QE_p500_fresh"), ("QE_p840", "QE_p840_fresh"), ("QE_p975", "QE_p975_fresh"),
             ("Nlike_used", "C_total"), ("eta_proposal", "eta_proposal_fresh"), ("pareto_k", "pareto_k_fresh"),
             ("max_w", "max_w_fresh"))
    maxdev = 0.0; n_cmp = 0
    for r in eachrow(prim_rt)
        fr = fresh_by_key[_key(r.problem, r.d, r.B, r.seed)]
        for (a, b) in pairs
            va = _f(r[Symbol(a)]); vb = _f(fr[Symbol(b)])
            maxdev = max(maxdev, abs(va - vb)); n_cmp += 1
        end
    end
    check!("eligible_rows_equal_cells_fresh_values", nrow(prim_rt) == 489 && maxdev == 0.0;
        expected = "489 rows, max |primary - cells_fresh| = 0", actual = "$(nrow(prim_rt)) rows, max dev = $maxdev over $n_cmp values")
    n_pop_leak = count(r -> r.eligibility_status == "unavailable" &&
                           any(isfinite(_f(r[Symbol(c)])) for c in ("Neff", "eta_Nlike", "W1_marginal_avg", "SWD", "dlogZ",
                                                                    "QE_p025", "QE_p160", "QE_p500", "QE_p840", "QE_p975")),
                       eachrow(mwp))
    check!("no_metric_value_on_unavailable_mw_rows", n_pop_leak == 0; expected = 0, actual = n_pop_leak)

    # ---- JSON ----------------------------------------------------------------
    headline_out = Dict{String,Any}()
    for p in PROBS
        haskey(headline, p) || continue
        h = headline[p]
        headline_out[p] = Dict(
            "d" => p == "eggbox" ? 2 : 5, "B" => 5e5, "n_eligible" => h["n_eligible"], "n_attempted" => h["n_attempted"],
            "eta_fresh_median" => h["eta"], "W1_fresh_median" => h["W1"], "abs_dlogZ_fresh_median" => h["abs_dlogZ"],
            "SWD_fresh_median" => h["SWD"], "QE500_fresh_median" => h["QE500"], "QE975_fresh_median" => h["QE975"],
            "TV_partition_fresh_median" => h["TV_partition_fresh"],
            "best_comparator_eta" => Dict("algorithm" => h["best_eta"][1], "median" => h["best_eta"][2]),
            "best_comparator_W1" => Dict("algorithm" => h["best_W1"][1], "median" => h["best_W1"][2]),
            "best_comparator_abs_dlogZ" => Dict("algorithm" => h["best_absdlogZ"][1], "median" => h["best_absdlogZ"][2]),
            "mw_wins" => Dict("eta" => h["win_eta"], "W1" => h["win_W1"], "abs_dlogZ" => h["win_absdlogZ"]))
    end
    elig_cells = [Dict("problem" => r.problem, "d" => r.d, "B" => r.B, "n_eligible" => r.n_eligible,
                       "n_attempted" => r.n_attempted)
                  for r in eachrow(meds) if r.algorithm == "mw" && r.n_eligible > 0]
    n_fail = count(c -> !c["pass"], CHECKS)
    out = Dict{String,Any}(
        "generated_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "tool" => "experiments/tools/primary_fresh_view.jl",
        "estimator" => ESTIMATOR,
        "inputs" => ["experiments/out/tables/cells.csv", "experiments/out/revision_2026-09-25/cells_fresh.csv",
                     "experiments/out/revision_2026-09-25/partition_mass.csv",
                     "experiments/out/revision_2026-09-25/fresh_runs/<cell>.jld2 (existence only)"],
        "outputs" => ["primary_cells.csv", "primary_manifest.csv", "primary_cell_medians.csv", "primary_checks.json"],
        "all_checks_passed" => n_fail == 0,
        "n_checks" => length(CHECKS), "n_checks_failed" => n_fail,
        "cells_csv" => Dict("rows_raw" => n_cells_raw, "rows_after_drop_v5_excluded_rows" => nrow(cells),
            "dropped_by_drop_v5_excluded_rows" => n_dropped,
            "drop_v5_excluded_rows_rule" => "eggbox / d = 5 / seed = 11 stray rows (none present in the current file if 0)",
            "n_notes_FAILED_SANITY" => n_failed_sanity,
            "comparator_admissibility_rule" => "notes contain none of RHAT-FAIL, BUDGET-VIOLATION, budget-infeasible"),
        "mw_runs" => Dict("n_total" => nrow(fresh), "n_eligible" => n_eligible, "n_unavailable" => n_unavail,
            "eligibility_rule" => "status == full_remaining_budget && weight_status == ok && N_fresh > 0 && finite fresh metrics",
            "eligibility_reason_counts" => reason_counts, "by_budget" => by_B,
            "n_eligible_with_C_total_equal_B" => n_ctotal_eq_B,
            "cost_identity" => "C_adaptation + N_fresh == C_total <= B asserted for every eligible row"),
        "eligible_cells" => Dict("n" => n_elig_cells, "by_budget" => Dict("B=5000" => n_elig_cells_B[5e3],
            "B=50000" => n_elig_cells_B[5e4], "B=500000" => n_elig_cells_B[5e5]), "cells" => elig_cells),
        "mw_wins_over_eligible_cells" => Dict("eta" => wins["eta"], "W1" => wins["W1"], "abs_dlogZ" => wins["absdlogZ"],
            "rule" => "descriptive: MW fresh cell median vs the best admissible comparator cell median (eta larger wins; W1 and |dlogZ| smaller wins; |dlogZ| comparators IS/NS only); no significance test"),
        "headline" => Dict("setting" => "B = 5e5, d = 5 (eggbox d = 2)",
            "wins" => Dict("eta" => headline_wins["eta"], "W1" => headline_wins["W1"], "abs_dlogZ" => headline_wins["absdlogZ"]),
            "W1_mw_wins" => sort(headline_W1_mw_wins), "W1_best_comparator" => headline_W1_comp_best,
            "per_target" => headline_out),
        "primary_cells_conventions" => Dict(
            "mw_wall_time_s" => "adaptation phase only (harness wall_time_s of the archived run; equals wall_adapt_s)",
            "mw_unavailable_rows" => "all metric columns NaN, Nlike_used = C_adaptation, terminated_by empty, notes = NO-FINAL-SAMPLE: <status>",
            "mw_eligible_rows" => "Nlike_used = C_total, Neff = Neff_fresh, eta_Nlike = eta_cost_fresh, W1/SWD/dlogZ/QE fresh, KL_cube/mode_recovery/mmd_rbf/Rhat_max NaN, terminated_by = final_sample, n_primal = adaptation primal count + N_fresh",
            "never_population" => "no population value is used anywhere in the primary view"),
        "checks" => CHECKS,
        "elapsed_s" => round(time() - t0; digits = 1))
    open(joinpath(PRIMARY, "primary_checks.json"), "w") do io
        _json(io, out); println(io)
    end
    @info "primary view written" dir = PRIMARY n_primary_rows = nrow(prim) n_manifest = nrow(man) n_medians = nrow(meds) n_checks = length(CHECKS) n_failed = n_fail
    n_fail == 0 || error("$(n_fail) primary check(s) failed; see primary_checks.json")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end

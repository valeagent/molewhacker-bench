# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# partition_mass.jl - weighted partition-mass error for the multimodal
# targets (revision of 25 September 2026; replaces the retired
# centroid-clustering mode-recovery rate, archived as `mode_recovery` in
# cells.csv).
# =============================================================================
#
# Definition (frozen before any algorithm ranking was inspected):
#
#   p_hat[k]     = sum(w_i : theta_i in region k) / sum(w_i)
#   TV_partition = 0.5 * sum_k |p_hat[k] - p_true[k]|
#
# computed directly from the archived weighted output (original weights;
# repeated MH states retained with their multiplicity; no clustering, no
# resampling). Descriptive threshold coverage: fraction of regions with
# p_hat[k] >= 0.25 p_true[k] (0.25 is a convention).
#
# Partitions and reference probabilities
#   smooth M-ridges : the 2^d coordinate-sign orthants (theta_j >= 0 -> "+").
#       p_true(first sign s) = I_s / ((I_- + I_+) 2^(d-1)),
#       I_s = integral over the half-line of g(t) = p1(t) Z_cond(t)^(d-1),
#       p1 the untruncated theta_1 mixture density and Z_cond(t) the cube
#       probability of one transverse conditional. Transverse sign
#       probabilities are exactly 1/2 by the symmetry of the truncated
#       conditional. These are orthants, not gradient-ascent basins.
#   spiky M-ridges  : seven theta_1 intervals bounded by the six density
#       minima between the seven verified maxima of the 20-component
#       theta_1 mixture, outer bounds -L, L; reference probabilities from
#       Gaussian CDF differences (equal component weights and the common
#       transverse cube factor cancel).
#   eggbox          : omitted (no validated complete mode list on the cube).
#
# Outputs (experiments/out/revision_2026-09-25/):
#   partition_reference.csv    regions and p_true per (problem, d)
#   partition_calibration.csv  truth-draw checks and multinomial floors
#   partition_mass.csv         per (problem, d, B, seed, algorithm, estimator)
#   partition_tests.csv        paired Wilcoxon / Cliff's delta at B = 5e5
#   figs: partition__<problem>__d5__Ball__all__all.pdf
#
# Usage:
#   julia --project=. -t 8 experiments/tools/partition_mass.jl --test
#   julia --project=. -t 8 experiments/tools/partition_mass.jl --compute [--with-fresh]
#   julia --project=. -t 8 experiments/tools/partition_mass.jl --figures --stats
# =============================================================================

include(joinpath(@__DIR__, "fresh_draw_recompute.jl"))   # definitions only (guarded main)
using QuadGK
using CairoMakie
using LaTeXStrings

const PART_PROBLEMS = (:mridges, :mridges_spiky)

# -----------------------------------------------------------------------------
# Reference probabilities
# -----------------------------------------------------------------------------
struct Partition
    problem::Symbol
    d::Int
    n_regions::Int
    p_true::Vector{Float64}
    boundaries::Vector{Float64}      # spiky: six interior boundaries; mridges: empty
    maxima::Vector{Float64}          # spiky: seven theta_1 maxima
    notes::String
end

function mridges_partition(cfg::ConfigMRidges)
    Δ, τ, a, b, c, s0, L, d = cfg.Δ, cfg.τ, cfg.a, cfg.b, cfg.c, cfg.s0, cfg.L, cfg.d
    p1(t) = 0.5 * pdf(Normal(Δ, τ), t) + 0.5 * pdf(Normal(-Δ, τ), t)
    function zcond(t)
        m = a + b * t^2; s = s0 * exp(c * t)
        0.5 * (cdf(Normal(m, s), L) - cdf(Normal(m, s), -L)) +
        0.5 * (cdf(Normal(-m, s), L) - cdf(Normal(-m, s), -L))
    end
    g(t) = p1(t) * zcond(t)^(d - 1)
    Im, em = quadgk(g, -L, 0.0; rtol = 1e-12, atol = 0.0)
    Ip, ep = quadgk(g, 0.0, L; rtol = 1e-12, atol = 0.0)
    pminus = Im / (Im + Ip); pplus = Ip / (Im + Ip)
    K = 2^d
    p = zeros(K)
    for k in 0:(K - 1)
        first_negative = (k & 0x1) == 1
        p[k + 1] = (first_negative ? pminus : pplus) / 2^(d - 1)
    end
    return Partition(:mridges, d, K, p, Float64[], Float64[],
        @sprintf("orthants by coordinate sign (bit j-1 set <=> theta_j < 0); P(theta_1<0)=%.10f, P(theta_1>=0)=%.10f; quadgk abs. errors %.1e, %.1e", pminus, pplus, em, ep))
end

function _spiky_density_funcs(cfg::ConfigMRidgesSpiky)
    mus = mridges_spiky_modes_theta1(cfg); σ = cfg.σ_spike
    f(t) = sum(pdf(Normal(μ, σ), t) for μ in mus)
    f1(t) = sum(-(t - μ) / σ^2 * pdf(Normal(μ, σ), t) for μ in mus)
    f2(t) = sum(((t - μ)^2 / σ^4 - 1 / σ^2) * pdf(Normal(μ, σ), t) for μ in mus)
    return mus, σ, f, f1, f2
end

function _bisect_root(h, lo, hi; iters = 200)
    flo = h(lo); fhi = h(hi)
    flo * fhi < 0 || error("root not bracketed on [$lo, $hi]: h = $flo, $fhi")
    for _ in 1:iters
        mid = 0.5 * (lo + hi); fm = h(mid)
        if flo * fm <= 0
            hi = mid; fhi = fm
        else
            lo = mid; flo = fm
        end
    end
    return 0.5 * (lo + hi)
end

function spiky_partition(cfg::ConfigMRidgesSpiky)
    cfg.kernel === :gaussian || error("spiky partition: Gaussian kernel expected")
    mus, σ, f, f1, f2 = _spiky_density_funcs(cfg)
    L = cfg.L
    # Maxima: grid scan for sign changes of f' from + to -, refined by bisection.
    lo, hi = minimum(mus) - 6σ, maximum(mus) + 6σ
    grid = range(lo, hi; length = 200_001)
    fp = f1.(grid)
    maxima = Float64[]
    for i in 1:(length(grid) - 1)
        if fp[i] > 0 && fp[i + 1] <= 0
            r = _bisect_root(f1, grid[i], grid[i + 1])
            f2(r) < 0 || error("stationary point at $r is not a maximum")
            push!(maxima, r)
        end
    end
    length(maxima) == 7 || error("expected 7 maxima, found $(length(maxima)): $maxima")
    # Minima between consecutive maxima: f' goes - to + exactly once.
    minima = Float64[]
    for i in 1:6
        r = _bisect_root(f1, maxima[i] + 1e-9, maxima[i + 1] - 1e-9)
        f2(r) > 0 || error("stationary point at $r is not a minimum")
        push!(minima, r)
    end
    edges = vcat(-L, minima, L)
    num(a, b) = sum(cdf(Normal(μ, σ), b) - cdf(Normal(μ, σ), a) for μ in mus)
    den = num(-L, L)
    p = [num(edges[i], edges[i + 1]) / den for i in 1:7]
    return Partition(:mridges_spiky, cfg.d, 7, p, minima, maxima,
        @sprintf("seven theta_1 intervals bounded by the six density minima between the seven maxima; 20 components (15 distinct centres) with equal weights; sigma_spike=%.3f", σ))
end

function build_partition(problem::Symbol, d::Int)
    cfg = _make_cfg(problem, d)
    problem === :mridges && return mridges_partition(cfg)
    problem === :mridges_spiky && return spiky_partition(cfg)
    error("no partition for $problem")
end

# -----------------------------------------------------------------------------
# Region assignment and the metric
# -----------------------------------------------------------------------------
@inline function region_index(part::Partition, θ::AbstractVector{<:Real})
    if part.problem === :mridges
        k = 0
        @inbounds for j in 1:part.d
            θ[j] < 0 && (k |= (1 << (j - 1)))
        end
        return k + 1
    else
        t = θ[1]; r = 1
        @inbounds for bnd in part.boundaries
            t >= bnd && (r += 1)
        end
        return r
    end
end

"""
    partition_mass(part, samples, weights) -> (TV, coverage_quarter, p_hat, n_used)

`samples` is d x N, `weights` length N (empty or uniform -> equal weights).
Non-finite weights raise; rows with NaN coordinates raise.
"""
function partition_mass(part::Partition, samples::AbstractMatrix{<:Real}, weights::AbstractVector{<:Real})
    d, N = size(samples)
    d == part.d || error("dimension mismatch: samples d=$d, partition d=$(part.d)")
    w = isempty(weights) ? ones(N) : Float64.(weights)
    all(isfinite, w) || error("non-finite weights")
    all(>=(0), w) || error("negative weights")
    p_hat = zeros(part.n_regions)
    @inbounds for i in 1:N
        θ = view(samples, :, i)
        any(isnan, θ) && error("NaN coordinate at sample $i")
        p_hat[region_index(part, θ)] += w[i]
    end
    S = sum(p_hat)
    S > 0 || error("zero total weight")
    p_hat ./= S
    TV = 0.5 * sum(abs.(p_hat .- part.p_true))
    cov4 = count(k -> p_hat[k] >= 0.25 * part.p_true[k], 1:part.n_regions) / part.n_regions
    return TV, cov4, p_hat, N
end

# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------
function multinomial_floor(part::Partition, N::Int; reps::Int = 2000, rng = Xoshiro(20260925))
    tvs = Vector{Float64}(undef, reps)
    covs = Vector{Float64}(undef, reps)
    mn = Multinomial(N, part.p_true)
    for r in 1:reps
        c = rand(rng, mn)
        ph = c ./ N
        tvs[r] = 0.5 * sum(abs.(ph .- part.p_true))
        covs[r] = count(k -> ph[k] >= 0.25 * part.p_true[k], 1:part.n_regions) / part.n_regions
    end
    return (; mean_TV = mean(tvs), median_TV = median(tvs), q025_TV = quantile(tvs, 0.025), q975_TV = quantile(tvs, 0.975),
              mean_cov4 = mean(covs), min_cov4 = minimum(covs))
end

function calibrate(parts::Dict)
    rows = Dict{Symbol,Any}[]
    for ((prob, d), part) in sort(collect(parts); by = x -> (String(x[1][1]), x[1][2]))
        truth = load_truth(joinpath(OUT, "truth", @sprintf("%s_d%d.h5", String(prob), d)))
        Nt = size(truth.samples, 2)
        TVt, cov4t, p_hat_t, _ = partition_mass(part, truth.samples, Float64[])
        maxdev = maximum(abs.(p_hat_t .- part.p_true))
        # empirical floor from disjoint 10^4 subsets of the truth draws
        nsub = min(100, Nt ÷ 10_000)
        tv_sub = [partition_mass(part, view(truth.samples, :, ((s - 1) * 10_000 + 1):(s * 10_000)), Float64[])[1] for s in 1:nsub]
        cov_sub = [partition_mass(part, view(truth.samples, :, ((s - 1) * 10_000 + 1):(s * 10_000)), Float64[])[2] for s in 1:nsub]
        m4 = multinomial_floor(part, 10_000)
        m5 = multinomial_floor(part, 100_000)
        m6 = multinomial_floor(part, Nt)
        push!(rows, Dict(:problem => String(prob), :d => d, :n_regions => part.n_regions, :N_truth => Nt,
            :TV_truth_full => TVt, :cov4_truth_full => cov4t, :max_abs_dev_p_truth_vs_ref => maxdev,
            :multinomial_q975_TV_at_N_truth => m6.q975_TV,
            :ref_check => maxdev <= 4 * sqrt(maximum(part.p_true) / Nt) ? "pass" : "FAIL",
            :TV_truth_1e4_subsets_mean => mean(tv_sub), :TV_truth_1e4_subsets_max => maximum(tv_sub), :n_subsets => nsub,
            :cov4_truth_1e4_subsets_mean => mean(cov_sub), :cov4_truth_1e4_subsets_min => minimum(cov_sub),
            :multinomial_1e4_mean_TV => m4.mean_TV, :multinomial_1e4_q975_TV => m4.q975_TV, :multinomial_1e4_mean_cov4 => m4.mean_cov4,
            :multinomial_1e5_mean_TV => m5.mean_TV, :multinomial_1e5_q975_TV => m5.q975_TV))
        @printf("calibration %-14s d=%2d regions=%5d | truth 1e6: TV=%.5f cov4=%.4f maxdev=%.2e (%s) | 1e4 subsets: TV mean=%.4f max=%.4f cov4 mean=%.4f | multinomial 1e4: TV mean=%.4f q975=%.4f cov4=%.4f\n",
            String(prob), d, part.n_regions, TVt, cov4t, maxdev, rows[end][:ref_check], mean(tv_sub), maximum(tv_sub), mean(cov_sub), m4.mean_TV, m4.q975_TV, m4.mean_cov4)
    end
    return DataFrame(rows)
end

# -----------------------------------------------------------------------------
# Compute for all archived cells (+ fresh MW)
# -----------------------------------------------------------------------------
function compute_all(parts::Dict; with_fresh::Bool = false)
    rows = Dict{Symbol,Any}[]
    cells = DataFrame(CSV.File(joinpath(OUT, "tables", "cells.csv")))
    fresh_summ = Dict{String,Any}()
    if with_fresh && isdir(FRESH_DIR)
        for f in readdir(FRESH_DIR)
            endswith(f, ".jld2") || continue
            fresh_summ[replace(f, ".jld2" => "")] = JLD2.jldopen(joinpath(FRESH_DIR, f), "r") do h; h["summary"]; end
        end
    end
    for ((prob, d), part) in sort(collect(parts); by = x -> (String(x[1][1]), x[1][2]))
        for alg in (:is, :mh, :nuts, :ns, :mw), B in BUDGETS, seed in SEED_GRID
            dir = cell_dir(OUT, prob, alg, d, B, seed)
            isfile(joinpath(dir, "result.h5")) || continue
            mr = load_method_result(dir)
            crow = cells[(cells.problem .== String(prob)) .& (cells.algorithm .== String(alg)) .& (cells.d .== d) .& (cells.B .== B) .& (cells.seed .== seed), :]
            notes = nrow(crow) == 1 ? String(coalesce(crow.notes[1], "")) : ""
            infeasible = occursin("budget-infeasible", notes) || size(mr.samples, 2) <= 1
            base = Dict{Symbol,Any}(:problem => String(prob), :d => d, :B => B, :seed => seed, :algorithm => String(alg),
                :notes => notes, :admissible => !(occursin("RHAT-FAIL", notes) || occursin("BUDGET-VIOLATION", notes) || infeasible),
                :n_regions => part.n_regions, :mode_recovery_v1 => nrow(crow) == 1 ? crow.mode_recovery[1] : NaN)
            if infeasible
                r = copy(base); r[:estimator] = "population"; r[:TV_partition] = NaN; r[:coverage_quarter] = NaN; r[:N_points] = 0
                push!(rows, r); continue
            end
            TV, cov4, _, N = partition_mass(part, mr.samples, mr.weights)
            r = copy(base); r[:estimator] = alg === :mw ? "population" : "output"; r[:TV_partition] = TV; r[:coverage_quarter] = cov4; r[:N_points] = N
            push!(rows, r)
            if alg === :mw && with_fresh
                name = basename(dir)
                s = get(fresh_summ, name, nothing)
                if s !== nothing && get(s, "status", "") in ("full_remaining_budget", "capped_within_budget") && get(s, "weight_status", "") == "ok"
                    mix = mr.extras[:mixture]
                    cfg = _make_cfg(prob, d)
                    counter = LikelihoodCounter(build_log_f(cfg)); reset!(counter)
                    L = cfg.L
                    nfresh = Int(s["N_fresh"])
                    Z, logq, logp = fresh_core(mix, nothing, nfresh, Xoshiro(Int(s["rng_seed"])); fast_target = (counter, L))
                    logw = logp .- logq
                    lz = _logsumexp(logw) - log(nfresh)
                    abs(lz - s["logZ_fresh"]) < 1e-9 || @warn "regenerated fresh draws differ from the recorded run" name lz s["logZ_fresh"]
                    theta = 2L .* _normcdf.(Z) .- L
                    TVf, cov4f, _, Nf = partition_mass(part, theta, exp.(logw .- maximum(logw)))
                    rf = copy(base); rf[:estimator] = "fresh"; rf[:TV_partition] = TVf; rf[:coverage_quarter] = cov4f; rf[:N_points] = Nf
                    rf[:regenerated_logZ_matches] = abs(lz - s["logZ_fresh"]) < 1e-9
                    push!(rows, rf)
                end
            end
        end
        println("computed $(String(prob)) d=$d")
        flush(stdout)
    end
    df = DataFrame(rows)
    return df
end

# -----------------------------------------------------------------------------
# Paired tests at the headline setting (d = 5, B = 5e5), admissible seeds
# -----------------------------------------------------------------------------
function paired_stats(df::DataFrame)
    rows = Dict{Symbol,Any}[]
    for prob in PART_PROBLEMS, est in ("population", "fresh")
        sub = df[(df.problem .== String(prob)) .& (df.d .== 5) .& (df.B .== 5e5) .& df.admissible, :]
        mw = sub[(sub.algorithm .== "mw") .& (sub.estimator .== est), :]
        isempty(mw) && continue
        for alg in ("is", "mh", "nuts", "ns")
            other = sub[(sub.algorithm .== alg) .& (sub.estimator .== "output"), :]
            common = intersect(mw.seed, other.seed)
            n = length(common)
            a = [mw.TV_partition[findfirst(==(s), mw.seed)] for s in common]
            b = [other.TV_partition[findfirst(==(s), other.seed)] for s in common]
            if n >= 3
                p = wilcoxon_signed_rank(a, b; tail = :both)[2]   # two-sided exact p
                δ = cliffs_delta(a, b)
            else
                p = NaN; δ = NaN
            end
            push!(rows, Dict(:problem => String(prob), :mw_estimator => est, :comparator => alg, :n_pairs => n,
                :median_TV_mw => isempty(a) ? NaN : median(a), :median_TV_comparator => isempty(b) ? NaN : median(b),
                :wilcoxon_p => p, :cliffs_delta => δ))
        end
    end
    out = DataFrame(rows)
    if nrow(out) > 0
        for g in groupby(out, [:problem, :mw_estimator])
            ps = Vector{Float64}(g.wilcoxon_p)
            ok = findall(isfinite, ps)
            adj = fill(NaN, length(ps))
            if !isempty(ok)
                adj[ok] = holm_bonferroni(ps[ok])
            end
            g[!, :holm_p] = adj
        end
    end
    return out
end

# -----------------------------------------------------------------------------
# Figure: TV vs budget, one curve per algorithm (medians over admissible
# seeds), MW population and MW fresh; i.i.d. floor at N = 10^4 as reference.
# -----------------------------------------------------------------------------
function fig_partition(df::DataFrame, calib::DataFrame, problem::Symbol; d::Int = 5)
    set_pub_theme!(class = :narrow)
    res = ExperimentsBase.figure_resolution(:narrow, :conv)
    fig = Figure(size = res)
    ax = Axis(fig[1, 1]; xlabel = tex_label(:budget),
        ylabel = L"\mathrm{TV}_{\mathrm{part}}\;\;(\text{partition mass error})", xscale = log10, yscale = log10, title = "")
    standard_axis!(ax)
    sub = df[(df.problem .== String(problem)) .& (df.d .== d) .& df.admissible, :]
    leg = []; labels = String[]
    for alg in ALG_ORDER
        for est in (alg === :mw ? ("population", "fresh") : ("output",))
            s = sub[(sub.algorithm .== String(alg)) .& (sub.estimator .== est), :]
            isempty(s) && continue
            g = combine(groupby(s, :B), :TV_partition => (v -> median(filter(isfinite, v))) => :med)
            sort!(g, :B)
            xs = Float64.(g.B); ys = Float64.(g.med); ok = isfinite.(ys)
            any(ok) || continue
            ls = est == "fresh" ? :dot : ALG_LINESTYLE[alg]
            ln = lines!(ax, xs[ok], ys[ok]; color = ALG_COLOR[alg], linewidth = ALG_LINEWIDTH[alg], linestyle = ls)
            scatter!(ax, xs[ok], ys[ok]; color = est == "fresh" ? :white : ALG_COLOR[alg],
                strokecolor = ALG_COLOR[alg], strokewidth = est == "fresh" ? 1.5 : 0, marker = ALG_MARKER[alg], markersize = 9)
            push!(leg, ln); push!(labels, est == "fresh" ? "MW fresh" : (est == "population" ? "MW pop." : ALG_LABEL[alg]))
        end
    end
    c = calib[(calib.problem .== String(problem)) .& (calib.d .== d), :]
    if nrow(c) == 1
        hl = hlines!(ax, [c.multinomial_1e4_q975_TV[1]]; color = :gray50, linestyle = :dash, linewidth = 1)
        push!(leg, hl); push!(labels, "i.i.d. floor")
    end
    ylims!(ax, 6e-4, 1.5)
    Bs = sort!(unique(Float64.(sub.B)))
    isempty(Bs) || (ax.xticks = (Bs, [LaTeXString("\$" * ExperimentsBase._budget_latex(b) * "\$") for b in Bs]); ax.xminorticksvisible = false)
    isempty(leg) || Legend(fig[2, 1], leg, labels; orientation = :horizontal, nbanks = 3, framevisible = false,
        labelsize = 7, patchsize = (14, 4), colgap = 8, rowgap = 0, padding = (0, 0, 0, 0))
    return fig
end

# -----------------------------------------------------------------------------
# Tests (calibration gates 1-4)
# -----------------------------------------------------------------------------
function run_partition_tests(parts::Dict)
    println("== gate 1: sums to one, every point in exactly one region")
    for ((prob, d), part) in parts
        @assert abs(sum(part.p_true) - 1) < 1e-12
        rng = Xoshiro(1); X = 10 .* (rand(rng, d, 5000) .- 0.5)
        idx = [region_index(part, view(X, :, i)) for i in 1:5000]
        @assert all(1 .<= idx .<= part.n_regions)
        TV, cov4, ph, _ = partition_mass(part, X, Float64[])
        @assert abs(sum(ph) - 1) < 1e-12
    end
    println("   ok")
    part = parts[(:mridges_spiky, 5)]; pm = parts[(:mridges, 5)]
    println("== gate 2: permutation and global weight-scaling invariance")
    rng = Xoshiro(2); X = randn(rng, 5, 4000) .* 1.5; w = rand(rng, 4000)
    for p in (part, pm)
        t1 = partition_mass(p, X, w)[1]
        perm = randperm(rng, 4000)
        t2 = partition_mass(p, X[:, perm], w[perm])[1]
        t3 = partition_mass(p, X, 1e7 .* w)[1]
        @assert abs(t1 - t2) < 1e-12 && abs(t1 - t3) < 1e-12
    end
    println("   ok")
    println("== gate 3: duplicate splitting (theta, w) -> m copies of (theta, w/m) leaves TV unchanged")
    for p in (part, pm)
        t1 = partition_mass(p, X, w)[1]
        Xd = repeat(X, 1, 3); wd = repeat(w ./ 3, 3)
        t2 = partition_mass(p, Xd, wd)[1]
        @assert abs(t1 - t2) < 1e-12
        ess1 = sum(w)^2 / sum(w .^ 2); ess2 = sum(wd)^2 / sum(wd .^ 2)
        @assert abs(ess2 - 3ess1) < 1e-6   # Kish ESS is not invariant (documented contrast)
    end
    println("   ok (Kish ESS triples under the same operation, as expected)")
    println("== gate 4: fixture with one point per region weighted p_true gives TV = 0; deleting region j gives TV = p_true[j]")
    for p in (part, pm)
        X = zeros(p.d, p.n_regions); w = copy(p.p_true)
        if p.problem === :mridges
            for k in 0:(p.n_regions - 1), j in 1:p.d
                X[j, k + 1] = ((k >> (j - 1)) & 0x1) == 1 ? -1.0 : 1.0
            end
        else
            X[1, :] .= p.maxima
        end
        @assert partition_mass(p, X, w)[1] < 1e-12
        j = 3
        keep = setdiff(1:p.n_regions, j)
        TVj = partition_mass(p, X[:, keep], w[keep])[1]
        @assert abs(TVj - p.p_true[j]) < 1e-12 "TV after deleting region $j: $TVj vs $(p.p_true[j])"
    end
    println("   ok")
    println("== spiky geometry: maxima and boundaries")
    println("   maxima     = ", round.(part.maxima; digits = 4))
    println("   boundaries = ", round.(part.boundaries; digits = 4))
    println("   p_true     = ", round.(part.p_true; digits = 5))
    println("== mridges reference: ", pm.notes)
    println("all partition tests passed")
end

# -----------------------------------------------------------------------------
function main_partition(args)
    do_test = "--test" in args; do_compute = "--compute" in args; with_fresh = "--with-fresh" in args
    do_figs = "--figures" in args; do_stats = "--stats" in args
    isdir(REV) || mkpath(REV)
    parts = Dict{Tuple{Symbol,Int},Partition}()
    for d in (2, 5, 10); parts[(:mridges, d)] = build_partition(:mridges, d); end
    parts[(:mridges_spiky, 5)] = build_partition(:mridges_spiky, 5)
    ref_rows = Dict{Symbol,Any}[]
    for ((prob, d), part) in parts, k in 1:part.n_regions
        push!(ref_rows, Dict(:problem => String(prob), :d => d, :region => k, :p_true => part.p_true[k],
            :lower_theta1 => prob === :mridges_spiky ? (k == 1 ? -_make_cfg(prob, d).L : part.boundaries[k - 1]) : NaN,
            :upper_theta1 => prob === :mridges_spiky ? (k == part.n_regions ? _make_cfg(prob, d).L : part.boundaries[k]) : NaN,
            :maximum_theta1 => prob === :mridges_spiky ? part.maxima[k] : NaN, :definition => part.notes))
    end
    CSV.write(joinpath(REV, "partition_reference.csv"), sort(DataFrame(ref_rows), [:problem, :d, :region]))
    do_test && run_partition_tests(parts)
    if do_compute
        calib = calibrate(parts)
        CSV.write(joinpath(REV, "partition_calibration.csv"), calib)
        df = compute_all(parts; with_fresh = with_fresh)
        CSV.write(joinpath(REV, "partition_mass.csv"), df)
        println("wrote partition_mass.csv with $(nrow(df)) rows")
    end
    if do_stats
        df = DataFrame(CSV.File(joinpath(REV, "partition_mass.csv")))
        st = paired_stats(df)
        CSV.write(joinpath(REV, "partition_tests.csv"), st)
        println(st)
    end
    if do_figs
        df = DataFrame(CSV.File(joinpath(REV, "partition_mass.csv")))
        calib = DataFrame(CSV.File(joinpath(REV, "partition_calibration.csv")))
        figdir = joinpath(REV, "figs"); isdir(figdir) || mkpath(figdir)
        for prob in PART_PROBLEMS
            fig = fig_partition(df, calib, prob; d = 5)
            save_pdf(fig, fig_filename(family = :partition, problem = prob, d = 5, B = :all, alg = :all, extra = "all"); dir = figdir)
            println("figure partition $prob written")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_partition(ARGS)
end

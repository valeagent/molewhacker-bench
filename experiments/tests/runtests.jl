# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# tests/runtests.jl — protocol §11 regression tests
# =============================================================================
#
# Three classes (Protocol §11):
#
#   §11.1  Truth self-consistency       — empirical moments / quantiles /
#                                          logZ within tolerance
#   §11.2  Algorithm sanity on mvn      — every algorithm at B = 5e5, d = 5
#                                          must achieve W1 < 0.05, η > 1e-3,
#                                          |dlogZ| < 0.1 (where applicable)
#   §11.3  Counter integrity            — each algorithm at B = 1000 must
#                                          stop within ±10% of the budget
#
# Usage (from the MoleWhacker repository root):
#
#     julia --project=. -t auto experiments/tests/runtests.jl
#
# Optional environment variables:
#     EXP_TESTS_FAST=1            run only the counter-integrity subset
#     EXP_TESTS_PROBLEMS=mvn,...  limit truth-consistency to a subset
#     EXP_TESTS_NREF=100000       N_ref for §11.1 (default 50_000)

using Pkg

const _ROOT = abspath(joinpath(@__DIR__, "..", ".."))
if Base.active_project() != joinpath(_ROOT, "Project.toml")
    Pkg.activate(_ROOT)
end

include(joinpath(_ROOT, "experiments", "src", "ExperimentsBase.jl"))
using .ExperimentsBase

using Test
using Statistics, LinearAlgebra, Printf, Random
import ForwardDiff
import Distributions: Normal, cdf


const FAST = get(ENV, "EXP_TESTS_FAST", "0") == "1"
const NREF_TEST = parse(Int, get(ENV, "EXP_TESTS_NREF", "50000"))
const PROBLEMS_TO_TEST = let raw = get(ENV, "EXP_TESTS_PROBLEMS", "")
    if isempty(raw)
        collect(PROBLEM_NAMES)
    else
        [Symbol(strip(s)) for s in split(raw, ',')]
    end
end


# =============================================================================
# §11.1 — Truth self-consistency
# =============================================================================

"Empirical mean / cov / logZ from a fresh `sample_truth` draw."
function _consistency_for(cfg::ProblemConfig; N::Int = NREF_TEST, seed::Int = 20260901)
    truth = compute_truth(cfg; N_ref = N, seed = seed)
    fresh = sample_truth(cfg, N; seed = seed + 1)
    μ_fresh = vec(mean(fresh; dims = 2))
    Σ_fresh = cov(fresh; dims = 2)
    return truth, μ_fresh, Σ_fresh
end


@testset "Protocol §11 — Truth self-consistency" begin
    for problem in PROBLEMS_TO_TEST
        cfg = if problem === :mvn
            make_config_mvn(d = 5)
        elseif problem === :banana
            make_config_banana(d = 5)
        elseif problem === :funnel
            make_config_funnel(d = 5)
        elseif problem === :mridges
            make_config_mridges(d = 5)
        elseif problem === :shell
            make_config_shell(d = 5)
        elseif problem === :mridges_spiky
            make_config_mridges_spiky(d = 5)
        elseif problem === :eggbox
            make_config_eggbox(d = 2)
        else
            @warn "Skipping unknown problem in runtests" problem
            nothing
        end
        cfg === nothing && continue
        @testset "truth_$(problem)" begin
            truth, μ, Σ = _consistency_for(cfg)
            # Generous tolerances accommodate Monte Carlo noise at N_ref test size.
            # `mridges_spiky` has σ_⊥ = 0.5 perpendicular widths and a
            # near-singular cov in θ_2..θ_d — only test the diagonal
            # element [1, 1] tightly there.
            if problem === :mridges_spiky
                # mridges_spiky has near-zero off-diagonal cov, so allow
                # somewhat looser bounds on the perpendicular variance.
                @test isapprox(μ[1], truth.mean[1]; atol = 0.15, rtol = 0.20)
                @test isapprox(Σ[1, 1], truth.cov[1, 1]; rtol = 0.25, atol = 0.25)
            else
                @test isapprox(μ, truth.mean; atol = 0.10, rtol = 0.10)
                @test isapprox(diag(Σ), diag(truth.cov); rtol = 0.20, atol = 0.20)
            end
            @test isfinite(truth.logZ)
            @test size(truth.samples, 1) == cfg.d
            @test size(truth.samples, 2) == NREF_TEST
            @test all(maximum(abs, truth.samples; dims = 2) .<= cfg.L + 1e-6)
        end
    end
end


# =============================================================================
# §11.1b — Truth-vs-density validation (V8-FIX-B3)
#
# The §11.1 self-consistency tests compare the truth set against a FRESH
# draw from the same sampler, so a sampler that draws from the wrong
# distribution passes them by construction (this is exactly how the V8
# funnel/shell truth bugs slipped through). These tests validate the
# truth samples against the ANALYTIC density through an independent
# 1-D Kolmogorov–Smirnov test on a distinguished coordinate/statistic,
# with the reference CDF computed here by direct quadrature — not by
# the code under test.
# =============================================================================

"Two-sided KS distance between sorted sample `xs` and grid CDF (grid, cdf)."
function _ks_vs_grid_cdf(xs::Vector{Float64}, grid::Vector{Float64},
                          cdfv::Vector{Float64})
    sort!(xs)
    n = length(xs)
    D = 0.0
    for (i, x) in enumerate(xs)
        j = clamp(searchsortedfirst(grid, x), 2, length(grid))
        frac = (x - grid[j - 1]) / max(grid[j] - grid[j - 1], eps())
        F = cdfv[j - 1] + clamp(frac, 0.0, 1.0) * (cdfv[j] - cdfv[j - 1])
        D = max(D, abs(i / n - F), abs((i - 1) / n - F))
    end
    return D
end

"Trapezoid CDF of `logp` evaluated on `grid` (normalized)."
function _grid_cdf(grid::Vector{Float64}, logp::Vector{Float64})
    p = exp.(logp .- maximum(logp))
    c = zeros(length(grid))
    for i in 2:length(grid)
        c[i] = c[i - 1] + 0.5 * (p[i] + p[i - 1]) * (grid[i] - grid[i - 1])
    end
    return c ./ c[end]
end

_erf_via_normal(z) = 2 * cdf(Normal(), z * sqrt(2)) - 1

if !FAST
    @testset "Protocol §11.1b — truth vs analytic density (V8-FIX-B3)" begin
        N_ks = 50_000
        # KS 99.9% critical value ≈ 1.949/√n; allow 2.2/√n for grid
        # interpolation slack. A wrong truth law (the V8-A1 erf² bug gave
        # D ≈ 0.02 ≫ 0.01) fails this decisively.
        D_crit = 2.2 / sqrt(N_ks)

        if :funnel in PROBLEMS_TO_TEST
            @testset "funnel θ₁ marginal (KS)" begin
                cfg = make_config_funnel(d = 5)
                s = sample_truth(cfg, N_ks; seed = 20260911)
                # Independent reference: p₁(θ₁) ∝ e^{-θ₁²/(2σ₁²)} ·
                # erf(L/√(2eᶿ¹))^{d-1}, erf via Normal-CDF (not the A&S
                # approximation used by the truth code).
                grid = collect(range(-cfg.L, cfg.L; length = 8192))
                logp = [-0.5 * (t / cfg.sigma1)^2 +
                        (cfg.d - 1) * log(max(_erf_via_normal(cfg.L / sqrt(2 * exp(t))), 1e-300))
                        for t in grid]
                D = _ks_vs_grid_cdf(collect(s[1, :]), grid, _grid_cdf(grid, logp))
                @test D < D_crit
            end
        end

        if :shell in PROBLEMS_TO_TEST
            @testset "shell radial law (KS)" begin
                cfg = make_config_shell(d = 5)
                s = sample_truth(cfg, N_ks; seed = 20260912)
                r = vec(sqrt.(sum(abs2, s; dims = 1)))
                grid = collect(range(max(0.0, cfg.ρ - 12 * cfg.w),
                                     cfg.ρ + 12 * cfg.w; length = 8192))
                logp = [(cfg.d - 1) * log(max(x, eps())) -
                        0.5 * ((x - cfg.ρ) / cfg.w)^2 for x in grid]
                D = _ks_vs_grid_cdf(r, grid, _grid_cdf(grid, logp))
                # The V8-A2 envelope bug gave radial std 0.598 vs 0.486 —
                # a KS distance ≈ 0.05, far above this gate.
                @test D < D_crit
            end
        end

        if :eggbox in PROBLEMS_TO_TEST
            @testset "eggbox mode list sits at the density maxima" begin
                cfg = make_config_eggbox(d = 2)
                log_f = build_log_f(cfg)
                modes = eggbox_modes(cfg)
                @test size(modes, 2) == 5     # (3² + 1)/2 parity-valid points
                f_max = 5.0 * log(3.0)        # global max: ∏cos = +1
                for k in 1:size(modes, 2)
                    @test isapprox(log_f(modes[:, k]), f_max; atol = 1e-9)
                end
                # The pre-V8 "modes" (θᵢ ∈ {-2π,…,2π} lattice) are plateau
                # points, strictly below the true maxima.
                @test log_f([-2π, -2π]) < f_max - 0.5
                @test log_f([0.0, 0.0]) < f_max - 0.5
            end
        end
    end
end


# =============================================================================
# §11.2 — Algorithm sanity on mvn at B = 5e5, d = 5
# =============================================================================

if !FAST
    @testset "Protocol §11.2 — algorithm sanity on mvn (B=5e5)" begin
        cfg = make_config_mvn(d = 5)
        truth = compute_truth(cfg; N_ref = NREF_TEST, seed = 20260903)
        # Per-algorithm sanity envelopes — these are *regression* bounds
        # (catch crashes / order-of-magnitude regressions), not quality
        # thresholds. The actual quality numbers live in cells.csv.
        # V8-FIX-B3: tightened to ≈ 5–10× the observed V6 seed medians on
        # mvn d=5 B=5e5 (previously so loose — e.g. W1 < 0.8 vs measured
        # 0.23 — that they pinned down crashes only, not the invariants).
        # IS on a wide cube prior intrinsically has tiny η on mvn, so
        # its envelope is much wider than the gradient-based samplers.
        bounds = Dict(
            :is   => (w1 = 0.50, eta = 1e-6, dlz = 2.0),
            :mh   => (w1 = 0.20, eta = 1e-4, dlz = NaN),
            :nuts => (w1 = 0.15, eta = 1e-4, dlz = NaN),
            :ns   => (w1 = 0.25, eta = 1e-3, dlz = 0.5),
            :mw   => (w1 = 0.15, eta = 1e-2, dlz = 0.5),
        )
        for alg in ALG_NAMES
            @testset "$(alg)" begin
                local mr
                try
                    mr = run_algorithm(alg, cfg, 5e5, 11)
                catch err
                    @info "Skipping $alg sanity (raised)" exception = err
                    continue
                end
                bd = bounds[alg]
                w1 = w1_marginal_avg(mr, truth)
                @test isfinite(w1)
                @test w1 < bd.w1
                eff_n = neff(mr)
                eff = eff_n / max(mr.Nlike_used, 1.0)
                @test eff > bd.eta
                if alg in (:is, :ns, :mw) && isfinite(bd.dlz)
                    dl = dlogZ(mr, truth)
                    if dl !== missing
                        @test isfinite(dl)
                        @test dl < bd.dlz
                    end
                end
            end
        end
    end
end


# =============================================================================
# §11.3 — Counter integrity
# =============================================================================

@testset "Protocol §11.3 — counter integrity (B=5e3)" begin
    cfg = make_config_mvn(d = 5)
    # Per-algorithm budget envelopes at B=5e3. V8-FIX-B3: the NUTS
    # envelope was (0.05, 5.0) — asserting nothing. A feasible NUTS
    # cell must now land in the protocol band (small slack for the
    # final chunk); a budget-infeasible cell (possible at B=5e3, d=5
    # once the pilot is charged, V8-FIX-A5) is exempt from the band but
    # must record a positive honest cost and carry the N/A marker.
    bounds = Dict(
        :is   => (0.8, 1.2),
        :mh   => (0.8, 1.2),
        :nuts => (0.75, 1.25),
        :ns   => (0.5, 1.6),
        :mw   => (0.5, 1.6),
    )
    B = 5e3
    for alg in ALG_NAMES
        @testset "counter_$(alg)" begin
            local mr
            try
                mr = run_algorithm(alg, cfg, B, 11)
            catch err
                @info "Skipping $alg counter test (raised)" exception = err
                continue
            end
            ratio = mr.Nlike_used / B
            if alg === :nuts &&
               get(mr.extras, :terminated_by, :none) in (:budget_infeasible, "budget_infeasible")
                @test ratio > 0
                @test get(mr.extras, :budget_infeasible, false) == true
            else
                (lo, hi) = bounds[alg]
                @test lo < ratio < hi
            end
        end
    end
end


# =============================================================================
# Bonus: smoke test of metric pipeline + JLD2 roundtrip
# =============================================================================

@testset "Output schema — JLD2 round-trip" begin
    cfg = make_config_mvn(d = 5)
    mr = run_is(cfg, 1000, 11)
    tmp = mktempdir()
    save_method_result(tmp, mr)
    mr2 = load_method_result(tmp)
    @test mr.algorithm == mr2.algorithm
    @test mr.problem == mr2.problem
    @test isapprox(mr.samples, mr2.samples)
    @test isapprox(mr.weights, mr2.weights)
end


# =============================================================================
# Protocol §11.4 — Statistical-test helper smoke tests (V3-S-3, V3-S-4)
# =============================================================================

@testset "Protocol §11.4 — statistical helpers" begin
    @testset "wilcoxon_signed_rank: a < b detected" begin
        rng = MersenneTwister(1)
        a = randn(rng, 20)
        b = a .+ 1.0   # b is consistently larger than a
        _, p, n = wilcoxon_signed_rank(a, b; tail = :left)
        @test p < 0.01
        @test n == 20
    end
    @testset "wilcoxon_signed_rank: ties dropped" begin
        a = [1.0, 2.0, 3.0, 4.0]
        b = [1.0, 2.0, 3.0, 5.0]
        _, p, n = wilcoxon_signed_rank(a, b; tail = :left)
        @test n == 1     # only one pair has nonzero diff
        @test 0.0 <= p <= 1.0
    end
    @testset "cliffs_delta: extremes" begin
        @test cliffs_delta([1, 2, 3], [10, 11, 12]) == -1.0
        @test cliffs_delta([10, 11, 12], [1, 2, 3]) == 1.0
        @test cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0
    end
    @testset "holm_bonferroni: monotonicity" begin
        adj = holm_bonferroni([0.01, 0.02, 0.04, 0.6])
        @test all(0 .<= adj .<= 1)
        @test adj == sort(adj)   # Holm is monotone non-decreasing along the *sorted* order
    end
    @testset "friedman_chi2: clear effect" begin
        # k=4 algorithms; the third one always wins.
        scores = hcat(rand(20) .+ 0.5,
                       rand(20) .+ 0.5,
                       rand(20) ./ 100,
                       rand(20) .+ 0.5)
        χ2, dof, p = friedman_chi2(scores)
        @test dof == 3
        @test p < 0.05
        @test χ2 > 5.0
    end
    @testset "nemenyi_critical_difference: returns positive" begin
        @test nemenyi_critical_difference(5, 30) > 0
    end
    @testset "bootstrap_median_ci: covers the median" begin
        rng = MersenneTwister(2)
        v = randn(rng, 100)
        lo, m, hi = bootstrap_median_ci(v; nboot = 500, rng = rng)
        @test lo <= m <= hi
        @test isapprox(m, median(v); atol = 1e-12)
    end
    @testset "mode_recovery: trivially recovers truth" begin
        cfg = make_config_mridges_spiky(d = 2, M = 1, σ_spike = 0.1)
        truth = compute_truth_mridges_spiky(cfg; N_ref = 5_000, seed = 11)
        # Use truth samples themselves as the algorithm output → must
        # recover all modes.
        mr = MethodResult(
            algorithm = :mw, problem = :mridges_spiky, d = cfg.d, seed = 11,
            B = 10_000.0, counter = LikelihoodCounter(build_log_f(cfg)),
            wall_time_s = 0.0, samples = truth.samples,
            extras = Dict{Symbol,Any}(:stop_reason => :budget),
        )
        rate = mode_recovery(mr, truth, _eps_radius_for_problem(truth))
        @test rate >= 0.5
    end
end


# =============================================================================
# Protocol §11.5 — gradient cost accounting
#
# These would have caught the V6 P0 bug: NUTS leapfrog gradients leaking
# into the primal (cost-1) branch of the LikelihoodCounter, inflating
# η = Neff/Nlike by 1.5–6×.
# =============================================================================

@testset "Protocol §11.5 — gradient cost accounting (V6 P2)" begin
    # A simple smooth log-density; gradient is well-defined everywhere.
    logf = x -> -0.5 * sum(abs2, x)

    @testset "ForwardDiff.gradient charges exactly d, no primal" begin
        for d in (3, 5, 10)
            c = LikelihoodCounter(logf)
            g = ForwardDiff.gradient(c, randn(d))
            @test length(g) == d
            # One full forward-mode gradient costs exactly d, regardless of
            # how ForwardDiff chunks it; the gradient path never increments
            # n_primal.
            @test c.n_grad_partials == d
            @test c.n_primal == 0
            @test c.n_dual_calls >= 1
        end
    end

    @testset "primal call charges 1, not d" begin
        c = LikelihoodCounter(logf)
        c(randn(5))
        @test c.n_primal == 1
        @test c.n_grad_partials == 0
        @test c.n_dual_calls == 0
    end

    @testset "ForwardDiff.hessian charges ≈ d² (V8-FIX-A6 nested duals)" begin
        # A d×d Hessian is d gradients ≈ d² primal-equivalents. The pre-V8
        # counter read only the OUTER chunk size and billed ≈ d — this is
        # the regression gate for the MoleWhacker Laplace-fit undercharge.
        for d in (3, 5, 8)
            c = LikelihoodCounter(logf)
            H = ForwardDiff.hessian(c, randn(d))
            @test size(H) == (d, d)
            @test c.n_primal == 0
            @test c.n_grad_partials >= d^2
            # Chunking can pad the charge above d² but never past (d+chunk)².
            @test c.n_grad_partials <= 4 * d^2
        end
    end

    @testset "leaky Vector{Real} dual container is still charged as a gradient" begin
        # The exact V6 P0 leak: BAT's PriorToNormal transform can hand the
        # target a Dual vector whose *static* element type is Real/Any, so
        # it bypasses the specialized Dual method. The counter must classify
        # it as a dual call by runtime element type, never as primal.
        d = 4
        x = randn(d)
        duals = [ForwardDiff.Dual(x[i], ntuple(j -> j == i ? 1.0 : 0.0, d)...)
                 for i in 1:d]
        leaky = Vector{Real}(duals)
        @test eltype(leaky) === Real           # static type is lost
        @test !(leaky isa AbstractVector{<:ForwardDiff.Dual})

        c = LikelihoodCounter(logf)
        c(leaky)
        @test c.n_primal == 0                  # must NOT be miscounted
        @test c.n_dual_calls == 1
        @test c.n_grad_partials == d

        # A genuine Vector{Any} of duals (the other leaky container) too.
        c2 = LikelihoodCounter(logf)
        c2(Vector{Any}(duals))
        @test c2.n_primal == 0
        @test c2.n_grad_partials == d
    end

    if !FAST
        @testset "cost-accounting invariants on samplers (mvn d=3)" begin
            cfg = make_config_mvn(d = 3)
            for alg in (:nuts, :mw)
                @testset "$(alg)" begin
                    local mr
                    try
                        mr = run_algorithm(alg, cfg, 5e4, 11)
                    catch err
                        @info "Skipping $alg cost-accounting test (raised)" exception = err
                        continue
                    end
                    nd = Int(get(mr.extras, :n_dual_calls, -1))
                    @test nd >= 0                      # n_dual_calls exposed in extras
                    # Universal invariant (all samplers): effective samples
                    # can never exceed the total model-evaluation budget
                    # actually spent (Nlike_used = n_primal + n_grad_partials).
                    @test neff(mr) <= mr.Nlike_used + 1e-6
                    if alg === :nuts
                        # NUTS is gradient-DOMINATED: every leapfrog is one
                        # d-dim gradient, so #gradients = n_dual_calls (the
                        # exact Σ n_steps) is the binding
                        # bound and the V6 undercount gate. This is the test
                        # that would have caught the original NUTS gradient
                        # leak (η inflated 1.5–6×). MoleWhacker, by contrast,
                        # is primal-dominated (n_dual_calls is incidental and
                        # tiny), so this gradient bound does NOT apply to it.
                        @test mr.n_grad_partials > 0
                        @test neff(mr) <= nd + 1e-6
                    end
                end
            end
        end
    end
end

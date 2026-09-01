# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# MoleWhacker.jl — the MoleWhacker adaptive importance sampler.
# This is the exact module version used by every benchmark run reported
# in the thesis (loaded by experiments/src/algorithms/algo_mw.jl).
# =============================================================================
module MoleWhacker

export importance_sampling,
       local_MGVI_approx,
       merge_duplicate_components,
       make_prior_samples,
       make_init_samples,
       whack_a_mole,
       whack_many_moles, WhackLogEntry




using Distributions
using DataStructures
using DensityInterface
using Optimization, OptimizationOptimJL, ADTypes
using MeasureBase
using LinearAlgebra
using PositiveFactorizations
using PDMats
using InverseFunctions
using Logging
using StatsBase
using ArraysOfArrays
using FileIO
using JLD2

# AFTER (clean, no conflicts)
using BAT: bat_transform, PriorToGaussian, bat_sample, IIDSampling, bat_eff_sample_size,
           KishESS, bat_findmode, OptimizationAlg, ExplicitInit, set_batcontext,
           SobolSampler, DensitySampleVector
using ADTypes: AutoForwardDiff                # get AutoForwardDiff from ADTypes
const PriorToNormal = PriorToGaussian         # keep your alias; do NOT also import it
# using BAT: bat_transform, PriorToGaussian, bat_sample, IIDSampling, bat_eff_sample_size, KishESS, bat_findmode, OptimizationAlg, ExplicitInit, AutoForwardDiff, set_batcontext, bat_transform, PriorToNormal, bat_sample, IIDSampling, bat_eff_sample_size, KishESS, bat_findmode, OptimizationAlg, ExplicitInit, AutoForwardDiff, set_batcontext, SobolSampler, DensitySampleVector
# # alias the old name to the new one:
# const PriorToNormal = PriorToGaussian

using MGVI
using ForwardDiff
using ProgressMeter
#using StatsBase: weights

function importance_sampling(pstr, approx_dist, nsamples)
    smpls_q, _ = bat_sample(approx_dist, IIDSampling(nsamples = nsamples))
    x_q = smpls_q.v
    logd_q = smpls_q.logd;
    logd_p = similar(logd_q)
    #@showprogress 
    Threads.@threads for i in eachindex(x_q)
        logd_p[i] = logdensityof(pstr, x_q[i])
    end
    logw_raw = logd_p .- logd_q;
    w = exp.(logw_raw .- maximum(logw_raw));
    smpls_p = DensitySampleVector(x_q, logd_p, weight=w)
end

function local_MGVI_approx(pstr, θ_sel)
    try
        m_tr = pstr.likelihood.k
        FI_inner = MGVI.fisher_information(m_tr(θ_sel))
        J = ForwardDiff.jacobian(MGVI.flat_params ∘ m_tr, θ_sel)
        Σ_raw = inv(Matrix(J' * FI_inner * J + I))
        Σ = PDMat(cholesky(Positive, Σ_raw))
        return MvNormal(θ_sel, Σ)
    catch
        # Fallback for likelihoods without MGVI support (e.g. MixtureModel):
        # Laplace approximation via numerical Hessian of the log-posterior
        logp = x -> logdensityof(pstr, x)
        H = ForwardDiff.hessian(logp, θ_sel)
        Σ_raw = inv(Matrix(-H + 1e-6 * I))
        Σ = PDMat(cholesky(Positive, Σ_raw))
        return MvNormal(θ_sel, Σ)
    end
end



function make_prior_samples(posterior, nsamples=10_000)
    pstr, f_trafo = bat_transform(PriorToNormal(), posterior)
    pr_dist = MvNormal(zeros(pstr.prior.dist._dim), ones(pstr.prior.dist._dim))

    @info "Generating initial samples"
    smpls_p = importance_sampling(pstr, pr_dist, nsamples)

    (approx_dist=pr_dist, samples_p=smpls_p, samples_user=bat_transform(inverse(f_trafo), smpls_p).result)

end

"""
merge_duplicate_components(mix::MixtureModel{<:Multivariate, <:Continuous, MvNormal, <:Categorical}; tol_mean, tol_cov)

Merge components in a MixtureModel if their means and covariances are similar (using `isapprox`), 
by computing a weighted average of the mean and covariance matrices. Zero-weight components are dropped.
"""
function merge_duplicate_components(
    mix::MixtureModel{<:Multivariate, <:Continuous, MvNormal, <:Categorical};
    tol_mean::Float64 = 1e-3,
    tol_cov::Float64 = 1e-5,)
#tol_mean maybe even bigger for higher dimensions, e.g. 1e-2
    comps = mix.components
    ws = probs(mix)
    n = length(comps)


    merged = falses(n)
    new_components = MvNormal[]
    new_weights = Float64[]
    merged_clusters = 0
    merged_weight_total = 0.0

    for i in 1:n
        if merged[i]
            continue
        end

        μi, Σi = comps[i].μ, comps[i].Σ
        group_idxs = [i]

        for j in i+1:n
            if merged[j]
                continue
            end

            μj, Σj = comps[j].μ, comps[j].Σ
            # Mahalanobis distance between means (in Σi's metric)
            δμ = μi - μj
            mahal_dist² = dot(δμ, inv(Σi) * δμ)
            mean_close = mahal_dist² < tol_mean^2
            # Frobenius norm between covariances
            cov_dist = norm(Σi - Σj)  # this is Frobenius norm by default for matrices LinearAlgebra!
            cov_close = cov_dist < tol_cov
            # If both means and covariances are close, merge
            if mean_close && cov_close
                push!(group_idxs, j)
                merged[j] = true
            end
        end

        group_ws = ws[group_idxs]
        group_w = sum(group_ws)

        # If no merge, keep original
        if length(group_idxs) == 1
            push!(new_components, comps[i])
            push!(new_weights, group_w)
            continue
        end

        # Weighted mean
        μ̄ = sum(group_ws[k] * comps[group_idxs[k]].μ for k in eachindex(group_idxs)) / group_w

        # Weighted covariance + between-component spread
        Σ̄ = zeros(size(Σi))
        for k in eachindex(group_idxs)
            μk = comps[group_idxs[k]].μ
            Σk = comps[group_idxs[k]].Σ
            δμ = μk - μ̄
            Σ̄ += group_ws[k] * (Σk + δμ * δμ')
        end
        Σ̄ /= group_w

        push!(new_components, MvNormal(μ̄, Symmetric(Σ̄)))
        push!(new_weights, group_w)

        merged_clusters += 1
        merged_weight_total += group_w
    end

    total_before = length(mix.components)
    total_after = length(new_components)

    @info "merge_duplicate_components(): Merged $merged_clusters clusters (reduced $(total_before - total_after) components)"
    @info " - Before: $total_before components"
    @info " - After:  $total_after components"
    @info " - Total merged weight: $(round(merged_weight_total, digits=6))"

    return MixtureModel(new_components, new_weights)
end

function make_init_samples(posterior, nseeds=10, nsamples=10_000)
    @info "make_init_samples() called with nseeds=$nseeds, nsamples=$nsamples"
    pstr, f_trafo = bat_transform(PriorToNormal(), posterior)

    seeds = bat_sample(pstr.prior, SobolSampler(nsamples=nseeds)).result.v#[2:end]
    #@show seeds
    components = Array{MvNormal}(undef, nseeds)

    @info "Finding modes"
    
    Threads.@threads for i in 1:nseeds
        #@show pstr
        #@show seeds[i]
        adsel = AutoForwardDiff()
        set_batcontext(ad = adsel)
        r = bat_findmode(pstr, OptimizationAlg(optalg=OptimizationOptimJL.LBFGS(), init = ExplicitInit([seeds[i]])))
        #@show r.result
        components[i] = local_MGVI_approx(pstr, r.result)
    end

    #@show components
    approx_dist = MixtureModel(components)

    mode_logd_p_approx = [logdensityof(pstr, mode(ad)) for ad in approx_dist.components]
    mode_logd_q_approx = [logdensityof(approx_dist, mode(ad)) for ad in approx_dist.components]
    
    raw_mixture_logw = mode_logd_p_approx .- mode_logd_q_approx
    raw_mixture_w = exp.(raw_mixture_logw .- maximum(raw_mixture_logw))
    mixture_w = raw_mixture_w ./ sum(raw_mixture_w)

    approx_dist = MixtureModel(approx_dist.components, mixture_w)

    approx_dist = merge_duplicate_components(approx_dist)
    components = approx_dist.components
    #@show components

    @info "Generating initial samples"
    smpls_p = importance_sampling(pstr, approx_dist, nsamples)
    @info "Initial samples generated"
    (approx_dist=approx_dist, samples_p=smpls_p, samples_user=bat_transform(inverse(f_trafo), smpls_p).result)

end










# function whack_a_mole(posterior, init_samples, n_whack=100)
    
#     pstr, f_trafo = bat_transform(PriorToNormal(), posterior)
#     smpls_p = init_samples.samples_p
    
#     #μ = mean(smpls_p.v, ProbabilityWeights(smpls_p.weight))
#     #Σ = cov(Matrix(flatview(smpls_p.v)'), ProbabilityWeights(smpls_p.weight))
#     #approx_dist = MvNormal(μ, Σ)
#     approx_dist = init_samples.approx_dist

#     if init_samples.approx_dist isa MixtureModel
#         approx_mix = approx_dist
#         mode_logd_p_mix = [logdensityof(pstr, mode(approx_dist)) for approx_dist in approx_mix.components]
#         #mode_logd_q_mix = [logdensityof(approx_dist, mode(approx_dist)) for approx_dist in approx_mix.components]
#     else
#         approx_mix = Distributions.MixtureModel([approx_dist], [1])
#         mode_logd_p_mix = [logdensityof(pstr, mode(approx_dist))]
#         #mode_logd_q_mix = [logdensityof(approx_dist, mode(approx_dist))]
#     end
    
#     samples_mix = smpls_p
    

    
#     for n in 1:n_whack

#         ess = bat_eff_sample_size(samples_mix, KishESS()).result
#         @info "Effective sample size = $ess"
#         eff = ess / length(samples_mix)
#         @info "Efficiency = $eff"
        
#         θ_iter_idx = findmax(samples_mix.weight)[2]
        
#         θ_iter = samples_mix.v[θ_iter_idx]
        
#         approx_dist = local_MGVI_approx(pstr, θ_iter)
        
#         mode_logd_p_approx = logdensityof(pstr, mode(approx_dist))        
        
#         append!(mode_logd_p_mix, mode_logd_p_approx)

#         approx_mix = Distributions.MixtureModel(vcat(approx_mix.components, [approx_dist]))

#         mode_logd_q_mix = [logdensityof(approx_mix, mode(ad)) for ad in approx_mix.components]
        
#         raw_mixture_logw = mode_logd_p_mix .- mode_logd_q_mix
#         raw_mixture_w = exp.(raw_mixture_logw .- maximum(raw_mixture_logw))
#         mixture_w = raw_mixture_w ./ sum(raw_mixture_w)

#         approx_mix = MixtureModel(approx_mix.components, mixture_w)
        
#         new_nsamples = floor(Int, last(mixture_w) * length(samples_mix))
#         @info "Generating $new_nsamples new samples"
#         if new_nsamples > 0
#             smpls_p = importance_sampling(pstr, approx_dist, new_nsamples)
#             samples_mix = vcat(samples_mix, smpls_p)
        
#         end
        
#         #approx_mix = Distributions.MixtureModel(vcat(approx_mix.components, [approx_dist]), mixture_w)
        
#         logd_p = samples_mix.logd
#         logd_q = logdensityof.(Ref(approx_mix), samples_mix.v)
#         logw_raw = logd_p .- logd_q;
#         w = exp.(logw_raw .- maximum(logw_raw));
#         samples_mix.weight .= w;


#     end

#     (approx_dist=approx_mix, samples_p=samples_mix, samples_user=bat_transform(inverse(f_trafo), samples_mix).result)
    
# end


struct WhackLogEntry
    iter::Int
    n_components::Int
    eff::Float64
    ess::Float64
    samples_file::Union{String,Nothing}  # Path to saved samples file (on disk, not in memory)
    cum_cost::Float64                    # cumulative likelihood cost at end of iter
end

# Constructor without cum_cost (for backward compatibility)
WhackLogEntry(iter::Int, n_components::Int, eff::Float64, ess::Float64,
              samples_file::Union{String,Nothing}) =
    WhackLogEntry(iter, n_components, eff, ess, samples_file, NaN)

# Constructor without samples_file (for backward compatibility)
WhackLogEntry(iter::Int, n_components::Int, eff::Float64, ess::Float64) =
    WhackLogEntry(iter, n_components, eff, ess, nothing, NaN)



"""
    _read_counter_cost(counter)

Helper that reads a cumulative likelihood-cost figure from any of the
counter variants we accept here:

- `nothing`               → returns NaN (caller flags the iteration as
                            "cum_cost unavailable").
- `Ref{<:Real}`           → returns the wrapped value (legacy contract).
- anything with `cost(c)` → returns `cost(c)` (the LikelihoodCounter
                            from the experiments harness).
"""
function _read_counter_cost(c)
    c === nothing && return NaN
    if applicable(getindex, c) && !(c isa AbstractArray)
        try
            return Float64(c[])
        catch
            # fall through
        end
    end
    if applicable(cost, c)
        try
            return Float64(cost(c))
        catch
            # fall through
        end
    end
    return NaN
end

function whack_many_moles(posterior, init_samples; target_efficiency=Inf, target_ess=Inf, maxiter=100, max_evals::Int=typemax(Int), counter=nothing, n_parallel=Threads.nthreads(), cache_dir=nothing, save_iteration_samples=false)
    @info "whack_many_moles() called with target_efficiency=$target_efficiency, target_ess=$target_ess, maxiter=$maxiter, save_iteration_samples=$save_iteration_samples"
    pstr, f_trafo = bat_transform(PriorToNormal(), posterior)
    smpls_p = init_samples.samples_p
    
    #μ = mean(smpls_p.v, ProbabilityWeights(smpls_p.weight))
    #Σ = cov(Matrix(flatview(smpls_p.v)'), ProbabilityWeights(smpls_p.weight))
    #approx_dist = MvNormal(μ, Σ)
    approx_dist = init_samples.approx_dist

    if approx_dist isa MixtureModel
        approx_mix = init_samples.approx_dist
        mode_logd_p_mix = [logdensityof(pstr, mode(d)) for d in approx_mix.components]
        #mode_logd_q_mix = [logdensityof(approx_dist, mode(approx_dist)) for approx_dist in approx_mix.components]
    else
        approx_mix = Distributions.MixtureModel([approx_dist], [1])
        mode_logd_p_mix = [logdensityof(pstr, mode(approx_dist))]
        #mode_logd_q_mix = [logdensityof(approx_dist, mode(approx_dist))]
    end
    
    samples_mix = smpls_p
    iter = 0

    if !isnothing(cache_dir)
        if !isdir(cache_dir)
            mkdir(cache_dir)
        end
    end
    whack_log = WhackLogEntry[]
    while true

        ess = bat_eff_sample_size(samples_mix, KishESS()).result
        eff = ess / length(samples_mix)
       
        @info "Iteration $iter: Efficiency=$eff, Effective sample size=$ess"

        #log stats (with optional sample snapshot saved to disk)
        n_components = length(approx_mix.components)
        samples_file = nothing
        if cache_dir !== nothing
            samples_file = joinpath(cache_dir, "iter_$(lpad(iter, 3, '0')).jld2")
            # For iter 0 ONLY: save initial samples (subsequent iterations are saved at end of loop)
            if iter == 0
                JLD2.jldopen(samples_file, "w"; iotype=IOStream) do f
                    f["approx_dist"] = approx_mix
                    f["iter"] = iter
                    if save_iteration_samples
                        # Save initial samples for potential resumption
                        f["samples_mix"] = samples_mix
                        f["samples_user"] = bat_transform(inverse(f_trafo), samples_mix).result
                        @info "Saved iteration $iter with initial samples to $samples_file"
                    else
                        @info "Saved iteration $iter approx_dist to $samples_file"
                    end
                end
            end
            # Note: iter > 0 files are saved at the END of each loop iteration (below)
        end
        cum_cost_now = _read_counter_cost(counter)
        push!(whack_log, WhackLogEntry(iter, n_components, eff, ess, samples_file, cum_cost_now))

        evals_so_far = isnan(cum_cost_now) ? 0 : cum_cost_now
        budget_exhausted = (max_evals < typemax(Int)) && (evals_so_far >= max_evals)
        if (eff >= target_efficiency) || (iter >= maxiter) || (ess >= target_ess) || budget_exhausted
            @info "Stopping: eff=$eff target=$target_efficiency, iter=$iter max=$maxiter, ess=$ess target=$target_ess, evals=$evals_so_far max=$max_evals"
            break
        end
        
        idxs = partialsortperm(samples_mix.weight, 1:n_parallel, rev=true)
        
        approx_dists = Array{MvNormal}(undef, n_parallel)
        mode_logd_p_approx = Array{Any}(undef, n_parallel)

        Threads.@threads for i in 1:n_parallel
            θ_iter = samples_mix.v[idxs[i]]
            approx_dists[i] = local_MGVI_approx(pstr, θ_iter)
            mode_logd_p_approx[i] = logdensityof(pstr, mode(approx_dists[i]))
        end
                
        append!(mode_logd_p_mix, mode_logd_p_approx)
        approx_mix = Distributions.MixtureModel(vcat(approx_mix.components, approx_dists))

        mode_logd_q_mix = [logdensityof(approx_mix, mode(ad)) for ad in approx_mix.components]
        
        raw_mixture_logw = mode_logd_p_mix .- mode_logd_q_mix
        raw_mixture_w = exp.(raw_mixture_logw .- maximum(raw_mixture_logw))
        mixture_w = raw_mixture_w ./ sum(raw_mixture_w)

        approx_mix = MixtureModel(approx_mix.components, mixture_w)


        #to parallelize: draw new samples from each new component in parallel
        new_nsamples = [floor(Int, w * length(samples_mix)) for w in last(mixture_w, n_parallel)]

        sampls_ps = Array{Any}(undef, n_parallel)
        
        Threads.@threads for i in 1:n_parallel
            n = new_nsamples[i]
            if n > 0
                sampls_ps[i] = importance_sampling(pstr, approx_dists[i], n)
            else
                sampls_ps[i] = nothing
            end
        end

        # Collect newly generated samples for incremental saving
        new_samples_list = [sp for sp in sampls_ps if sp !== nothing]
        
        # concatenate all newly generated sample batches into the main cloud
        for sp in new_samples_list
            samples_mix = vcat(samples_mix, sp)
        end

        logd_p = samples_mix.logd
        logd_q = logdensityof.(Ref(approx_mix), samples_mix.v)
        logw_raw = logd_p .- logd_q;
        w = exp.(logw_raw .- maximum(logw_raw));
        samples_mix.weight .= w;

        iter += 1
        @info "Iteration $iter completed: efficiency=$eff, iter=$iter, ess=$ess"
        
        # Save iteration data: always save approx_dist, optionally save NEW samples (incremental)
        if !isnothing(cache_dir)
            iter_file = joinpath(cache_dir, "iter_$(lpad(iter, 3, '0')).jld2")
            JLD2.jldopen(iter_file, "w"; iotype=IOStream) do f
                f["approx_dist"] = approx_mix
                f["iter"] = iter
                # Save ONLY the new samples generated this iteration (incremental, not cumulative)
                # This keeps file sizes bounded while allowing full reconstruction
                if save_iteration_samples && !isempty(new_samples_list)
                    # Concatenate new samples for this iteration only
                    new_samples_mix = reduce(vcat, new_samples_list)
                    new_samples_user = bat_transform(inverse(f_trafo), new_samples_mix).result
                    f["new_samples_mix"] = new_samples_mix
                    f["new_samples_user"] = new_samples_user
                    @info "Saved iteration $iter with $(length(new_samples_mix)) new samples"
                else
                    @info "Saved iteration $iter approx_dist"
                end
            end
        end

    end

    (approx_dist=approx_mix, samples_p=samples_mix, samples_user=bat_transform(inverse(f_trafo), samples_mix).result,
    whack_log=whack_log,)
    
end

end # module
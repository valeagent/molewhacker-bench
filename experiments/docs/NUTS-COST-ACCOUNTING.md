<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
See LICENSE at the repository root for the full MIT license text.
-->

# NUTS gradient-cost accounting

**Status:** authoritative methodology note.
**Scope:** how the benchmark charges the No-U-Turn Sampler (NUTS, via
`BAT.HamiltonianMC` → AdvancedHMC.jl) on the common cost axis, why the naive
likelihood-call counter under-counts it, what was tried, the method actually
used, and how it is validated. This file is written to be read directly into
the thesis (benchmark-design cost section + the implementation appendix).

Code: [`experiments/src/algorithms/algo_nuts.jl`](../src/algorithms/algo_nuts.jl),
[`experiments/src/counter.jl`](../src/counter.jl).

---

## 1. The fairness cost model

Every algorithm in this benchmark is compared on a single budget axis: the
number of **log-density-equivalent evaluations** it is allowed to spend. The
contract (`LikelihoodCounter`, `counter.jl`) is:

| operation                                   | charge |
| ------------------------------------------- | ------ |
| one **primal** evaluation `log f(θ)`        | `1`    |
| one **gradient** `∇ log f(θ)` in `d` dims   | `d`    |

so

```
cost = n_primal + n_grad_partials,   with n_grad_partials = d · (#gradients).
```

The rationale is that a `d`-dimensional forward-mode gradient costs the same as
`d` primal evaluations (one per partial / dual chunk), which is the standard
"a gradient is `d` likelihoods" convention. This makes gradient-based samplers
(NUTS) and gradient-free samplers (IS, RWMH, nested sampling, MoleWhacker) pay
for the *information* they extract from the target on a common scale. It is the
**central fairness axis of the entire study**, so the gradient count for NUTS
must be exact, not estimated.

A primal call increments `n_primal` by 1. A `ForwardDiff.Dual` call increments
`n_dual_calls` by 1 and `n_grad_partials` by the dual chunk size; summed over
chunks, one complete `d`-dimensional gradient contributes exactly `d`. This is
what `LikelihoodCounter` measures for **every other algorithm**, and it is
correct for them because they evaluate the target *through the counter*.

---

## 2. Why the likelihood-level counter is blind to NUTS gradients

NUTS does **not** evaluate the target through the `LikelihoodCounter`. BAT builds
the leapfrog value-and-gradient function once,

```julia
fg = valgrad_func(f, adsel)                       # AutoDiffOperators
hamiltonian = AdvancedHMC.Hamiltonian(metric, f, fg)   # BATAdvancedHMCExt
```

and hands `fg` to AdvancedHMC. From that point AdvancedHMC differentiates
through an **internally cached closure** that is built around the AD backend,
not around our counter object. Empirically:

* the counter's dual-dispatch tally **saturates at the warmup/tuning gradient
  count** (~`10^3`), *independent of how many production samples are drawn*;
* even an explicit wrapper placed around the BAT-supplied `fg` saturates at the
  same number — the production leapfrog gradients never call back through it.

The introspection that exposed the bug:

```
sum over transitions of tree_depth  ≈  24 000 gradient evaluations
n_dual (counted by LikelihoodCounter)  =      967
```

i.e. the likelihood-level counter saw `967` gradients while the sampler
actually performed on the order of `24 000`. Charging only the counted `967`
would give NUTS a **1.5–6× inflated efficiency** `η = N_eff / N_like` — it would
appear to extract effective samples almost for free. A per-call counter on the
likelihood is therefore **structurally blind** to NUTS's production gradients;
this is not a tuning problem, it is where BAT constructs the gradient.

> This is a "code guesses hard data" pattern on the central cost axis, so it
> had to be fixed with an exact count, not patched with an estimate.

---

## 3. The fix, in priority order

### (a) PREFERRED — an explicit gradient counter (attempted, not reachable)

The first choice was to route AdvancedHMC's leapfrog gradient through a counting
wrapper so every actual gradient eval (warmup **and** production) bumps the
counter by exactly `d`. We tried wrapping the BAT-supplied `fg`/`valgrad_func`.
**It does not capture production gradients:** BAT owns the gradient construction
and AdvancedHMC caches its own closure (the same reason the `LikelihoodCounter`
is bypassed). The wrapper's count saturates at the warmup number, identically to
the per-call counter. Injecting a counter *inside* AdvancedHMC's Hamiltonian
would mean monkey-patching the package internals on the hot path. This route was
time-boxed and abandoned.

### (b) EXACT FALLBACK — the sampler's own recorded leapfrog count (USED)

AdvancedHMC records the **actual number of leapfrog steps per transition** as
`n_steps` (`tree.nα` in `AdvancedHMC/src/trajectory.jl`). This is the real
trajectory length: it already accounts for **early U-turn termination and
divergence**, so it is *exact*, not an upper bound.

BAT's public sample metadata (`AHMCSampleID`) keeps only `tree_depth`, dropping
`n_steps`. But `n_steps` is still present on the transition statistics
`AdvancedHMC.stat(transition)` at the point where BAT builds the sample id. We
intercept that one function:

```julia
function BAT._get_sample_id(proposal::BAT.HMCProposalState, chainid, walkerid,
                            cycle, stepno, sample_type)
    tstat = AdvancedHMC.stat(proposal.transition)
    if (sample_type == BAT.ACCEPTED_SAMPLE || sample_type == BAT.REJECTED_SAMPLE) &&
       hasproperty(tstat, :n_steps)
        ns = Int(tstat.n_steps)
        Threads.atomic_add!(_NUTS_NSTEPS, ns)          # Σ n_steps (charged)
        Threads.atomic_max!(_NUTS_NSTEPS_MAX, ns)      # worst single transition
    end
    # ... rebuild the AHMCSampleID exactly as the BAT extension does ...
end
```

`BAT._get_sample_id` is called **exactly once per MCMC step**
(`ACCEPTED_SAMPLE`/`REJECTED_SAMPLE` in `mcmc_state.jl`), for **both warmup and
production** transitions. We skip the `CURRENT`/`PROPOSED` setup records to avoid
double counting. The override body is byte-for-byte the BAT AdvancedHMC
extension plus the two harvesting side-effects, and it is installed after
`import AdvancedHMC` has loaded the extension, so it wins on dispatch. Only this
experiment uses NUTS, so the global override is safe (MH/NS never hit this
method).

After sampling we install the exact counts authoritatively:

```julia
set_gradient_accounting!(counter, Σn_steps, d · Σn_steps)
# ⇒ cost(counter) = n_primal + d · Σn_steps
```

This **charges every leapfrog gradient — warmup, tuning, and production — at
cost `d`**, which is precisely the fairness model of §1. NUTS no longer gets
free warmup, and its efficiency is computed against the true gradient work.

### (c) LAST RESORT — `2^tree_depth − 1` (kept only as a diagnostic)

The previous implementation reconstructed gradients as
`Σ weight · (2^tree_depth − 1)`. That is a **conservative upper bound**: a
trajectory that U-turns partway through its final doubling performs *fewer*
leapfrogs than `2^tree_depth − 1`. It over-charges NUTS by an unknown factor.

We retain this quantity **only as a labelled diagnostic** (`_leapfrog_upper_bound`,
reported as `:leapfrog_upper_bound` next to the exact `:leapfrog_exact`) so the
over-counting margin of the old estimate is visible in the logs. **It is never
charged.** The fair cost always uses the exact `n_steps` count from (b).

> Integrity note: even the worst-case route (c) errs *against* NUTS (it
> over-charges), so no choice among (a)/(b)/(c) can revive the original
> efficiency-inflation bug. (b) simply gives the cleanest, most defensible
> numbers.

---

## 4. Budget enforcement: an exact-budget incremental engine

Charging warmup gradients makes small budgets tight, and a naive
"draw `N`, then check" loop cannot hit a budget exactly because `bat_sample`
bundles warmup and production. We therefore drive BAT's MCMC internals directly:

1. **Warm up once per chain** — replicate `bat_sample`'s setup
   (`transform_and_unshape → mcmc_init! → mcmc_burnin! → next_cycle!`,
   helper `_nuts_warmup`). Warmup is paid exactly once.
2. **Draw production in measured chunks** — call `mcmc_iterate!!` ourselves
   (`_nuts_produce!`), charging the exact `Σ n_steps` after each chunk, and
   stop as soon as the realised cost reaches the aim. Production after warmup
   is pure (no warmup mixed in), so its per-step cost is measured exactly.

Because the cost is measured after every chunk and we stop on first crossing,
the realised `N_like / B` lands inside the sanity band without padding.

### The pilot is charged

Before the measured run, the engine warms **one** chain and draws a short
(8-transition) production burst to size the chain count and seed the per-step
cost estimate. In an earlier revision this pilot was *uncharged* on the
argument that it
"mirrors a practitioner's pilot tuning". That argument fails the suite's own
fairness contract: MH pays for its pilot and tuning (3 % + 5 % of `B`) and
MoleWhacker pays for its entire initialisation, so an uncharged NUTS pilot —
several thousand cost units at `d = 5`, comparable to the whole `B = 5·10³`
budget — was a hidden subsidy. In the final engine the pilot's primal and
gradient work
stays on the counter: the production controller budgets against the realised
total (pilot + warmup + production), and the feasibility rule gains a second
clause — a cell is also budget-infeasible when `pilot + warmup + one
transition > 1.2·B`. Consequence: some small-budget cells that were
"feasible" under the earlier accounting only by virtue of the free pilot are
now honestly N/A.

### Never overshoot (the hard part)

NUTS per-transition cost is **heavy-tailed**: in thin-ridge geometry (`mridges`)
the chains migrate into a deeper-tree regime *during* production, so a chunk
sized from the mean per-step cost can blow the budget (we measured up to
`1.86·B` before the guard). Three defences keep **every** cell `≤ 1.2·B`:

1. **Damped step** — a chunk closes only *half* the remaining gap to the aim, so
   a single mis-estimate cannot overshoot far.
2. **Conservative sizing** — the per-step cost used for sizing is
   `max(mean, ½·running-max)`, so a target that has already shown deep trees is
   sized cautiously.
3. **Hard backstop** — a chunk may advance no more transitions than fit under
   `1.2·B` *even if every one of them were `2.5×` as deep as the deepest single
   transition seen so far* (`_NUTS_NSTEPS_MAX`). The theoretical worst case
   (`2^max_depth = 1024` leapfrogs per transition) is unusably pessimistic for
   chunk sizing on easy targets, so we use this measured worst-case bound
   instead.

The aim is `0.90·B` (band centre, with cushion to the `1.2·B` gate).

### Budget-infeasible cells → N/A (never padded)

If **one honestly-tuned trajectory** — BAT's minimum warmup plus a single
transition — already costs `> 1.2·B`, the cell is **budget-infeasible for
NUTS**. It is recorded as N/A:

```
terminated_by = budget_infeasible,  no quality metrics (all NaN)
```

We do **not** pad, overshoot, or re-tune to force it into band. This is honest
and is itself a reportable finding: *"a gradient sampler cannot complete a
single trajectory at this budget."* `compute_metrics_for_cell` (`metrics.jl`)
writes NaN for every distributional metric of such a cell, keeping only the
cost/eval columns that document *why* it is infeasible. (With `d = 20` dropped
from the thesis, this band largely vanishes in practice; the rule stands
generally and applies to any gradient sampler.)

---

## 5. Validation gate and results

Every NUTS cell must satisfy, post-hoc over `cells.csv`:

```
N_eff ≤ n_dual_calls            (= Σ n_steps = #gradients;  no free effective samples)
0.8 ≤ N_like / B ≤ 1.2          (the budget band)         OR  terminated_by = budget_infeasible
```

`N_eff ≤ #gradients` is the invariant that would have caught the original bug:
with the leak, `N_eff` (thousands) exceeded the counted gradients, which is
physically impossible — each effective sample needs at least one gradient.
An in-runner guard (`run_nuts`) asserts this per cell; the post-hoc gate
over `cells.csv` (`experiments/tools/sanity_check.jl`) is authoritative
for shipping.

The runner also exposes, per cell, in `extras` / `cells.csv`:

* `leapfrog_exact` = `Σ n_steps` (the charged, exact leapfrog count),
* `n_dual_calls`   = `Σ n_steps`, `n_grad_partials` = `d · Σ n_steps`
  (so `n_grad_partials / d` reproduces the leapfrog count),
* `leapfrog_upper_bound` = the old `Σ (2^tree_depth − 1)` diagnostic,
* `N_eff`, `N_like / B`.

**Validation outcome (current engine).** Overshoot stress probes over the
heavy-tailed region
(`mridges` at all 20 seeds, plus a `d`/`B`/problem sampling) and the broad
post-hoc gate report every feasible cell inside
`[0.8, 1.2]·B` with `N_eff ≤ n_dual_calls`, and infeasible cells correctly
marked N/A; see the small-budget `B = 5e3` rows of `cells.csv` at
`d = 2/5/10`.

---

## 6. Summary

* The likelihood-level `LikelihoodCounter` cannot see NUTS's production
  gradients (AdvancedHMC caches its own gradient closure around the AD backend);
  an explicit wrapper on BAT's `fg` is bypassed for the same reason.
* We charge the **exact** leapfrog count `Σ n_steps` recorded by AdvancedHMC
  (U-turn aware, not the `2^tree_depth − 1` upper bound), for **warmup +
  production**, at cost `d` per gradient — harvested via a one-method override
  of `BAT._get_sample_id`.
* Budget is enforced **exactly** by an incremental engine that warms up once and
  draws production in measured, overshoot-bounded chunks; infeasible cells are
  honest N/A.
* The result is validated by `N_eff ≤ #gradients` and the `0.8–1.2·B` band on
  every cell.

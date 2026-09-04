# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# plotting.jl — Catalogues A, B, C, D (Protocol §10, rev April 2026)
# =============================================================================
#
# Implements every figure of the protocol's catalogue with strict
# typography, palette, dimensions, and filename conventions. Every
# figure produced lives in `out/figs/<filename>.{pdf,png}`.
#
# A May 2026 visual-review pass flagged a number
# of P2 issues that this file now addresses:
#
#   P2-1   axis / panel titles use `LaTeXStrings.@L_str` for math
#   P2-2/3 heatmap colorbar labels and titles via metric-keyed dicts
#   P2-4/5/6 convergence plot uses configured B as x-axis; reference
#          line anchored to data; explicit log-y tick formatting
#   P2-7   triangle plots — equal panel sizes, tight per-problem axes,
#          self-contained metric annotation
#   P2-8   Pareto plot — separate algorithm-colour and problem-shape
#          legends; outliers clipped with chevron annotation
#   P2-9/10 density figures get a colorbar; truth contours overlaid
#   P2-11  divergence map skipped when no divergent samples are stored
#   P0-6   dlogZ heatmap caps the score axis at -log10(1e-10)
#   P1-3   MoleWhacker saturation annotation in convergence plots
#   P1-5   MW iteration log uses the recorded `cum_cost` field
#   P1-8   truth scatter rendered first, 60 % alpha; algorithm 30 %
#   P1-9   tight axis limits per problem from the truth's empirical
#          99.5 % bounding box

using CairoMakie
import CairoMakie: Makie
using LaTeXStrings
using Printf
using Statistics
using StatsBase: mean, std, ProbabilityWeights
using KernelDensity: kde

# -----------------------------------------------------------------------------
# §10.6 — palettes (Wong 2011 colourblind-safe; algorithm + problem)
# -----------------------------------------------------------------------------

const ALG_COLOR = Dict{Symbol,Any}(
    # V8-FIX-C3: IS lightened (#7F7F7F → #999999) so the IS overlay in
    # tri__*__is corner plots is distinguishable from the darkened truth
    # scatter (TRUTH_COLOR below); IS additionally always renders dotted.
    :is   => "#999999",
    :mh   => "#E69F00",
    :nuts => "#56B4E9",
    :ns   => "#009E73",
    :mw   => "#D55E00",
)

const ALG_LINESTYLE = Dict{Symbol,Any}(
    :is   => :dot,
    :mh   => :dash,
    :nuts => nothing,
    :ns   => :dashdot,
    :mw   => nothing,
)

const ALG_MARKER = Dict{Symbol,Any}(
    :is   => :circle,
    :mh   => :rect,
    :nuts => :diamond,
    :ns   => :utriangle,
    :mw   => :star5,
)

const ALG_LINEWIDTH = Dict{Symbol,Float64}(
    # V5 §1: primary curves at 1.6pt, reference (chance baseline) at 1.0pt.
    # IS is the cheapest "Monte-Carlo reference" sampler, so we keep it at
    # 1.0pt to match the reference-curve thickness; MH/NUTS/NS/MW are
    # primary algorithm curves at 1.6pt.
    :is   => 1.0,
    :mh   => 1.6,
    :nuts => 1.6,
    :ns   => 1.6,
    :mw   => 1.6,
)

const ALG_LABEL = Dict{Symbol,String}(
    :is   => "IS",
    :mh   => "MH",
    :nuts => "NUTS",
    :ns   => "NS",
    :mw   => "MoleWhacker",
)

const PROBLEM_COLOR = Dict{Symbol,Any}(
    :mvn                            => "#332288",
    :banana                         => "#117733",
    :funnel                         => "#88CCEE",
    :mridges                        => "#CC6677",
    :shell                          => "#DDCC77",
    # V3 additions (Protocol §4.7).
    :mridges_spiky                  => "#882255",
    :eggbox                         => "#44AA99",
)

const PROBLEM_LABEL = Dict{Symbol,String}(
    :mvn                            => "MVN",
    :banana                         => "Banana",
    :funnel                         => "Funnel",
    :mridges                        => "M-Ridges",
    :shell                          => "Shell",
    :mridges_spiky                  => "Spiky M-Ridges",
    :eggbox                         => "Eggbox",
)

const PROBLEM_MARKER = Dict{Symbol,Symbol}(
    :mvn                            => :circle,
    :banana                         => :rect,
    :funnel                         => :utriangle,
    :mridges                        => :star5,
    :shell                          => :diamond,
    :mridges_spiky                  => :hexagon,
    :eggbox                         => :pentagon,
)

const TRUTH_COLOR = "#555555"   # V8-FIX-C3: darker than IS grey #999999
const ALG_ORDER = collect(ALG_NAMES)
const PROBLEM_ORDER = collect(PROBLEM_NAMES)


# -----------------------------------------------------------------------------
# Metric-key → human/LaTeX label dictionaries (Protocol §10.11 / rev §P2-2)
# -----------------------------------------------------------------------------

const METRIC_TITLE = Dict{Symbol,LaTeXString}(
    :W1_marginal_avg => L"\overline{W}_1\;(\text{marginal Wasserstein-1})",
    :SWD             => L"\mathrm{SWD}_1\;(\text{sliced Wasserstein})",
    :eta_Nlike       => L"\eta = N_{\mathrm{eff}}/N_{\mathcal L}\;(\text{cost efficiency})",
    :dlogZ           => L"|\Delta \log \mathcal{Z}|\;(\text{log-evidence error})",
    :KL_cube         => L"D_{\mathrm{KL}}(q_T\,\|\,p)",
    :mode_recovery   => L"R(\hat\pi)\;(\text{mode recovery rate})",
    :mmd_rbf         => L"\mathrm{MMD}_{\mathrm{RBF}}\;(\text{kernel discrepancy})",
)

const SCORE_LABEL = Dict{Symbol,LaTeXString}(
    :W1_marginal_avg => L"-\log_{10}\overline{W}_1\;\;(\text{higher = closer to truth})",
    :SWD             => L"-\log_{10}\mathrm{SWD}_1\;\;(\text{higher = closer to truth})",
    :eta_Nlike       => L"\log_{10}\eta\;\;(\text{higher = more efficient})",
    :dlogZ           => L"-\log_{10}|\Delta\log\mathcal{Z}|\;\;(\text{higher = better})",
    :KL_cube         => L"-\log_{10}D_{\mathrm{KL}}\;\;(\text{higher = better})",
    :mode_recovery   => L"R(\hat\pi)\;\;(1.0 = \text{all modes recovered})",
    :mmd_rbf         => L"-\log_{10}\mathrm{MMD}_{\mathrm{RBF}}\;\;(\text{higher = better})",
)


# -----------------------------------------------------------------------------
# §10.5 — typography & §10.4 — figure_size (V8-FIX-C0)
#
# EVERY figure is now produced at its FINAL physical size, with fonts in
# real points, and exported with `pt_per_unit = 1` — 1 Makie figure unit
# = 1 PostScript pt in the PDF. The thesis (`tumphthesis`, text block
# 396 pt × 581 pt) then embeds the PDF at its natural width, never
# scaling. This replaces the V5 scheme, which compounded three shrink
# factors (a ×1.8 canvas inflation whose fonts did NOT follow, the
# CairoMakie default `pt_per_unit = 0.75`, and a LaTeX scale-down from
# a 205 mm-wide PDF to the 139.7 mm text block) into effective text
# sizes of 3.5–6.4 pt across all figure families.
#
#   * Full-width figure: exactly 396 pt (= \textwidth) wide.
#   * Half-width figure (`subfigure{\halfwidth}`): 190 pt (0.48\textwidth).
#   * Fonts (real pt at print): axis labels 10, ticks 9, legend 8,
#     title 10; half-width: axis 9, ticks 8. Annotation floor: 8 pt.
# -----------------------------------------------------------------------------

const _FULL_W_PT = 396.0    # \textwidth of the tumphthesis text block
const _HALF_W_PT = 190.0    # 0.48 \textwidth (subfigure half-width)

"""
    figure_size(class::Symbol, family::Symbol) -> (W_pt, H_pt)

Canonical (width, height) pair in **points** at final print size
(V8-FIX-C0). Class names kept stable:

| `class`      | width               | use                      |
|--------------|---------------------|--------------------------|
| `:narrow`    | 190 pt              | half-width subfigure     |
| `:wide`      | 396 pt              | full-width single figure |
| `:full`      | 396 pt              | full-width single figure |
| `:standard`  | 396 pt              | full-width single figure |
"""
function figure_size(class::Symbol, family::Symbol)
    W = class === :narrow ? _HALF_W_PT : _FULL_W_PT
    if family in (:conv, :eff, :logz, :qe, :scaling, :recovery, :hardness)
        # Landscape 3:2 rectangle.
        H = class === :narrow ? W : (W * 2.0 / 3.0)
    elseif family === :viz_marginal
        H = W / 1.618
    elseif family in (:viz_density, :viz_surface, :tri, :pareto, :pareto_grand)
        H = W
    elseif family === :iter
        H = 1.4 * W
    elseif family === :heatmap
        # 5 columns × 7 rows + colorbar → square canvas.
        H = W
    elseif family === :dim
        # 2 × 4 panel grid (V8-FIX-C3).
        H = 0.78 * W
    else
        H = 0.75 * W
    end
    return (W, H)
end

function figure_resolution(class::Symbol, family::Symbol)
    # V8-FIX-C0: figure units ARE points (export with `pt_per_unit = 1`);
    # the old ×1.8 inflation shrank all text 1.8× relative to the canvas
    # because Makie does NOT scale fonts with the figure size.
    W, H = figure_size(class, family)
    return (round(Int, W), round(Int, H))
end


# -----------------------------------------------------------------------------
# Pub theme — V5 §1 typography
# -----------------------------------------------------------------------------

"""
    set_pub_theme!(; class = :narrow)

Apply the V8 publication theme (V8-FIX-C0). Figures are produced at
final print size with `pt_per_unit = 1`, so every `fontsize` below is a
REAL point size on the thesis page:

* `:wide` / `:full` (full-width figures, 396 pt): axis labels 10 pt,
  ticks 9 pt, legend 8 pt, title 10 pt.
* `:narrow` (half-width subfigures, 190 pt): axis 9 pt, ticks 8 pt.

The theme is global; each figure function sets the appropriate class as
its first call. Colour palette and per-algorithm line/marker styling
live in the `ALG_*` constants and are unchanged across classes.

Font: the protocol's Helvetica → Arial → Arimo fallback chain is
resolved explicitly (Helvetica is typically absent on Windows and the
implicit fallback used to pick an arbitrary sans).
"""
function _resolve_theme_font()
    for name in ("Helvetica", "Arial", "Arimo")
        try
            Makie.to_font(name) !== nothing && return name
        catch
        end
    end
    return "TeX Gyre Heros Makie"
end

function set_pub_theme!(; class::Symbol = :narrow)
    wide = class !== :narrow
    base_pt  = wide ? 10 : 9
    axis_pt  = wide ? 10 : 9
    tick_pt  = wide ?  9 : 8
    title_pt = 10
    legend_pt = wide ? 8 : 8
    theme = Theme(
        fontsize = base_pt,
        font = _resolve_theme_font(),
        Axis = (
            xlabelsize = axis_pt,
            ylabelsize = axis_pt,
            xticklabelsize = tick_pt,
            yticklabelsize = tick_pt,
            titlesize = title_pt,
            titlefont = :bold,
            spinewidth = 0.7,
            # Tick direction outward, length 3pt at print size.
            xtickalign = 0.0,
            ytickalign = 0.0,
            xticksize  = 3,
            yticksize  = 3,
            xtickwidth = 0.7,
            ytickwidth = 0.7,
            xminorticksvisible = true,
            yminorticksvisible = true,
            xminortickalign = 0.0,
            yminortickalign = 0.0,
            xminorticksize  = 1.8,
            yminorticksize  = 1.8,
            xgridvisible = false,
            ygridvisible = false,
        ),
        Legend = (
            labelsize = legend_pt,
            titlesize = 9,          # V8-FIX-C3: Makie default 16 leaked through
            framevisible = false,
            backgroundcolor = (:white, 0.8),
            padding = (4, 4, 3, 3),
            rowgap = 1,
            colgap = 8,
        ),
        Lines = (linewidth = 1.2,),
        Scatter = (markersize = wide ? 6 : 5,),
    )
    set_theme!(theme)
    return nothing
end

function standard_axis!(ax::Axis)
    # Outward ticks of length 3pt at print size; minor ticks visible.
    ax.xtickalign = 0.0
    ax.ytickalign = 0.0
    ax.xticksize  = 3.0
    ax.yticksize  = 3.0
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.spinewidth = 0.7
    ax.xminorticksvisible = true
    ax.yminorticksvisible = true
    return ax
end


# -----------------------------------------------------------------------------
# §10.5 — tex_label dictionary
# -----------------------------------------------------------------------------

const _TEX_LABELS = Dict{Symbol,LaTeXString}(
    :Nlike      => L"N_{\mathcal L}",
    :Neff       => L"N_{\mathrm{eff}}",
    :W1         => L"\overline{W}_{1}",
    :SWD        => L"\mathrm{SWD}_{1}",
    :dlogZ      => L"|\Delta \log \mathcal{Z}|",
    :KL         => L"D_{\mathrm{KL}}(q_T\,\|\,p)",
    :B          => L"B = N_{\mathcal L}",
    # No hyphen: MathTeXEngine renders text-mode hyphens as minus signs.
    :budget     => L"B\;\;(\text{likelihood evaluation budget})",
    :d          => L"d",
    :t          => L"t",
    :eta        => L"N_{\mathrm{eff}}\,/\,N_{\mathcal L}",
    :wall       => L"t_{\mathrm{wall}}\,(\mathrm{s})",
    :time_ess   => L"t_{\mathrm{wall}}\,/\,N_{\mathrm{eff}}",
    :Nlike_ess  => L"N_{\mathcal L}\,/\,N_{\mathrm{eff}}",
    :sigma_spike => LaTeXString("\$\\sigma_\\mathrm{spike}\$"),
    :recovery   => L"R(\hat\pi)",
    :mmd        => L"\mathrm{MMD}_{\mathrm{RBF}}",
)

function tex_label(key::Symbol)
    haskey(_TEX_LABELS, key) || error("tex_label: unknown key $key")
    return _TEX_LABELS[key]
end

# -----------------------------------------------------------------------------
# V8-FIX-C2 — shared text helpers for titles and annotations
# -----------------------------------------------------------------------------

"""
    _budget_latex(B) -> String

Render a budget as thesis-style scientific notation for use inside a
math-mode LaTeXString: `5e5 → 5\\times10^{5}`. (The old titles printed
programmer notation "B = 5e5" verbatim.)
"""
function _budget_latex(B::Real)
    e = floor(Int, log10(abs(Float64(B))))
    m = Float64(B) / 10.0^e
    ms = isapprox(m, round(m); atol = 1e-9) ? string(round(Int, m)) :
         @sprintf("%.1f", m)
    return string(ms, "\\times10^{", e, "}")
end

"""
    _title_latex(prose::AbstractString; math::AbstractString = "") -> String

Build a figure title whose prose renders as REAL text (upright, real
hyphens). The old pattern `LaTeXString(string("M-Ridges — …"))` fed
prose through MathTeXEngine's math mode, turning every hyphen into a
spaced minus sign ("M − Ridges — mode − recovery"); even `\\text{}`
does not restore text hyphens in MathTeXEngine. Titles are therefore
plain strings; the optional `math` tail is converted to Unicode
(superscript exponents via `_budget_latex` markup → ×10ⁿ, `\\;` →
space) so the whole title stays out of math mode.
"""
function _title_latex(prose::AbstractString; math::AbstractString = "")
    isempty(math) && return String(prose)
    tail = replace(String(math),
        "\\times10^{" => "×10^{",
        "\\times 10^{" => "×10^{",
        "\\;" => " ",
        "\\," => " ")
    # Convert remaining ^{...} exponents to Unicode superscripts.
    sup = Dict('0' => '⁰', '1' => '¹', '2' => '²', '3' => '³', '4' => '⁴',
               '5' => '⁵', '6' => '⁶', '7' => '⁷', '8' => '⁸', '9' => '⁹',
               '-' => '⁻')
    tail = replace(tail, r"\^\{([0-9-]+)\}" => s -> begin
        digits = match(r"\^\{([0-9-]+)\}", s).captures[1]
        join(get(sup, c, c) for c in digits)
    end)
    return string(prose, "  ", tail)
end

"Math body of a LaTeXString (without the surrounding \$ delimiters)."
_strip_math(s::LaTeXString) = strip(String(s), '$')
_strip_math(s::AbstractString) = strip(String(s), '$')

"""
    _cell_text_color(score, lo, hi) -> Symbol

V8-FIX-C3: luminance-aware text colour for viridis heatmap cells. The
old hard-coded white was invisible on the bright-yellow high end —
exactly on the best cells. White below 60 % of the range, black above.
"""
function _cell_text_color(score::Real, lo::Real, hi::Real)
    (isnan(score) || hi <= lo) && return :white
    t = clamp((score - lo) / (hi - lo), 0.0, 1.0)
    return t < 0.6 ? :white : :black
end


# -----------------------------------------------------------------------------
# Fix P1-9: per-problem tight axis limits from the truth's 99.5 % box.
# -----------------------------------------------------------------------------

"""
    tight_limits(truth::TruthSet; pad=0.10, coords=(1, 2))

Return `(xlims, ylims)` (each a 2-tuple) computed from the empirical
0.0025 / 0.9975 quantiles of the truth samples in coordinates
`(coords[1], coords[2])`, padded by `pad` of the range on each side.

Falls back to `(-L, L)` if the truth set is too small to estimate
quantiles reliably.
"""
function tight_limits(truth::TruthSet; pad::Real = 0.10,
                          coords::Tuple{Int,Int} = (1, 2))
    Nref = size(truth.samples, 2)
    L = truth.cfg.L
    if Nref < 200
        return (-L, L), (-L, L)
    end
    function _qbox(j)
        x = view(truth.samples, j, :)
        lo = quantile(x, 0.0025)
        hi = quantile(x, 0.9975)
        if !(isfinite(lo) && isfinite(hi)) || hi <= lo
            return (-Float64(L), Float64(L))
        end
        rng = hi - lo
        # Symmetric pad
        return (lo - pad * rng, hi + pad * rng)
    end
    xlims = _qbox(coords[1])
    ylims = _qbox(coords[2])
    # Never go outside the cube prior — the box [-L, L] is part of the
    # target.
    xlims = (max(xlims[1], -Float64(L)), min(xlims[2], Float64(L)))
    ylims = (max(ylims[1], -Float64(L)), min(ylims[2], Float64(L)))
    return xlims, ylims
end


# -----------------------------------------------------------------------------
# Filename + I/O helpers
# -----------------------------------------------------------------------------

function fig_filename(; family::Union{Symbol,AbstractString},
                         problem::Union{Symbol,AbstractString} = :all,
                         d::Union{Integer,Nothing} = HEADLINE_DIMENSION,
                         B::Union{Real,Symbol,Nothing} = nothing,
                         alg::Union{Symbol,AbstractString,Nothing} = nothing,
                         extra::AbstractString = "")
    parts = String[]
    push!(parts, String(family))
    push!(parts, String(problem))
    if d !== nothing
        push!(parts, "d$(Int(d))")
    end
    if B !== nothing
        if B isa Symbol
            push!(parts, "B$(String(B))")
        elseif B isa Real
            push!(parts, "B$(_budget_token(B))")
        end
    end
    if alg !== nothing
        push!(parts, String(alg))
    end
    if !isempty(extra)
        push!(parts, extra)
    end
    return join(parts, "__")
end

function save_pdf(fig, basename::AbstractString; dir::AbstractString = "out/figs",
                    raster_only::Bool = false)
    isdir(dir) || mkpath(dir)
    isdir(joinpath(dir, "png")) || mkpath(joinpath(dir, "png"))
    pdf_path = joinpath(dir, basename * ".pdf")
    png_path = joinpath(dir, "png", basename * ".png")
    # V8-FIX-C0: 1 figure unit = 1 pt in the PDF (CairoMakie's default
    # pt_per_unit = 0.75 silently shrank all text by 25 %). The PNG
    # mirror at 4 px/pt ≈ 288 DPI at final print size.
    raster_only || save(pdf_path, fig; pt_per_unit = 1)
    save(png_path, fig; px_per_unit = 4)
    @info "Saved figure" pdf = raster_only ? "(skipped)" : pdf_path png = png_path
    return raster_only ? png_path : pdf_path
end

function save_png(fig, basename::AbstractString; dir::AbstractString = "out/figs",
                    dpi::Integer = 600)
    isdir(dir) || mkpath(dir)
    isdir(joinpath(dir, "png")) || mkpath(joinpath(dir, "png"))
    png_path = joinpath(dir, "png", basename * ".png")
    pdf_path = joinpath(dir, basename * ".pdf")
    save(png_path, fig; px_per_unit = dpi / 72)
    save(pdf_path, fig; pt_per_unit = 1)   # V8-FIX-C0: vector PDF at print size
    return pdf_path
end


# -----------------------------------------------------------------------------
# Aggregator helpers
# -----------------------------------------------------------------------------

"""
    _seed_summary(df, group_cols, value_col)

Median + min/max band over seeds. Drops `RHAT-FAIL` rows for chain
samplers and genuine `BUDGET-VIOLATION` / `FAILED-SANITY` rows; per
Protocol §6.1.1 (V2-FIX-1) **advisory** under-budget cells (MW
`T_max`/`Neff_target`, NS `dlogz`) are *kept* and the dominant
`stop_reason` is reported back so the plotting layer can attach a
saturation glyph.

The returned DataFrame has columns:

| Column          | Meaning |
|-----------------|---------|
| `<group_cols>`  | grouping keys passed in by the caller |
| `median`        | median of the kept values, NaN if everything was filtered out |
| `lower`/`upper` | min/max band across kept seeds |
| `n`             | number of seeds contributing to the cell |
| `n_advisory`    | number of seeds that were *kept* and carry a protocol-allowed terminator |
| `advisory_tag`  | `:T_max`, `:Neff_target`, `:dlogz`, or `:none` (modal value) |
| `is_advisory`   | `true` iff `n_advisory == n`; the cell as a whole is saturated |
"""
function _seed_summary(df::DataFrame, group_cols::Vector{Symbol}, value_col::Symbol;
                          honour_flags::Bool = true)
    out = combine(groupby(df, group_cols)) do sub
        v = sub[!, value_col]
        keep = trues(nrow(sub))
        advisory = falses(nrow(sub))
        terminators = fill(:none, nrow(sub))
        if honour_flags && hasproperty(sub, :notes)
            for (k, n) in enumerate(sub.notes)
                ns = String(coalesce(n, ""))
                # Headline-killing flags — drop the row.
                if occursin("RHAT-FAIL", ns) || occursin("FAILED-SANITY", ns)
                    keep[k] = false
                    continue
                end
                # Genuine BUDGET-VIOLATION (no exempt terminator) — drop.
                if occursin("BUDGET-VIOLATION", ns)
                    keep[k] = false
                    continue
                end
            end
        end
        if hasproperty(sub, :terminated_by) && hasproperty(sub, :algorithm)
            for k in 1:nrow(sub)
                tb = String(coalesce(sub.terminated_by[k], ""))
                alg_str = String(coalesce(sub.algorithm[k], ""))
                alg_sym = Symbol(alg_str)
                if isempty(tb)
                    continue
                end
                if (alg_sym == :mw && tb in ("T_max", "Neff_target")) ||
                   (alg_sym == :ns && tb == "dlogz")
                    if keep[k]
                        advisory[k] = true
                        terminators[k] = Symbol(tb)
                    end
                end
            end
        end
        v_kept_idx = findall(keep)
        v_kept = Float64[Float64(v[i]) for i in v_kept_idx if isfinite(v[i])]
        if isempty(v_kept)
            return DataFrame(
                median = NaN, lower = NaN, upper = NaN,
                n = 0, n_advisory = 0,
                advisory_tag = String(:none), is_advisory = false,
            )
        end
        n_kept = length(v_kept_idx)
        n_adv = sum(advisory[i] for i in v_kept_idx)
        # Modal advisory tag among the kept rows (ignoring `:none`).
        tag_counts = Dict{Symbol,Int}()
        for i in v_kept_idx
            t = terminators[i]
            t === :none && continue
            tag_counts[t] = get(tag_counts, t, 0) + 1
        end
        advisory_tag = isempty(tag_counts) ? :none :
            argmax(tag_counts)
        return DataFrame(
            median = median(v_kept),
            lower = minimum(v_kept),
            upper = maximum(v_kept),
            n = n_kept,
            n_advisory = n_adv,
            advisory_tag = String(advisory_tag),
            is_advisory = (n_adv > 0 && n_adv == n_kept),
        )
    end
    return out
end

_filter_problem_d(df, prob, d) = df[df.problem .== String(prob) .&& df.d .== Int(d), :]
_filter_alg(df, alg)            = df[df.algorithm .== String(alg), :]


# =============================================================================
# CATALOGUE A — Benchmark visualisations (§10.8)
# =============================================================================

# -- A1: 2D log-density heatmap with contour overlays + colorbar --------------

"""
    fig_viz_density(cfg; truth=nothing, coords=(1,2), N_grid=400) -> Figure
"""
function fig_viz_density(cfg::ProblemConfig; truth::Union{Nothing,TruthSet} = nothing,
                            coords::Tuple{Int,Int} = (1, 2),
                            N_grid::Integer = 400)
    set_pub_theme!()
    res = figure_resolution(:narrow, :viz_density)
    fig = Figure(size = res)
    ax = Axis(fig[1, 1];
        xlabel = L"\theta_1",
        ylabel = L"\theta_2",
        title = L"\log f(\theta)",
        aspect = DataAspect())
    standard_axis!(ax)

    # Choose limits: tight per problem if a truth set is given.
    xlims, ylims = if truth !== nothing
        tight_limits(truth; coords = coords)
    else
        ((-Float64(cfg.L), Float64(cfg.L)),
         (-Float64(cfg.L), Float64(cfg.L)))
    end

    log_f = build_log_f(cfg)
    g1 = collect(range(xlims[1], xlims[2]; length = N_grid))
    g2 = collect(range(ylims[1], ylims[2]; length = N_grid))
    Z = Array{Float64}(undef, N_grid, N_grid)
    base = zeros(cfg.d)
    @inbounds for j in 1:N_grid, i in 1:N_grid
        x = copy(base)
        x[coords[1]] = g1[i]
        x[coords[2]] = g2[j]
        Z[i, j] = log_f(x)
    end
    Zmax = maximum(Z)
    Z_show = max.(Z, Zmax - 12)
    # V8-FIX-C4: rasterize the 400×400 field (as vector quads it pushed
    # the PDF past the size gate).
    hm = heatmap!(ax, g1, g2, Z_show; colormap = :viridis, rasterize = 2)
    contour!(ax, g1, g2, Z_show;
        levels = [Zmax - 5.99, Zmax - 1.84, Zmax - 0.10],
        color = :black, linewidth = 0.6, rasterize = 2)
    poly!(ax, [(-cfg.L, -cfg.L), (cfg.L, -cfg.L), (cfg.L, cfg.L), (-cfg.L, cfg.L)];
        color = :transparent, strokecolor = :black, strokewidth = 0.5,
        linestyle = :dash)
    xlims!(ax, xlims[1], xlims[2])
    ylims!(ax, ylims[1], ylims[2])

    Colorbar(fig[1, 2], hm; label = L"\log f(\theta) - \max_\theta \log f")
    colsize!(fig.layout, 1, Aspect(1, 1.0))
    return fig
end

# -- A2: truth-sample scatter overlay -----------------------------------------

"""
    fig_viz_samples_truth(truth::TruthSet; coords=(1,2), N_show=2000) -> Figure
"""
function fig_viz_samples_truth(truth::TruthSet; coords::Tuple{Int,Int} = (1, 2),
                                  N_show::Integer = 2_000)
    set_pub_theme!()
    res = figure_resolution(:narrow, :viz_density)
    fig = Figure(size = res)
    ax = Axis(fig[1, 1];
        xlabel = L"\theta_1",
        ylabel = L"\theta_2",
        title = _title_latex(string(PROBLEM_LABEL[truth.problem],
                                    ", truth samples")),
        aspect = DataAspect())
    standard_axis!(ax)
    Nref = size(truth.samples, 2)
    idx = rand(1:Nref, min(N_show, Nref))
    s = truth.samples[:, idx]
    pcolor = PROBLEM_COLOR[truth.problem]
    scatter!(ax, s[coords[1], :], s[coords[2], :];
        markersize = 2.5, color = pcolor, alpha = 0.6, strokewidth = 0,
        rasterize = 2)   # V8-FIX-C4: keep PDFs under the size gate
    xlims, ylims = tight_limits(truth; coords = coords)
    xlims!(ax, xlims[1], xlims[2])
    ylims!(ax, ylims[1], ylims[2])
    return fig
end

# -- A3: 1D marginals --------------------------------------------------------

function fig_viz_marginals(truth::TruthSet; max_panels::Integer = 4)
    set_pub_theme!()
    npanels = min(truth.d, max_panels)
    res = figure_resolution(:narrow, :viz_marginal)
    fig = Figure(size = (res[1], res[2] * npanels))
    pcolor = PROBLEM_COLOR[truth.problem]
    for j in 1:npanels
        ax = Axis(fig[j, 1];
            xlabel = LaTeXString("\$\\theta_$(j)\$"),
            ylabel = "density")
        standard_axis!(ax)
        x = vec(truth.samples[j, :])
        density!(ax, x; color = (pcolor, 0.30), strokecolor = pcolor, strokewidth = 1.0)
    end
    return fig
end

# -- A overview composite ----------------------------------------------------

"""
    fig_viz_overview(cfg, truth)

V5 §3b — half-width corner-plot overview of a target at the headline
dimension `d = 5`. Layout:

```
+-----------------------------+--+
|                             |  |  ← 1-D marginal of θ₂
|   2-D filled contour of     |  |     (right edge)
|   log f on (θ₁, θ₂)         |  |
|   with faint reference      |  |
|   samples (alpha = 0.15,    |  |
|   markersize = 1pt)         |  |
+-----------------------------+--+
|   1-D marginal of θ₁        |
+--------------------------------+
```

Half-width canvas (3.6 in × 3.6 in producer-side, 2.66 in embedded);
viridis colourmap with redundant height encoding so the figure stays
legible in grayscale; LaTeX `θ₁`, `θ₂` axis labels.

Axis limits come from the truth's empirical 99.5 % bounding box
(`tight_limits`), padded by 10 % on each side.
"""
function fig_viz_overview(cfg::ProblemConfig, truth::TruthSet)
    set_pub_theme!(class = :narrow)
    res = figure_resolution(:narrow, :viz_density)
    fig = Figure(size = res)
    xlims, ylims = tight_limits(truth)
    pcolor = PROBLEM_COLOR[truth.problem]

    # 4-cell grid: (row 1 col 1) main 2-D, (row 1 col 2) right marginal,
    # (row 2 col 1) bottom marginal, (row 2 col 2) corner spacer.
    ax_main = Axis(fig[1, 1];
        xlabel = L"\theta_1",
        ylabel = L"\theta_2",
        aspect = DataAspect())
    standard_axis!(ax_main)
    hm = _draw_density_panel!(ax_main, cfg;
                                  N_grid = 400, xlims = xlims, ylims = ylims)

    # Faint reference samples (V5 §3b).
    Nref = size(truth.samples, 2)
    Nref_show = min(Nref, 4_000)
    if Nref_show > 0
        idx = rand(1:Nref, Nref_show)
        scatter!(ax_main, vec(truth.samples[1, idx]),
                 vec(truth.samples[2, idx]);
            markersize = 1.5, color = (:white, 0.28),
            strokewidth = 0, rasterize = 2)   # V8-FIX-C4
    end
    xlims!(ax_main, xlims[1], xlims[2])
    ylims!(ax_main, ylims[1], ylims[2])

    # Right-edge: 1-D marginal of θ₂. Minor ticks must be disabled
    # explicitly — the theme enables them globally and they rendered as
    # stray dashes on the otherwise decoration-free marginal panels.
    ax_right = Axis(fig[1, 2];
        xlabel = "",
        ylabel = "",
        xticksvisible = false,
        xticklabelsvisible = false,
        yticksvisible = false,
        yticklabelsvisible = false,
        xminorticksvisible = false,
        yminorticksvisible = false,
        leftspinevisible = false,
        rightspinevisible = false,
        topspinevisible = false,
        bottomspinevisible = false)
    if Nref >= 100
        density!(ax_right, vec(truth.samples[2, :]);
            color = (pcolor, 0.32),
            strokecolor = pcolor, strokewidth = 1.0,
            direction = :y)
    end
    ylims!(ax_right, ylims[1], ylims[2])

    # Bottom-edge: 1-D marginal of θ₁ (minor ticks off, as above).
    ax_bottom = Axis(fig[2, 1];
        xlabel = "",
        ylabel = "",
        xticksvisible = false,
        xticklabelsvisible = false,
        yticksvisible = false,
        yticklabelsvisible = false,
        xminorticksvisible = false,
        yminorticksvisible = false,
        leftspinevisible = false,
        rightspinevisible = false,
        topspinevisible = false,
        bottomspinevisible = false)
    if Nref >= 100
        density!(ax_bottom, vec(truth.samples[1, :]);
            color = (pcolor, 0.32),
            strokecolor = pcolor, strokewidth = 1.0)
    end
    xlims!(ax_bottom, xlims[1], xlims[2])

    # Layout: keep main panel square, marginals slim. We deliberately
    # omit the colourbar — colour redundantly encodes height in the 3-D
    # surface companion figure and the marginal
    # densities along the right and bottom edges are the more
    # informative panel for the d = 5 overview at half-width.
    rowsize!(fig.layout, 1, Relative(0.80))
    rowsize!(fig.layout, 2, Relative(0.18))
    colsize!(fig.layout, 1, Relative(0.80))
    colsize!(fig.layout, 2, Relative(0.18))
    rowgap!(fig.layout, 6)
    colgap!(fig.layout, 6)
    return fig
end


# =============================================================================
# A.5 — `viz__<p>__d2__surface.pdf`
# =============================================================================
#
# Mukherjee-style 3-D surface render of the target log-density on the
# 2-D version of every benchmark. Half-width canvas, camera elevation
# 30°, azimuth -60°. Colour by the (log-)density value with viridis;
# faint wireframe overlay every 20 grid lines for depth perception.

"""
    fig_viz_surface_3d(cfg; mode=:auto, n=300, eggbox_n=600)

3-D surface plot of the target on `(θ₁, θ₂)` over the truncation cube
`[-L, L]^2` for a `d = 2` instance of the problem. The vertical axis
is `f` (density) for problems whose density has a clean dynamic range
on the cube (eggbox, m-spiky), and `log f` (log-density) otherwise
(MVN, banana, funnel, m-ridges, shell). Pass `mode = :density` or
`mode = :log_density` to override the heuristic.

This is the figure embedded directly into each per-problem subsection
of §7.2 of the thesis. It exists as a half-width producer canvas so
LaTeX scales it to 192pt = 2.66in at embed time.

Style follows V5 §3a:

* Camera elevation 30°, azimuth -60° (Mukherjee 2006 convention).
* `viridis` colourmap (gold for high, dark navy for low) — redundant
  with height so the figure remains legible in grayscale.
* Grid resolution `n × n` per axis (`eggbox_n` for eggbox to render
  the spike tips cleanly).
* Faint wireframe (alpha 0.15, line width 0.4pt) every 20 grid lines.
* Box around the plotting domain matches `[-L, L]^2`.
* Small text annotation `d = 2` in the upper-right corner.
"""
function fig_viz_surface_3d(cfg::ProblemConfig;
                                mode::Symbol = :auto,
                                n::Integer = 300,
                                eggbox_n::Integer = 600,
                                cam_elevation_deg::Real = 30.0,
                                cam_azimuth_deg::Real = -60.0)
    cfg.d == 2 || throw(ArgumentError(
        "fig_viz_surface_3d expects a `d = 2` configuration; got d = $(cfg.d). " *
        "Construct the surface from `make_config_<problem>(d = 2)`."))
    set_pub_theme!(class = :narrow)
    res = figure_resolution(:narrow, :viz_surface)
    fig = Figure(size = res)

    L = Float64(cfg.L)
    prob_sym = _problem_symbol_for_cfg(cfg)
    use_density = if mode === :density
        true
    elseif mode === :log_density
        false
    else
        # Heuristic: eggbox and m-spiky look better in density domain
        # because the log-density swings ~10² nats and washes out the
        # surface in 3-D; everything else uses log-density.
        prob_sym in (:eggbox, :mridges_spiky)
    end
    grid_n = prob_sym === :eggbox ? Int(eggbox_n) : Int(n)

    g1 = collect(range(-L, L; length = grid_n))
    g2 = collect(range(-L, L; length = grid_n))
    log_f = build_log_f(cfg)
    Z = Array{Float64}(undef, grid_n, grid_n)
    base = zeros(2)
    @inbounds for j in 1:grid_n, i in 1:grid_n
        base[1] = g1[i]; base[2] = g2[j]
        Z[i, j] = log_f(base)
    end
    Zmax = maximum(Z)

    Z_show = if use_density
        # Rescale to density and clip to the 99 % range so a rare
        # spike at the origin (e.g. funnel) does not flatten the rest.
        D = exp.(Z .- Zmax)
        cap = quantile(vec(D), 0.999)
        clamp.(D, 0.0, cap)
    else
        max.(Z .- Zmax, -12.0)
    end

    z_label = use_density ?
        LaTeXString("\$f(\\theta_1,\\theta_2)\$") :
        LaTeXString("\$\\log f(\\theta_1,\\theta_2)\$")

    ax = Axis3(fig[1, 1];
        xlabel = L"\theta_1",
        ylabel = L"\theta_2",
        zlabel = z_label,
        elevation = deg2rad(cam_elevation_deg),
        azimuth   = deg2rad(cam_azimuth_deg),
        xtickwidth = 0.7,
        ytickwidth = 0.7,
        ztickwidth = 0.7,
        xlabelsize = 10,
        ylabelsize = 10,
        zlabelsize = 10,
        xticklabelsize = 9,
        yticklabelsize = 9,
        zticklabelsize = 9,
        # Three ticks per axis — five ticks overlapped each other on
        # the 190 pt canvas ("−10 −5 0 5 10" collided along the bottom
        # edges).
        xticks = WilkinsonTicks(3),
        yticks = WilkinsonTicks(3),
        zticks = WilkinsonTicks(3),
        # Per-side protrusions. A uniform 55 pt reserved 110 pt of
        # the 190 pt canvas and shrank the surface to a thumbnail; only
        # the right side needs room for the z tick labels + z label
        # (52 pt — at 42 pt the tail of "log f(θ₁,θ₂)" was clipped).
        protrusions = (22, 52, 30, 8),
    )
    # Rasterize the surface so the PDF stays small. Without this every
    # grid cell becomes a vector quad and the PDF balloons to >100 MB
    # (eggbox at n=600 is 360k quads, ~370 MB un-rasterized). The
    # wireframe overlay below stays vector-based — it has only ~400
    # line segments and renders at print resolution.
    surface!(ax, g1, g2, Z_show;
        colormap = :viridis,
        transparency = false,
        shading = NoShading,
        rasterize = 4,
    )
    # Faint wireframe overlay every 20 grid lines for depth perception.
    step = max(1, div(grid_n, 20))
    sub_idx = 1:step:grid_n
    if length(sub_idx) >= 4
        wireframe!(ax, g1[sub_idx], g2[sub_idx], Z_show[sub_idx, sub_idx];
            color = (:black, 0.15), linewidth = 0.4)
    end
    # `d = 2` annotation in the upper-right corner of the figure.
    Label(fig[1, 1, TopRight()], LaTeXString("\$d = 2\$");
        padding = (0, 8, 8, 0), fontsize = 10, halign = :right,
        valign = :top, tellheight = false, tellwidth = false)
    return fig
end

# Internal: map a `ProblemConfig` to its canonical problem symbol.
# Falls back on `:unknown` if the config type is not registered (used
# only inside `fig_viz_surface_3d`).
function _problem_symbol_for_cfg(cfg::ProblemConfig)
    cfg isa ConfigMVN           && return :mvn
    cfg isa ConfigBanana        && return :banana
    cfg isa ConfigFunnel        && return :funnel
    cfg isa ConfigMRidges       && return :mridges
    cfg isa ConfigShell         && return :shell
    cfg isa ConfigMRidgesSpiky  && return :mridges_spiky
    cfg isa ConfigEggbox        && return :eggbox
    return :unknown
end


# -----------------------------------------------------------------------------
# V5 §2 — data cleaning
# -----------------------------------------------------------------------------

"""
    drop_v5_excluded_rows!(df) -> Int

Remove the two stray `eggbox / d = 5 / seed = 11` cells (1 IS,
1 NS row) from `cells.csv`. They are 0.04 % leftovers of an early
grid draft (eggbox is benchmarked at `d = 2` only) and are dropped
from every downstream computation. Returns the number of rows dropped.
"""
function drop_v5_excluded_rows!(df::DataFrame)
    n_before = nrow(df)
    drop = (df.problem .== "eggbox") .& (df.d .== 5) .& (df.seed .== 11)
    if any(drop)
        keep = .!drop
        deleteat!(df, findall(.!keep))
    end
    return n_before - nrow(df)
end

function _draw_density_panel!(ax, cfg::ProblemConfig; N_grid::Integer = 300,
                                  xlims = nothing, ylims = nothing)
    log_f = build_log_f(cfg)
    L = cfg.L
    if xlims === nothing
        xlims = (-Float64(L), Float64(L))
    end
    if ylims === nothing
        ylims = (-Float64(L), Float64(L))
    end
    g1 = collect(range(xlims[1], xlims[2]; length = N_grid))
    g2 = collect(range(ylims[1], ylims[2]; length = N_grid))
    Z = Array{Float64}(undef, N_grid, N_grid)
    base = zeros(cfg.d)
    @inbounds for j in 1:N_grid, i in 1:N_grid
        x = copy(base)
        x[1] = g1[i]
        x[2] = g2[j]
        Z[i, j] = log_f(x)
    end
    Zmax = maximum(Z)
    Z_show = max.(Z, Zmax - 12)
    # V8-FIX-C4: rasterize the dense 300×300 field and its contours —
    # as vector quads/paths they pushed each viz PDF to 1.8–2.5 MB.
    hm = heatmap!(ax, g1, g2, Z_show; colormap = :viridis, rasterize = 2)
    contour!(ax, g1, g2, Z_show;
        levels = [Zmax - 5.99, Zmax - 1.84, Zmax - 0.10],
        color = :black, linewidth = 0.5, rasterize = 2)
    poly!(ax, [(-L, -L), (L, -L), (L, L), (-L, L)];
        color = :transparent, strokecolor = :black, strokewidth = 0.4,
        linestyle = :dash)
    xlims!(ax, xlims[1], xlims[2])
    ylims!(ax, ylims[1], ylims[2])
    return hm
end

function _draw_truth_scatter_panel!(ax, truth::TruthSet; N_show::Integer = 2_000,
                                          xlims = nothing, ylims = nothing)
    Nref = size(truth.samples, 2)
    idx = rand(1:Nref, min(N_show, Nref))
    s = truth.samples[:, idx]
    pcolor = PROBLEM_COLOR[truth.problem]
    scatter!(ax, s[1, :], s[2, :];
        markersize = 2.5, color = pcolor, alpha = 0.6, strokewidth = 0,
        rasterize = 2)   # V8-FIX-C4
    if xlims === nothing
        xlims = (-Float64(truth.cfg.L), Float64(truth.cfg.L))
    end
    if ylims === nothing
        ylims = (-Float64(truth.cfg.L), Float64(truth.cfg.L))
    end
    xlims!(ax, xlims[1], xlims[2])
    ylims!(ax, ylims[1], ylims[2])
    return ax
end

function _overlay_truth_contour!(ax, truth::TruthSet;
                                       xlims = nothing, ylims = nothing,
                                       N_grid::Integer = 200)
    Nref = size(truth.samples, 2)
    Nref < 500 && return
    s = truth.samples
    if xlims === nothing
        xlims = (Float64(minimum(s[1, :])), Float64(maximum(s[1, :])))
    end
    if ylims === nothing
        ylims = (Float64(minimum(s[2, :])), Float64(maximum(s[2, :])))
    end
    try
        k = kde((vec(s[1, :]), vec(s[2, :])))
        # CairoMakie's contour over the kde density.
        # We need to evaluate the density on a regular grid manually
        # because kde returns a `BivariateKDE` object with `.x`, `.y`, `.density`.
        contour!(ax, k.x, k.y, k.density; color = :white, linewidth = 0.6,
                 levels = 3)
    catch err
        @warn "Truth contour overlay failed — skipping" exception = (err, catch_backtrace())
    end
    return ax
end

function fig_viz_gallery(cfgs::Vector{<:ProblemConfig}, truths::Vector{TruthSet})
    # V8-FIX-C3: 2×4 grid instead of one 7-panel row — at full text width
    # a 1×7 row gave each panel ≈ 22 mm, unreadable. Axis labels only on
    # the outer edge; panel titles carry the problem names.
    set_pub_theme!(class = :full)
    res = figure_resolution(:full, :viz_density)
    n = length(cfgs)
    cols = 4
    rows = ceil(Int, n / cols)
    fig = Figure(size = (res[1], round(Int, res[1] * rows / cols)))
    for (i, cfg) in enumerate(cfgs)
        r = div(i - 1, cols) + 1
        c = mod(i - 1, cols) + 1
        xlims, ylims = tight_limits(truths[i])
        bottom = (r == rows) || (i + cols > n)
        ax = Axis(fig[r, c];
            xlabel = bottom ? L"\theta_1" : "",
            ylabel = c == 1 ? L"\theta_2" : "",
            title = PROBLEM_LABEL[truths[i].problem],
            aspect = DataAspect())
        standard_axis!(ax)
        _draw_density_panel!(ax, cfg; xlims = xlims, ylims = ylims)
    end
    return fig
end


# =============================================================================
# CATALOGUE B — Cross-algorithm comparison plots (§10.9)
# =============================================================================

function _per_alg_curves(df::DataFrame, problem::Symbol, d::Int, value_col::Symbol;
                              honour_flags::Bool = true)
    sub = _filter_problem_d(df, problem, d)
    rows = Dict{Symbol,DataFrame}()
    for alg in ALG_ORDER
        sub_alg = sub[sub.algorithm .== String(alg), :]
        rows[alg] = _seed_summary(sub_alg, [:B], value_col;
                                       honour_flags = honour_flags)
        sort!(rows[alg], :B)
    end
    return rows
end


# B.1 — Convergence  ----------------------------------------------------------
#
# Fix P2-4: x-axis is the *configured* budget B (3 categorical points).
# Fix P2-5 + V2-FIX-4: reference -1/2 line anchored to the *upper-left*
#         and only half a decade long, so it does not collide with the
#         right-hand-side MW saturation annotation.
# Fix P2-6: explicit log-y tick formatter.
# Fix P1-3: MoleWhacker saturation marker at the right end.
# V2-FIX-1: render `is_advisory` cells with hollow markers + saturation
#         glyph annotation so the reader sees that those cells exist
#         and are *protocol-allowed* under-spends, not erased data.

"""
    _draw_alg_curve!(ax, c, alg; saturation_text=true) -> LineElement | nothing

Draw one algorithm's median-curve + min/max band on `ax` for a
DataFrame `c` with columns `B, median, lower, upper, n, n_advisory,
advisory_tag, is_advisory` (the schema produced by `_seed_summary`).
`is_advisory` rows are rendered with **hollow** markers (white fill,
algorithm-coloured stroke) and the `advisory_tag` (`T_max`,
`Neff_target`, `dlogz`) is written next to the right-most kept point.
Returns the line handle for legend assembly, or `nothing` if no kept
data.
"""
function _draw_alg_curve!(ax, c::AbstractDataFrame, alg::Symbol;
                            saturation_text::Bool = true,
                            x_col::Symbol = :B,
                            ymin::Float64 = 1e-12)
    isempty(c) && return nothing
    xs   = Vector{Float64}(c[!, x_col])
    meds = Vector{Float64}(c.median)
    los  = Vector{Float64}(c.lower)
    his  = Vector{Float64}(c.upper)
    is_adv = hasproperty(c, :is_advisory) ? Vector{Bool}(c.is_advisory) :
        falses(length(meds))
    tags = hasproperty(c, :advisory_tag) ? Vector{String}(c.advisory_tag) :
        fill("none", length(meds))
    ok = .!isnan.(meds) .& (meds .> 0)
    any(ok) || return nothing
    ln = lines!(ax, xs[ok], meds[ok];
        color = ALG_COLOR[alg],
        linestyle = ALG_LINESTYLE[alg],
        linewidth = ALG_LINEWIDTH[alg])
    # Plot solid markers for non-advisory points, hollow for advisory.
    solid_idx = ok .& .!is_adv
    adv_idx   = ok .&  is_adv
    if any(solid_idx)
        scatter!(ax, xs[solid_idx], meds[solid_idx];
            color = ALG_COLOR[alg], marker = ALG_MARKER[alg], markersize = 8,
            strokewidth = 0)
    end
    if any(adv_idx)
        scatter!(ax, xs[adv_idx], meds[adv_idx];
            color = (:white, 1.0), marker = ALG_MARKER[alg], markersize = 8,
            strokecolor = ALG_COLOR[alg], strokewidth = 1.4)
    end
    band!(ax, xs[ok], max.(los[ok], ymin), his[ok];
        color = (ALG_COLOR[alg], 0.30))
    if saturation_text && any(adv_idx)
        # Annotate the right-most advisory point with its tag.
        last_idx = findlast(adv_idx)
        last_idx === nothing && return ln
        tag = tags[last_idx]
        label = if tag == "T_max"
            L"\text{T}_{\max}\text{-saturated}"
        elseif tag == "Neff_target"
            L"N_{\mathrm{eff}}\text{-target reached}"
        elseif tag == "dlogz"
            L"\Delta\log\mathcal{Z}\text{-converged}"
        else
            LaTeXString("saturated")
        end
        text!(ax, xs[last_idx], meds[last_idx]; text = label,
            offset = (6, 0), fontsize = 8,
            color = ALG_COLOR[alg], align = (:left, :center))
    end
    return ln
end

"""
    fig_conv(df, problem, d) -> Figure  (B.1 — convergence W1 vs B)
"""
function fig_conv(df::DataFrame, problem::Symbol, d::Integer = HEADLINE_DIMENSION)
    set_pub_theme!(class = :wide)
    res = figure_resolution(:wide, :conv)
    fig = Figure(size = res)
    ax = Axis(fig[1, 1];
        xlabel = tex_label(:B),
        ylabel = tex_label(:W1),
        xscale = log10, yscale = log10,
        title = _title_latex(String(PROBLEM_LABEL[problem]);
            math = string("(d = ", Int(d), ")")))
    standard_axis!(ax)

    curves = _per_alg_curves(df, problem, d, :W1_marginal_avg)
    leg_lines = []
    leg_labels = String[]
    all_x = Float64[]
    all_y = Float64[]
    for alg in ALG_ORDER
        c = curves[alg]
        isempty(c) && continue
        ln = _draw_alg_curve!(ax, c, alg; saturation_text = (alg in (:mw, :ns)))
        ln === nothing && continue
        push!(leg_lines, ln)
        push!(leg_labels, ALG_LABEL[alg])
        meds = Vector{Float64}(c.median)
        ok = .!isnan.(meds) .& (meds .> 0)
        append!(all_x, Vector{Float64}(c.B)[ok])
        append!(all_y, meds[ok])
    end

    # V2-FIX-4: half-decade reference -1/2 line anchored to the
    # upper-left, with explicit "slope -1/2" label above it. Keeps the
    # right-hand-side clear for the MW / NS saturation annotations.
    if !isempty(all_x) && !isempty(all_y)
        x_left  = minimum(all_x)
        y_top   = maximum(all_y) * 0.95
        # span half a decade in x.
        x_right = x_left * sqrt(10.0)
        y_right = y_top / sqrt(10.0)^0.5  # slope -1/2 in log-log
        lines!(ax, [x_left, x_right], [y_top, y_right];
            color = (:gray, 0.7), linestyle = :dot, linewidth = 0.8)
        text!(ax, sqrt(x_left * x_right), y_top * 1.10;
            text = L"\propto N_{\mathcal L}^{-1/2}",
            fontsize = 8, color = :gray40, align = (:center, :bottom))
    end

    _apply_log10_yticks!(ax, all_y)
    _apply_log10_xticks!(ax, all_x)
    axislegend(ax, leg_lines, leg_labels; position = :rt)
    return fig
end

# Anchor explicit log-y / log-x tick labels (fix P2-6).
function _apply_log10_yticks!(ax::Axis, ys::AbstractVector{<:Real})
    isempty(ys) && return
    finite = filter(isfinite, Float64.(ys))
    isempty(finite) && return
    finite = filter(>(0), finite)
    isempty(finite) && return
    lo, hi = log10(minimum(finite)), log10(maximum(finite))
    klo = floor(Int, lo) - 1
    khi = ceil(Int, hi) + 1
    if khi - klo > 8
        klo = floor(Int, lo)
        khi = ceil(Int, hi)
    end
    decades = collect(klo:khi)
    ticks = (10.0 .^ decades, [_pretty_decade_label(k) for k in decades])
    ax.yticks = ticks
    return ax
end

function _apply_log10_xticks!(ax::Axis, xs::AbstractVector{<:Real})
    isempty(xs) && return
    finite = filter(isfinite, Float64.(xs))
    finite = filter(>(0), finite)
    isempty(finite) && return
    lo, hi = log10(minimum(finite)), log10(maximum(finite))
    klo = floor(Int, lo)
    khi = ceil(Int, hi)
    decades = collect(klo:khi)
    ticks = (10.0 .^ decades, [_pretty_decade_label(k) for k in decades])
    ax.xticks = ticks
    return ax
end

"""
    _pretty_decade_label(k) -> LaTeXString

Render a tick at `10^k` as `LaTeXString("\$10^{k}\$")`. Per Protocol
§10.16.9 (V2-FIX-2 + P2-NEW-2) every decade is rendered uniformly as
a math-mode exponent so neighbouring ticks share a single visual
vocabulary; Makie passes the LaTeXString through to MathTeXEngine /
MathJax which produces a real superscript instead of leaking the
curly-brace markup as raw text.
"""
function _pretty_decade_label(k::Integer)
    return LaTeXString(@sprintf("\$10^{%d}\$", k))
end


# B.2 — Efficiency  -----------------------------------------------------------

function fig_eff(df::DataFrame, problem::Symbol, d::Integer = HEADLINE_DIMENSION)
    set_pub_theme!(class = :wide)
    res = figure_resolution(:wide, :eff)
    fig = Figure(size = res)
    ax = Axis(fig[1, 1];
        xlabel = tex_label(:B),
        ylabel = tex_label(:eta),
        xscale = log10, yscale = log10,
        title = _title_latex(String(PROBLEM_LABEL[problem]);
            math = string("(d = ", Int(d), ")")))
    standard_axis!(ax)
    curves = _per_alg_curves(df, problem, d, :eta_Nlike)
    leg_lines = []
    leg_labels = String[]
    all_x = Float64[]; all_y = Float64[]
    for alg in ALG_ORDER
        c = curves[alg]
        isempty(c) && continue
        ln = _draw_alg_curve!(ax, c, alg;
            saturation_text = (alg in (:mw, :ns)),
            ymin = 1e-10)
        ln === nothing && continue
        push!(leg_lines, ln)
        push!(leg_labels, ALG_LABEL[alg])
        meds = Vector{Float64}(c.median)
        ok = .!isnan.(meds) .& (meds .> 0)
        append!(all_x, Vector{Float64}(c.B)[ok])
        append!(all_y, meds[ok])
    end
    _apply_log10_yticks!(ax, all_y)
    _apply_log10_xticks!(ax, all_x)
    axislegend(ax, leg_lines, leg_labels; position = :rt)
    return fig
end


# B.3 — Log-evidence error  ---------------------------------------------------

function fig_logz(df::DataFrame, problem::Symbol, d::Integer = HEADLINE_DIMENSION;
                      cap::Real = 1e10)
    set_pub_theme!(class = :wide)
    res = figure_resolution(:wide, :logz)
    fig = Figure(size = res)
    ax = Axis(fig[1, 1];
        xlabel = tex_label(:B),
        ylabel = tex_label(:dlogZ),
        xscale = log10, yscale = log10,
        title = _title_latex(String(PROBLEM_LABEL[problem]);
            math = string("(d = ", Int(d), ")")))
    standard_axis!(ax)
    curves = _per_alg_curves(df, problem, d, :dlogZ)
    leg_lines = []
    leg_labels = String[]
    all_x = Float64[]; all_y = Float64[]
    for alg in (:is, :ns, :mw)
        c = curves[alg]
        isempty(c) && continue
        # Cap to keep one catastrophic IS cell from flattening the rest
        # of the y-axis (P0-6). Mutates a copy of the curve frame.
        c2 = copy(c)
        for col in (:median, :lower, :upper)
            c2[!, col] = [isnan(v) ? NaN : max(min(v, cap), 1e-12)
                          for v in c[!, col]]
        end
        ln = _draw_alg_curve!(ax, c2, alg;
            saturation_text = (alg in (:mw, :ns)),
            ymin = 1e-12)
        ln === nothing && continue
        push!(leg_lines, ln)
        push!(leg_labels, ALG_LABEL[alg])
        meds = Vector{Float64}(c2.median)
        ok = .!isnan.(meds) .& (meds .> 0)
        append!(all_x, Vector{Float64}(c2.B)[ok])
        append!(all_y, meds[ok])
    end
    _apply_log10_yticks!(ax, all_y)
    _apply_log10_xticks!(ax, all_x)
    axislegend(ax, leg_lines, leg_labels; position = :rt)
    return fig
end


# B.4 — Quantile errors  ------------------------------------------------------

function fig_qe(df::DataFrame, problem::Symbol, d::Integer = HEADLINE_DIMENSION)
    set_pub_theme!(class = :wide)
    res = figure_resolution(:wide, :qe)
    fig = Figure(size = res)
    ax = Axis(fig[1, 1];
        xlabel = "quantile",
        ylabel = L"|q_{\mathrm{emp}} - q_{\mathrm{ref}}|",
        title = _title_latex(String(PROBLEM_LABEL[problem]);
            math = string("(d = ", Int(d), ",\\; B = ", _budget_latex(5e5), ")")),
        xticks = (1:5, ["0.025", "0.16", "0.50", "0.84", "0.975"]))
    standard_axis!(ax)
    sub = _filter_problem_d(df, problem, d)
    sub = sub[sub.B .== 5e5, :]
    levels = (:QE_p025, :QE_p160, :QE_p500, :QE_p840, :QE_p975)
    width = 0.15
    leg_lines = []
    leg_labels = String[]
    for (k, alg) in enumerate(ALG_ORDER)
        sub_alg = sub[sub.algorithm .== String(alg), :]
        isempty(sub_alg) && continue
        x_offset = (k - 3) * width
        bar_x = collect(1:5) .+ x_offset
        meds = Float64[]
        for lev in levels
            v = sub_alg[!, lev]
            v_clean = filter(!isnan, Vector{Float64}(v))
            push!(meds, isempty(v_clean) ? NaN : median(v_clean))
        end
        bars = barplot!(ax, bar_x, meds;
            width = width * 0.9, color = ALG_COLOR[alg], strokewidth = 0)
        push!(leg_lines, bars)
        push!(leg_labels, ALG_LABEL[alg])
    end
    axislegend(ax, leg_lines, leg_labels; position = :rt)
    return fig
end


# B.5 — Triangle plot  --------------------------------------------------------
#
# Fix P1-8: render truth scatter first at 60 % alpha; algorithm at 30 %.
# Fix P1-9 / P2-7: tight axis limits, equal panel sizes.

"""
    fig_tri(mr::MethodResult, truth::TruthSet) -> Figure

Catalogue B triangle (corner) plot for a single algorithm cell.
Conventions (Protocol §10.16.10–§10.16.11):

* The footer `N_eff` annotation is computed via `neff(mr)` so chain
  samplers (`mh`, `nuts`) report their autocorrelation-aware ESS,
  *not* the Kish ESS over a deserialiser-filled `weights` vector. The
  same `neff(mr)` value drives the headline heatmap, so the corner
  plot footer and the heatmap now agree (v2 had MH on funnel
  reading η = 0.93 in the corner plot footer and η = 0.0083 in the
  heatmap; the corner plot was wrong).
* Off-diagonal scatter panels are drawn from `_resample_to_eval_N`
  (the same helper used by W₁ / SWD) at `N_show = 5 000` equal-weight
  samples. For weighted samplers — IS in particular — this prevents
  the panel from showing flat cube-uniform noise; the eye now sees
  exactly what the W₁ estimator sees.
"""
function fig_tri(mr::MethodResult, truth::TruthSet;
                  N_show::Int = 5_000,
                  rng::AbstractRNG = MersenneTwister(20260603 + mr.seed),
                  fontscale::Real = 2.0,
                  coords::Union{Nothing,Vector{Int}} = nothing)
    # The thesis embeds every triangle plot at half
    # column width (≈195 pt), so all text is produced `fontscale`×
    # larger on the 396 pt canvas and prints at the intended size.
    # `coords` restricts the panel grid to a subset of coordinates —
    # at d = 10 the full 10×10 grid is illegible at any font size, so
    # the caller passes a representative subset (default: all coords
    # for d ≤ 6, else θ₁, θ₂, θ₃ and θ_d).
    set_pub_theme!(class = :wide)
    res = figure_resolution(:wide, :tri)
    d = mr.d
    show_coords = coords !== nothing ? coords :
                  (d <= 6 ? collect(1:d) : [1, 2, 3, d])
    all(1 .<= show_coords .<= d) || throw(ArgumentError(
        "fig_tri: coords $(show_coords) out of range for d = $d"))
    nc = length(show_coords)
    fs(x) = round(Int, x * fontscale)
    fig = Figure(size = res)
    pcol = ALG_COLOR[mr.algorithm]
    Nref = size(truth.samples, 2)
    truth_idx = rand(rng, 1:Nref, min(2_000, Nref))
    truth_sub = truth.samples[:, truth_idx]
    w = mr.weights
    wsum = sum(w)
    pw = wsum > 0 ? ProbabilityWeights(w ./ wsum) : ProbabilityWeights(fill(1.0, length(w)))

    # V2-FIX-3b — equal-weight resample for the off-diagonal scatter.
    # Bound by the algorithm's actual sample count so we don't fabricate
    # "extra" samples for very-small-budget cells.
    N_show_use = max(1, min(N_show, size(mr.samples, 2)))
    samples_eq = _resample_to_eval_N(mr, N_show_use, rng)

    # Per-coordinate tight limits from the truth (P1-9).
    coord_lims = [tight_limits(truth; coords = (j, j))[1] for j in 1:d]

    # V2-FIX-3a — chain-aware Neff for mh/nuts, Kish for is/ns/mw.
    Neff_val = neff(mr)
    eta = mr.Nlike_used > 0 && isfinite(Neff_val) ? Neff_val / mr.Nlike_used : NaN

    for i in 1:nc, j in 1:nc
        ci = show_coords[i]
        cj = show_coords[j]
        ax = Axis(fig[i, j])
        standard_axis!(ax)
        # V8-FIX-C3: interior panels carry no tick labels (only the left
        # column and bottom row), ≤3 major ticks per axis, no minor
        # ticks — at d = 5 the 25-axis grid was unreadably dense.
        # Tick/label sizes scaled for half-width embedding.
        ax.xminorticksvisible = false
        ax.yminorticksvisible = false
        ax.xticks = Makie.WilkinsonTicks(3)
        ax.yticks = Makie.WilkinsonTicks(3)
        # fs(7) tick labels — at fs(8) three adjacent labels such
        # as "−3 0 3" nearly touched on the ~70 pt panels and read as
        # one merged string ("−30 3").
        ax.xticklabelsize = fs(7)
        ax.yticklabelsize = fs(7)
        ax.xlabelsize = fs(9)
        ax.ylabelsize = fs(9)
        (i < nc) && (ax.xticklabelsvisible = false)
        (j > 1) && (ax.yticklabelsvisible = false)
        if j > i
            hidedecorations!(ax)
            hidespines!(ax)
            continue
        elseif i == j
            # 1-D marginal: truth first (60 %), algorithm overlay (weighted KDE).
            ax.yticklabelsvisible = false   # density scale is arbitrary
            density!(ax, vec(truth_sub[ci, :]);
                color = (TRUTH_COLOR, 0.6),
                strokecolor = TRUTH_COLOR, strokewidth = 0.8)
            xs = vec(mr.samples[ci, :])
            try
                density!(ax, xs; weights = collect(pw),
                    color = (pcol, 0.30), strokecolor = pcol, strokewidth = 1.0)
            catch
                density!(ax, xs;
                    color = (pcol, 0.30), strokecolor = pcol, strokewidth = 1.0)
            end
            if i == nc
                ax.xlabel = LaTeXString("\$\\theta_{$(ci)}\$")
                ax.xticklabelsvisible = true
            end
            xl = coord_lims[ci]
            xlims!(ax, xl[1], xl[2])
        else
            # 2-D scatter: truth first (60 %), algorithm overlay from
            # the equal-weight resample (V2-FIX-3b). V8-FIX-C3:
            # rasterize the dense scatter panels — pure-vector markers
            # made each tri PDF 2–6 MB and slowed the thesis build.
            scatter!(ax, vec(truth_sub[cj, :]), vec(truth_sub[ci, :]);
                color = (TRUTH_COLOR, 0.6), markersize = 1.8, strokewidth = 0,
                rasterize = 2)
            x = vec(samples_eq[cj, :])
            y = vec(samples_eq[ci, :])
            scatter!(ax, x, y; color = (pcol, 0.30), markersize = 2.6,
                strokewidth = 0, rasterize = 2)
            if j == 1
                ax.ylabel = LaTeXString("\$\\theta_{$(ci)}\$")
                ax.yticklabelsvisible = true
            end
            if i == nc
                ax.xlabel = LaTeXString("\$\\theta_{$(cj)}\$")
                ax.xticklabelsvisible = true
            end
            xl = coord_lims[cj]; yl = coord_lims[ci]
            xlims!(ax, xl[1], xl[2])
            ylims!(ax, yl[1], yl[2])
        end
    end

    # V8-FIX-C3: in-figure legend naming the two sample clouds, placed
    # in the empty upper-right triangle. Explicit MarkerElements
    # with full-size, near-opaque swatches — the old legend reused the
    # raw scatter handles, whose 1.8–2.6 pt markers at 30–60 % alpha
    # rendered as invisible dots, so the legend showed text with no
    # visible colour patch.
    if nc >= 2
        truth_el = MarkerElement(color = (TRUTH_COLOR, 0.9),
            marker = :circle, markersize = fs(5))
        alg_el = MarkerElement(color = (pcol, 0.9),
            marker = :circle, markersize = fs(5))
        Legend(fig[1, nc], [truth_el, alg_el],
            ["truth reference", ALG_LABEL[mr.algorithm]];
            framevisible = false, tellwidth = false, tellheight = false,
            halign = :right, valign = :top, labelsize = fs(8),
            patchsize = (fs(6), fs(6)))
    end

    # Header label and metric annotation. The footer prints the
    # *protocol* Neff (chain-aware for mh/nuts) so it agrees with
    # cells.csv and the heatmap. V8-FIX-C2: thesis notation (no
    # code-style N_eff / N_show tokens); prose stays out of math mode
    # so problem-name hyphens render as hyphens.
    title_str = _title_latex(
        string(ALG_LABEL[mr.algorithm], " on ", PROBLEM_LABEL[mr.problem], ",");
        math = string("d = ", d, ",\\, B = ", _budget_latex(mr.B)))
    # Title at fs(7)/footer at fs(7) with compact wording — at fs(8)
    # the bold title and the "of the weighted draws shown" footer both
    # overran the right edge of the 396 pt canvas.
    Label(fig[0, :], title_str; fontsize = fs(7), font = :bold)
    metric_txt = LaTeXString(@sprintf(
        "\$N_{\\mathrm{eff}} = %.0f,\\;\\; \\eta = %.2g\\;\\; (%d\\;\\text{draws shown})\$",
        Neff_val, eta, N_show_use))
    Label(fig[nc + 1, :], metric_txt; fontsize = fs(7))
    # Tighten the inter-panel gaps (default ~18 pt each) so the
    # data panels claim more of the canvas — at d = 5 this enlarges
    # every panel by roughly 15 % at the embedded print size. Small
    # grids (d = 2) keep a wider gap: their panels are large and their
    # wide edge tick labels ("−10") would otherwise touch at the seam.
    # nc = 2 needs ~24 pt: the corner tick labels ("10" | "−10") extend
    # roughly half their width into the gap from both sides and touched
    # even at Makie's default gap.
    gap = nc <= 3 ? fs(12) : fs(2.5)
    rowgap!(fig.layout, gap)
    colgap!(fig.layout, gap)
    return fig
end


# B.6 — Per-problem Pareto  ---------------------------------------------------

function fig_pareto_per_problem(df::DataFrame, problem::Symbol,
                                  d::Integer = HEADLINE_DIMENSION)
    set_pub_theme!(class = :wide)
    res = figure_resolution(:wide, :pareto)
    fig = Figure(size = res)
    ax = Axis(fig[1, 1];
        xlabel = tex_label(:time_ess),
        ylabel = tex_label(:Nlike_ess),
        xscale = log10, yscale = log10,
        title = _title_latex(String(PROBLEM_LABEL[problem]);
            math = string("(d = ", Int(d), ",\\; B = ", _budget_latex(5e5), ")")))
    standard_axis!(ax)
    sub = _filter_problem_d(df, problem, d)
    sub = sub[sub.B .== 5e5, :]
    leg_pts = []
    leg_labels = String[]
    for alg in ALG_ORDER
        sub_alg = sub[sub.algorithm .== String(alg), :]
        isempty(sub_alg) && continue
        x = sub_alg.wall_time_s ./ max.(sub_alg.Neff, 1.0)
        y = sub_alg.Nlike_used ./ max.(sub_alg.Neff, 1.0)
        x_med = median(filter(!isnan, Vector{Float64}(x)))
        y_med = median(filter(!isnan, Vector{Float64}(y)))
        sc = scatter!(ax, [x_med], [y_med];
            color = ALG_COLOR[alg], marker = ALG_MARKER[alg], markersize = 12,
            strokecolor = ALG_COLOR[alg], strokewidth = 0.5)
        push!(leg_pts, sc)
        push!(leg_labels, ALG_LABEL[alg])
    end
    axislegend(ax, leg_pts, leg_labels; position = :lt)
    return fig
end


# =============================================================================
# CATALOGUE C — Algorithm-specific diagnostics (§10.10)
# =============================================================================

# C.1 — MoleWhacker iteration log (fix P1-5)  --------------------------------

"""
    fig_iter_mw(mr; KL_estimates=nothing) -> Figure

Top panel: η(t) = ess(t) / cum_cost(t), using the **recorded**
per-iteration cumulative cost from the MoleWhacker iteration log
(rather than a linear extrapolation, which previously placed the
initialisation at η = 400 — a bookkeeping artefact, not a sampler
property).

Bottom panel: KL-on-cube if estimates are provided, else component
count.
"""
function fig_iter_mw(mr::MethodResult; KL_estimates::Union{Nothing,Vector{Float64}} = nothing)
    @assert mr.algorithm == :mw
    set_pub_theme!()
    res = figure_resolution(:narrow, :iter)
    fig = Figure(size = res)
    iter_log = get(mr.extras, :iter_log, NamedTuple[])
    iters = [Float64(e.iter) for e in iter_log]
    eff = [Float64(e.eff) for e in iter_log]
    ess = [Float64(e.ess) for e in iter_log]
    n_components = [Float64(e.n_components) for e in iter_log]
    cum_cost = [haskey(e, :cum_cost) ? Float64(e.cum_cost) : NaN for e in iter_log]
    have_cum = any(isfinite, cum_cost)
    if !have_cum
        # Fallback: linear extrapolation. Marked in subtitle.
        cum_cost = max.(iters .* (mr.Nlike_used / max(maximum(iters), 1)), 1.0)
    end
    eta = ess ./ max.(cum_cost, 1.0)

    ax1 = Axis(fig[1, 1];
        xlabel = tex_label(:t),
        ylabel = tex_label(:eta),
        yscale = log10,
        title = have_cum ? L"\text{MoleWhacker iteration log}" :
            L"\text{MoleWhacker iteration log (linear cum-cost)}")
    standard_axis!(ax1)
    ok = (eta .> 0) .& isfinite.(eta)
    if any(ok)
        lines!(ax1, iters[ok], eta[ok]; color = ALG_COLOR[:mw], linewidth = 1.4)
        scatter!(ax1, iters[ok], eta[ok]; color = ALG_COLOR[:mw], markersize = 6)
    end

    ax2 = Axis(fig[2, 1];
        xlabel = tex_label(:t),
        ylabel = KL_estimates !== nothing ? tex_label(:KL) : L"\text{n components}")
    standard_axis!(ax2)
    if KL_estimates !== nothing
        ax2.yscale = log10
        lines!(ax2, iters, KL_estimates; color = ALG_COLOR[:mw], linewidth = 1.4)
        scatter!(ax2, iters, KL_estimates; color = ALG_COLOR[:mw], markersize = 6)
    else
        lines!(ax2, iters, n_components; color = ALG_COLOR[:mw], linewidth = 1.4)
        scatter!(ax2, iters, n_components; color = ALG_COLOR[:mw], markersize = 6)
    end
    return fig
end


# C.1b — MW iteration-convergence curves across seeds  ------------------

"""
    fig_mw_itercurves(runs; problem, d, B, class=:narrow) -> Figure

The core MoleWhacker iteration plot: cost efficiency
η(t) = N_eff(t) / N_L(t) over the adaptive iteration t, for every
available seed of one (problem, d, B) cell. One thin line per seed,
the across-seed median in bold, and an end-of-run marker on each seed
curve showing WHICH stopping rule fired:

* filled star  — iteration cap `T_max` reached (protocol-allowed
  early termination; the benchmark protocol disables the success
  stops, so this is the regular full-length run),
* open circle  — likelihood budget `B` exhausted mid-adaptation
  (the loop was cut off before `T_max`).

Both quantities are read from the per-iteration `iter_log` recorded
by every MoleWhacker run (ess and the exact cumulative counter cost),
so the figure needs no re-running.
"""
# Shared panel body for the iteration-convergence figures: per-seed
# curves at low alpha, the pointwise across-seed median in black, and a
# stop marker on each curve's final point (star = T_max cap, open
# circle = budget). Returns `true` when at least one curve was drawn.
function _mw_itercurves_panel!(ax, runs::Vector{MethodResult})
    curves = Vector{Tuple{Vector{Float64},Vector{Float64},Symbol}}()
    t_max_seen = 0
    for mr in runs
        il = get(mr.extras, :iter_log, NamedTuple[])
        isempty(il) && continue
        ts = Float64[]
        etas = Float64[]
        for e in il
            cum = haskey(e, :cum_cost) ? Float64(e.cum_cost) : NaN
            (isfinite(cum) && cum > 0) || continue
            eta = Float64(e.ess) / cum
            (isfinite(eta) && eta > 0) || continue
            push!(ts, Float64(e.iter))
            push!(etas, eta)
        end
        isempty(ts) && continue
        stop = Symbol(get(mr.extras, :stop_reason, :unknown))
        push!(curves, (ts, etas, stop))
        t_max_seen = max(t_max_seen, Int(maximum(ts)))
    end
    if isempty(curves)
        text!(ax, 0.5, 0.5; text = "no iteration logs",
            align = (:center, :center), space = :relative)
        return false
    end

    for (ts, etas, stop) in curves
        lines!(ax, ts, etas; color = (ALG_COLOR[:mw], 0.30), linewidth = 0.8)
        endcolor = ALG_COLOR[:mw]
        if stop === :T_max
            scatter!(ax, [ts[end]], [etas[end]]; marker = :star5,
                color = endcolor, markersize = 7, strokewidth = 0)
        else
            scatter!(ax, [ts[end]], [etas[end]]; marker = :circle,
                color = :white, strokecolor = endcolor, strokewidth = 1.0,
                markersize = 6)
        end
    end

    # Across-seed median at every iteration with at least one curve.
    med_t = Float64[]
    med_eta = Float64[]
    for t in 0:t_max_seen
        vals = Float64[]
        for (ts, etas, _) in curves
            k = findfirst(==(Float64(t)), ts)
            k === nothing || push!(vals, etas[k])
        end
        isempty(vals) && continue
        push!(med_t, Float64(t))
        push!(med_eta, median(vals))
    end
    length(med_t) > 1 && lines!(ax, med_t, med_eta;
        color = :black, linewidth = 1.6)

    # Tick fix: when the data span less than two decades (shell:
    # everything within a factor ~1.3) Makie's log locator falls back to
    # fractional-exponent ticks like 10^-4.72. Widen the limits to a
    # two-decade window centered on the data so tick marks sit on clean
    # integer decades.
    all_eta = Float64[]
    for (_, etas, _) in curves
        append!(all_eta, etas)
    end
    lo, hi = extrema(all_eta)
    if log10(hi / lo) < 2.0
        c = sqrt(lo * hi)
        ylims!(ax, c / 10^1.1, c * 10^1.1)
    end
    return true
end

const _MW_ITER_LEGEND_ELS = () -> [
    LineElement(color = :black, linewidth = 1.6),
    LineElement(color = (ALG_COLOR[:mw], 0.5), linewidth = 0.8),
    MarkerElement(marker = :star5, color = ALG_COLOR[:mw], markersize = 8),
    MarkerElement(marker = :circle, color = :white,
        strokecolor = ALG_COLOR[:mw], strokewidth = 1.0, markersize = 7)]

const _MW_ITER_LEGEND_LABELS = ["median", "per seed",
    L"\text{stop: } T_{\max}", L"\text{stop: budget}"]

function fig_mw_itercurves(runs::Vector{MethodResult};
                              problem::Symbol, d::Integer, B::Real,
                              class::Symbol = :narrow)
    set_pub_theme!(class = class)
    res = figure_resolution(class, :scaling)
    narrow = class === :narrow
    fig = Figure(size = res)
    ax = Axis(fig[1, 1];
        xlabel = L"t\;\;(\text{adaptive iteration})",
        ylabel = tex_label(:eta),
        yscale = log10,
        title = narrow ? "" :
            _title_latex(string(PROBLEM_LABEL[problem], ",");
                math = string("d = ", Int(d), ",\\, B = ", _budget_latex(B))))
    standard_axis!(ax)
    ax.xminorticksvisible = false
    ok = _mw_itercurves_panel!(ax, runs)
    ok && axislegend(ax, _MW_ITER_LEGEND_ELS(), _MW_ITER_LEGEND_LABELS;
        position = :rb, framevisible = false,
        labelsize = narrow ? 7 : 8, patchsize = (10, 10),
        rowgap = 0, padding = (2, 2, 2, 2))
    return fig
end

"""
    fig_mw_itercurves_quad(panels; B) -> Figure

Chapter-5 headline iteration figure: a 2x2 grid of
iteration-convergence panels, one per benchmark target, sharing one
legend. `panels` is a vector of `(problem, d, runs)` NamedTuples in
display order (row-major).
"""
function fig_mw_itercurves_quad(panels::Vector{<:NamedTuple}; B::Real = 5e5)
    set_pub_theme!(class = :wide)
    W, _ = figure_size(:wide, :scaling)
    fig = Figure(size = (round(Int, W), round(Int, 0.82 * W)))
    for (idx, p) in enumerate(panels)
        r = div(idx - 1, 2) + 1
        c = mod(idx - 1, 2) + 1
        ax = Axis(fig[r, c];
            xlabel = r == 2 ? L"t\;\;(\text{adaptive iteration})" : "",
            ylabel = c == 1 ? tex_label(:eta) : "",
            yscale = log10,
            title = _title_latex(string(PROBLEM_LABEL[p.problem], ",");
                math = string("d = ", Int(p.d))))
        standard_axis!(ax)
        ax.xminorticksvisible = false
        _mw_itercurves_panel!(ax, p.runs)
    end
    Legend(fig[3, :], _MW_ITER_LEGEND_ELS(), _MW_ITER_LEGEND_LABELS;
        orientation = :horizontal, framevisible = false,
        labelsize = 8, patchsize = (12, 10))
    rowgap!(fig.layout, 8)
    colgap!(fig.layout, 10)
    return fig
end

"""
    fig_convergence_cell(runs_by_alg; problem, d, B, markers) -> Figure

Cross-algorithm running-convergence overlay: for one
benchmark cell, the trajectory of the running efficiency
η(N_L) = N_eff(N_L)/N_L within a single representative run of every
algorithm, reconstructed from the stored output by
`convergence_curve`. The NS curve's horizontal placement is a uniform
cost smear (its per-iteration rejection cost is not recorded) and is
drawn dotted with an explicit "(approx. cost)" legend label. Optional
`markers` overlays the medians of the ACTUAL runs at each grid budget
as filled markers, so the within-run trajectory can be checked
against the independently run budget grid.
"""
function fig_convergence_cell(runs_by_alg::Vector{<:Pair};
                                 problem::Symbol, d::Integer, B::Real,
                                 markers::AbstractDict = Dict{Symbol,Any}())
    set_pub_theme!(class = :wide)
    res = figure_resolution(:wide, :scaling)
    fig = Figure(size = res)
    ax = Axis(fig[1, 1];
        xlabel = L"N_L\;\;\text{consumed (likelihood equivalents)}",
        ylabel = tex_label(:eta),
        xscale = log10, yscale = log10,
        title = _title_latex(string(PROBLEM_LABEL[problem], ",");
            math = string("d = ", Int(d), ",\\, B = ", _budget_latex(B))))
    standard_axis!(ax)
    ax.xminorticksvisible = false

    # Decluttered: clean curves only — no per-point
    # markers along the trajectories and no hollow budget-grid markers
    # (they read as unexplained symbols); a single filled marker tags
    # each curve's endpoint. Ultra-short prefixes (< 100 cost units)
    # are suppressed: their ESS estimates are noise and made the left
    # edge chaotic.
    leg_el = []
    leg_lab = String[]
    for (alg, mr) in runs_by_alg
        cc = convergence_curve(mr)
        length(cc.cost) >= 2 || continue
        eta = cc.neff ./ cc.cost
        ok = isfinite.(eta) .& (eta .> 0) .& (cc.cost .>= 100)
        count(ok) >= 2 || continue
        xs = cc.cost[ok]
        ys = eta[ok]
        style = alg === :ns ? :dot : ALG_LINESTYLE[alg]
        ln = lines!(ax, xs, ys;
            color = ALG_COLOR[alg],
            linewidth = ALG_LINEWIDTH[alg] + 0.4,
            linestyle = style)
        scatter!(ax, [xs[end]], [ys[end]];
            color = ALG_COLOR[alg], marker = ALG_MARKER[alg],
            markersize = 8, strokecolor = :white, strokewidth = 0.5)
        push!(leg_el, ln)
        push!(leg_lab, alg === :ns ?
            string(ALG_LABEL[alg], " (approx. cost)") : ALG_LABEL[alg])
    end
    if isempty(leg_el)
        text!(ax, 0.5, 0.5; text = "no reconstructible runs",
            align = (:center, :center), space = :relative)
        return fig
    end
    axislegend(ax, leg_el, leg_lab; position = :lb, framevisible = false,
        labelsize = 8, patchsize = (16, 8), rowgap = 0)
    return fig
end


"""
    fig_accuracy_cell(runs_by_alg; problem, d, B, mw_curve=nothing) -> Figure

Cross-algorithm accuracy-over-time overlay: the running marginal
Wasserstein-1 distance to the truth within one representative run per
algorithm (log-log, lower = better), reconstructed by
`accuracy_curve`. The MoleWhacker history is not reconstructible from
stored runs (final-weights-only) and is passed in explicitly as
`mw_curve = (cost, w1)` from a per-iteration snapshot run.
"""
function fig_accuracy_cell(runs_by_alg::Vector{<:Pair};
                              problem::Symbol, d::Integer, B::Real,
                              truth::TruthSet,
                              mw_curve::Union{Nothing,Tuple} = nothing)
    set_pub_theme!(class = :wide)
    res = figure_resolution(:wide, :scaling)
    fig = Figure(size = res)
    ax = Axis(fig[1, 1];
        xlabel = L"N_L\;\;\text{consumed (likelihood equivalents)}",
        ylabel = tex_label(:W1),
        xscale = log10, yscale = log10,
        title = _title_latex(string(PROBLEM_LABEL[problem], ",");
            math = string("d = ", Int(d), ",\\, B = ", _budget_latex(B))))
    standard_axis!(ax)
    ax.xminorticksvisible = false

    leg_el = []
    leg_lab = String[]
    for (alg, mr) in runs_by_alg
        alg === :mw && continue
        ac = accuracy_curve(mr, truth)
        length(ac.cost) >= 2 || continue
        ok = isfinite.(ac.w1) .& (ac.w1 .> 0) .& (ac.cost .>= 100)
        count(ok) >= 2 || continue
        style = alg === :ns ? :dot : ALG_LINESTYLE[alg]
        ln = lines!(ax, ac.cost[ok], ac.w1[ok];
            color = ALG_COLOR[alg], linewidth = ALG_LINEWIDTH[alg] + 0.4,
            linestyle = style)
        scatter!(ax, [ac.cost[ok][end]], [ac.w1[ok][end]];
            color = ALG_COLOR[alg], marker = ALG_MARKER[alg],
            markersize = 8, strokecolor = :white, strokewidth = 0.5)
        push!(leg_el, ln)
        push!(leg_lab, alg === :ns ?
            string(ALG_LABEL[alg], " (approx. cost)") : ALG_LABEL[alg])
    end
    if mw_curve !== nothing
        xs, ys = mw_curve
        ok = isfinite.(ys) .& (ys .> 0) .& (xs .> 0)
        if count(ok) >= 2
            ln = lines!(ax, xs[ok], ys[ok];
                color = ALG_COLOR[:mw], linewidth = ALG_LINEWIDTH[:mw] + 0.4,
                linestyle = ALG_LINESTYLE[:mw])
            scatter!(ax, [xs[ok][end]], [ys[ok][end]];
                color = ALG_COLOR[:mw], marker = ALG_MARKER[:mw],
                markersize = 8, strokecolor = :white, strokewidth = 0.5)
            push!(leg_el, ln)
            push!(leg_lab, string(ALG_LABEL[:mw], " (snapshot run)"))
        end
    end
    if isempty(leg_el)
        text!(ax, 0.5, 0.5; text = "no reconstructible runs",
            align = (:center, :center), space = :relative)
        return fig
    end
    axislegend(ax, leg_el, leg_lab; position = :lb, framevisible = false,
        labelsize = 8, patchsize = (16, 8), rowgap = 0)
    return fig
end


"""
    fig_runconv_panel(runs; algorithm, problem, d, B, class=:narrow) -> Figure

Single-algorithm running-efficiency panel (appendix atlas): for
one benchmark cell and ONE algorithm, the running efficiency
η = N_eff(n)/N_L(n) against the number of samples produced so far n
(log-log), with one light curve per seed and the across-seed median in
black — the same visual language as the MoleWhacker iteration panels.
The per-algorithm meaning of n and the reconstruction fidelity are
those of `convergence_curve`.
"""
function fig_runconv_panel(runs::Vector{MethodResult};
                              algorithm::Symbol, problem::Symbol,
                              d::Integer, B::Real, class::Symbol = :narrow)
    set_pub_theme!(class = class)
    res = figure_resolution(class, :scaling)
    narrow = class === :narrow
    fig = Figure(size = res)
    ax = Axis(fig[1, 1];
        xlabel = L"n\;\;(\text{samples produced})",
        ylabel = tex_label(:eta),
        xscale = log10, yscale = log10,
        title = narrow ? "" :
            _title_latex(string(ALG_LABEL[algorithm], " on ",
                                PROBLEM_LABEL[problem], ",");
                math = string("d = ", Int(d), ",\\, B = ", _budget_latex(B))))
    standard_axis!(ax)
    ax.xminorticksvisible = false
    _runconv_panel_body!(ax, runs, ALG_COLOR[algorithm])
    return fig
end

"""
    fig_runconv_grid(runs_by_alg; problem, d, B) -> Figure

Per-problem running-efficiency grid (appendix atlas): the five
single-algorithm panels of one benchmark cell combined into a 2x3
grid (one panel per algorithm, all seeds + across-seed median), with
a shared key in the sixth slot.
"""
function fig_runconv_grid(runs_by_alg::Vector{<:Pair};
                             problem::Symbol, d::Integer, B::Real)
    set_pub_theme!(class = :wide)
    W, _ = figure_size(:wide, :scaling)
    fig = Figure(size = (round(Int, W), round(Int, 0.72 * W)))
    n = length(runs_by_alg)
    for (idx, (alg, runs)) in enumerate(runs_by_alg)
        r = div(idx - 1, 3) + 1
        c = mod(idx - 1, 3) + 1
        ax = Axis(fig[r, c];
            xlabel = (idx + 3 > n) ? L"n\;\;(\text{samples})" : "",
            ylabel = c == 1 ? tex_label(:eta) : "",
            xscale = log10, yscale = log10,
            title = ALG_LABEL[alg])
        standard_axis!(ax)
        ax.xminorticksvisible = false
        ax.xticklabelsize = 7
        ax.yticklabelsize = 7
        _runconv_panel_body!(ax, runs, ALG_COLOR[alg])
    end
    Legend(fig[2, 3],
        [LineElement(color = :black, linewidth = 1.6),
         LineElement(color = (:gray50, 0.6), linewidth = 0.8)],
        ["across-seed median", "per seed"];
        framevisible = false, labelsize = 8, patchsize = (14, 8),
        halign = :center, valign = :center)
    rowgap!(fig.layout, 8)
    colgap!(fig.layout, 8)
    return fig
end

# Shared body of the running-efficiency panels: per-seed curves, the
# across-seed median on a common log grid, minimum-window handling for
# clean decade ticks, and the degenerate-cell annotation.
function _runconv_panel_body!(ax, runs::Vector{MethodResult}, col)
    curves = Vector{Tuple{Vector{Float64},Vector{Float64}}}()
    for mr in runs
        cc = convergence_curve(mr)
        length(cc.n) >= 2 || continue
        eta = cc.neff ./ cc.cost
        ok = isfinite.(eta) .& (eta .> 0) .& isfinite.(cc.n) .& (cc.n .> 0)
        count(ok) >= 2 || continue
        push!(curves, (cc.n[ok], eta[ok]))
    end
    if isempty(curves)
        text!(ax, 0.5, 0.5; text = "no reconstructible runs",
            align = (:center, :center), space = :relative)
        return nothing
    end

    for (ns, etas) in curves
        lines!(ax, ns, etas; color = (col, 0.30), linewidth = 0.8)
    end

    # Across-seed median on a common log-n grid (linear interpolation in
    # log-log space, defined where at least half the curves are).
    lo = maximum(first.(getindex.(curves, 1)))
    hi = minimum(last.(getindex.(curves, 1)))
    if hi > lo * 1.5
        grid = exp10.(range(log10(lo), log10(hi); length = 24))
        med_x = Float64[]
        med_y = Float64[]
        for g in grid
            vals = Float64[]
            for (ns, etas) in curves
                (g < ns[1] || g > ns[end]) && continue
                k = searchsortedfirst(ns, g)
                if k == 1
                    push!(vals, etas[1])
                else
                    t = (log10(g) - log10(ns[k-1])) /
                        (log10(ns[k]) - log10(ns[k-1]))
                    push!(vals, exp10((1 - t) * log10(etas[k-1]) +
                                      t * log10(etas[k])))
                end
            end
            length(vals) >= max(1, div(length(curves), 2)) || continue
            push!(med_x, g)
            push!(med_y, median(vals))
        end
        length(med_x) > 1 && lines!(ax, med_x, med_y;
            color = :black, linewidth = 1.6)
    end

    # Two-decade minimum y-span for clean decade ticks (as in the
    # MoleWhacker iteration panels).
    all_eta = Float64[]
    for (_, etas) in curves
        append!(all_eta, etas)
    end
    lo_e, hi_e = extrema(all_eta)
    if log10(hi_e / lo_e) < 2.0
        c = sqrt(lo_e * hi_e)
        ylims!(ax, c / 10^1.1, c * 10^1.1)
    end
    # Axis-window fix: on degenerate cells the sample count barely
    # moves (shell: the new components receive ~zero targeted draws, so
    # the population stays at its initial size) and Makie pads the
    # near-empty log axis to absurd ranges. Enforce a minimum
    # two-decade window centered on the data instead, render the
    # point-like curves as visible markers, and annotate the freeze so
    # the panel explains itself.
    all_n = Float64[]
    for (ns, _) in curves
        append!(all_n, ns)
    end
    lo_n, hi_n = extrema(all_n)
    if lo_n > 0 && log10(hi_n / lo_n) < 2.0
        c = sqrt(lo_n * hi_n)
        xlims!(ax, c / 10^1.1, c * 10^1.1)
    end
    if lo_n > 0 && log10(hi_n / lo_n) < 0.1
        for (ns, etas) in curves
            scatter!(ax, ns, etas; color = (col, 0.55), markersize = 5,
                strokewidth = 0)
        end
        text!(ax, 0.5, 0.72;
            text = @sprintf("population frozen\nat n = %d", round(Int, hi_n)),
            align = (:center, :center), space = :relative, fontsize = 8,
            color = :gray25)
    end
    return nothing
end


"""
    fig_mw_storyboard(frames, truth_samples; L=10.0, coords=(1,2)) -> Figure

Whacking-loop storyboard (thesis Ch. 5 §5.4): one panel per selected
iteration showing the truth reference (gray), an equal-weight resample
of the current population (algorithm color), and the current mixture
component centres (black-edged markers sized by mixture weight). All
positions are mapped from the PriorToNormal z-space to the user space
through the exact box-prior transform θ_i = 2L·Φ(z_i) − L and projected
onto `coords`.

`frames` is a vector of NamedTuples
`(t, K, centers_z, center_weights, samples_user, sample_weights)` where
`centers_z` is d×K, `samples_user` is d×N (already in user space, as
saved by the snapshot hook), and `sample_weights` are the current
importance weights of those samples.
"""
function fig_mw_storyboard(frames::Vector{<:NamedTuple},
                              truth_samples::AbstractMatrix;
                              L::Real = 10.0,
                              coords::Tuple{Int,Int} = (1, 2),
                              N_show::Int = 1_500,
                              rng::AbstractRNG = MersenneTwister(20260830))
    set_pub_theme!(class = :wide)
    n = length(frames)
    W, _ = figure_size(:wide, :scaling)
    panel = W / n
    fig = Figure(size = (round(Int, W), round(Int, panel + 34)))
    stdn = Normal()
    ci, cj = coords

    # Shared axis limits from the truth cloud.
    Nref = size(truth_samples, 2)
    tidx = rand(rng, 1:Nref, min(2_000, Nref))
    tx = truth_samples[ci, tidx]
    ty = truth_samples[cj, tidx]
    padx = 0.08 * (maximum(tx) - minimum(tx))
    pady = 0.08 * (maximum(ty) - minimum(ty))

    # Clean integer ticks sized to the truth cloud (Wilkinson picked
    # colliding decimal labels like "-2.5 0.0 2.5" on the narrow panels).
    tickx = let r = max(abs(minimum(tx)), abs(maximum(tx)))
        s = r > 4 ? 4.0 : 2.0
        [-s, 0.0, s]
    end
    ticky = let r = max(abs(minimum(ty)), abs(maximum(ty)))
        s = r > 4 ? 4.0 : 2.0
        [-s, 0.0, s]
    end

    for (k, fr) in enumerate(frames)
        ax = Axis(fig[1, k];
            xlabel = LaTeXString("\$\\theta_{$(ci)}\$"),
            ylabel = k == 1 ? LaTeXString("\$\\theta_{$(cj)}\$") : "",
            title = LaTeXString(@sprintf("\$t = %d\\;\\;(K = %d)\$", fr.t, fr.K)))
        standard_axis!(ax)
        ax.xminorticksvisible = false
        ax.yminorticksvisible = false
        ax.xticks = (tickx, [@sprintf("%g", v) for v in tickx])
        ax.yticks = (ticky, [@sprintf("%g", v) for v in ticky])
        (k > 1) && (ax.yticklabelsvisible = false)

        scatter!(ax, tx, ty; color = (TRUTH_COLOR, 0.5), markersize = 1.6,
            strokewidth = 0, rasterize = 2)

        # Equal-weight resample of the current population.
        w = fr.sample_weights
        Ns = size(fr.samples_user, 2)
        if Ns > 0 && sum(w) > 0
            pw = ProbabilityWeights(Float64.(w) ./ sum(w))
            sel = sample(rng, 1:Ns, pw, min(N_show, Ns); replace = true)
            scatter!(ax, fr.samples_user[ci, sel], fr.samples_user[cj, sel];
                color = (ALG_COLOR[:mw], 0.35), markersize = 2.4,
                strokewidth = 0, rasterize = 2)
        end

        # Component centres: z -> user space, projected. White outline
        # keeps them legible on top of the orange sample cloud.
        K = size(fr.centers_z, 2)
        for kk in 1:K
            mx = 2 * L * cdf(stdn, fr.centers_z[ci, kk]) - L
            my = 2 * L * cdf(stdn, fr.centers_z[cj, kk]) - L
            wk = fr.center_weights[kk]
            scatter!(ax, [mx], [my];
                color = :black, marker = :diamond,
                markersize = 3.5 + 7 * sqrt(max(wk, 0.0)),
                strokecolor = :white, strokewidth = 0.4)
        end

        xlims!(ax, minimum(tx) - padx, maximum(tx) + padx)
        ylims!(ax, minimum(ty) - pady, maximum(ty) + pady)
    end
    colgap!(fig.layout, 8)
    return fig
end


# C.2 — MW mole map  ---------------------------------------------------------

function fig_diag_mw_moles(mr::MethodResult, truth::TruthSet; L::Real = 10.0)
    @assert mr.algorithm == :mw
    set_pub_theme!()
    res = figure_resolution(:narrow, :viz_density)
    fig = Figure(size = res)
    ax = Axis(fig[1, 1];
        xlabel = L"\theta_1",
        ylabel = L"\theta_2",
        title = L"\text{MW component centres}",
        aspect = DataAspect())
    standard_axis!(ax)
    Nref = size(truth.samples, 2)
    idx = rand(1:Nref, min(1500, Nref))
    s = truth.samples[:, idx]
    # Truth first, larger and more visible (P1-8).
    scatter!(ax, s[1, :], s[2, :];
        color = (TRUTH_COLOR, 0.6), markersize = 1.5, strokewidth = 0)
    mix = get(mr.extras, :mixture, nothing)
    if mix !== nothing
        # NOTE: the mixture lives in the PriorToNormal-transformed
        # space; an earlier revision plotted its centres RAW over the
        # user-space truth, which collapsed them near the origin. Map
        # each centre through the box-prior inverse transform
        # θ_i = 2L·Φ(z_i) − L before overlaying.
        stdn = Normal()
        try
            for (k, comp) in enumerate(mix.components)
                μz = mean(comp)
                μ1 = 2 * L * cdf(stdn, μz[1]) - L
                μ2 = 2 * L * cdf(stdn, μz[2]) - L
                w = probs(mix)[k]
                scatter!(ax, [μ1], [μ2];
                    color = ALG_COLOR[:mw], markersize = 6 + 18 * w,
                    strokewidth = 0.4, strokecolor = :black)
            end
        catch
        end
    end
    xlims, ylims = tight_limits(truth)
    xlims!(ax, xlims[1], xlims[2])
    ylims!(ax, ylims[1], ylims[2])
    return fig
end


# C.3 — NUTS divergence map (fix P2-11)  --------------------------------------

"""
    fig_diag_nuts_div(mr, truth) -> Union{Figure, Nothing}

Returns `nothing` if no divergent transitions are stored in the
extras dict (P2-11) — the caller should not save an empty figure.
"""
function fig_diag_nuts_div(mr::MethodResult, truth::TruthSet)
    @assert mr.algorithm == :nuts
    div_idx = get(mr.extras, :divergent_indices, Int[])
    if !(div_idx isa AbstractVector) || isempty(div_idx)
        return nothing       # caller drops the figure
    end
    set_pub_theme!()
    res = figure_resolution(:narrow, :viz_density)
    fig = Figure(size = res)
    ax = Axis(fig[1, 1];
        xlabel = L"\theta_1",
        ylabel = L"\theta_2",
        title = L"\text{NUTS samples; divergent transitions in red}",
        aspect = DataAspect())
    standard_axis!(ax)
    scatter!(ax, vec(mr.samples[1, :]), vec(mr.samples[2, :]);
        color = (ALG_COLOR[:nuts], 0.30), markersize = 2, strokewidth = 0)
    scatter!(ax, vec(mr.samples[1, div_idx]), vec(mr.samples[2, div_idx]);
        color = :red, markersize = 4, strokewidth = 0)
    xlims, ylims = tight_limits(truth)
    xlims!(ax, xlims[1], xlims[2])
    ylims!(ax, ylims[1], ylims[2])
    return fig
end


# C.4 — MH trace + ACF  -------------------------------------------------------

function fig_diag_mh_trace(mr::MethodResult)
    @assert mr.algorithm == :mh
    set_pub_theme!(class = :wide)
    res = figure_resolution(:wide, :conv)
    fig = Figure(size = res)
    ax1 = Axis(fig[1, 1];
        xlabel = "step",
        ylabel = L"\theta_1",
        title = L"\text{trace}")
    standard_axis!(ax1)
    lines!(ax1, 1:size(mr.samples, 2), vec(mr.samples[1, :]);
        color = ALG_COLOR[:mh], linewidth = 0.8)
    ax2 = Axis(fig[1, 2];
        xlabel = "step",
        ylabel = L"\theta_2")
    standard_axis!(ax2)
    lines!(ax2, 1:size(mr.samples, 2), vec(mr.samples[2, :]);
        color = ALG_COLOR[:mh], linewidth = 0.8)
    ax3 = Axis(fig[2, 1];
        xlabel = "lag",
        ylabel = "ACF",
        title = L"\text{autocorrelation}")
    standard_axis!(ax3)
    lags = 0:200
    acf1 = _acf(vec(mr.samples[1, :]), maximum(lags))
    lines!(ax3, lags, acf1; color = ALG_COLOR[:mh], linewidth = 0.8)
    ax4 = Axis(fig[2, 2]; xlabel = "lag", ylabel = "ACF")
    standard_axis!(ax4)
    acf2 = _acf(vec(mr.samples[2, :]), maximum(lags))
    lines!(ax4, lags, acf2; color = ALG_COLOR[:mh], linewidth = 0.8)
    return fig
end

function _acf(x::AbstractVector, maxlag::Integer)
    n = length(x)
    μ = mean(x)
    var_x = var(x; corrected = false)
    acf = zeros(maxlag + 1)
    @inbounds for k in 0:maxlag
        s = 0.0
        for i in 1:(n - k)
            s += (x[i] - μ) * (x[i + k] - μ)
        end
        acf[k + 1] = s / ((n - k) * max(var_x, eps()))
    end
    return acf
end


# C.5 — NS log-density-vs-rank (relabelled per P1-6) --------------------------

"""
    fig_diag_ns_live(mr) -> Figure

Plots posterior log-density across the NS draws sorted by rank.
This is **not** the live-point log-likelihood evolution (P1-6); it is
what the post-NS resampled output looks like. The title and the y-axis
label are honest about that.
"""
function fig_diag_ns_live(mr::MethodResult)
    @assert mr.algorithm == :ns
    set_pub_theme!(class = :wide)
    res = figure_resolution(:wide, :conv)
    fig = Figure(size = res)
    ax = Axis(fig[1, 1];
        xlabel = L"-\log\,\text{rank}",
        ylabel = L"\log f(\theta_{\text{rank}})",
        title = L"\text{NS posterior log-density across draws (sorted)}")
    standard_axis!(ax)
    logL = sort(mr.logd)
    logX = -log.(reverse(1:length(logL)))
    lines!(ax, logX, logL; color = ALG_COLOR[:ns], linewidth = 1.0)
    return fig
end


# C.6 — IS weight histogram  --------------------------------------------------

function fig_diag_is_weights(mr::MethodResult)
    @assert mr.algorithm == :is
    set_pub_theme!()
    res = figure_resolution(:narrow, :viz_marginal)
    fig = Figure(size = res)
    ax = Axis(fig[1, 1];
        xlabel = L"\log w",
        ylabel = "count",
        yscale = log10,
        title = _title_latex("IS weights";
            math = string("(d = ", mr.d, ",\\; B = ", _budget_latex(mr.B), ")")))
    standard_axis!(ax)
    lw = log.(mr.weights .+ eps())
    # V8-FIX-C3: manual bin counts with a 0.5 floor — `hist!` on a log-y
    # axis errors/clips on zero-count bins for sparse weight histograms.
    lo, hi = extrema(lw)
    hi <= lo && (hi = lo + 1.0)
    nb = 50
    edges = range(lo, hi; length = nb + 1)
    counts = zeros(nb)
    for v in lw
        b = clamp(1 + floor(Int, (v - lo) / (hi - lo) * nb), 1, nb)
        counts[b] += 1
    end
    centers = collect(edges[1:end-1] .+ step(edges) / 2)
    barplot!(ax, centers, max.(counts, 0.5);
        color = ALG_COLOR[:is], strokewidth = 0, gap = 0.0)
    eta = neff_kish(mr) / max(mr.Nlike_used, 1.0)
    text!(ax, 0.05, 0.95;
        text = LaTeXString(@sprintf("\$\\eta = %.2g\$", eta)),
        space = :relative, fontsize = 8, align = (:left, :top))
    return fig
end


# =============================================================================
# CATALOGUE D — Aggregated summaries (§10.11)
# =============================================================================

# D.1 — Performance heatmap (V5 §2 — eggbox d=2 sub-block layout)  ----------

"""
    HEATMAP_PROBLEM_ORDER

Row order for headline summary heatmaps (top-to-bottom). V8-FIX-C3:
now identical to the canonical `PROBLEM_ORDER` — the V5 variant swapped
shell/mridges_spiky, so heatmaps and LaTeX tables ordered problems
differently from every other figure family. Eggbox stays last as the
labelled `d = 2` sub-block (separated by a thin horizontal rule).
"""
const HEATMAP_PROBLEM_ORDER = PROBLEM_NAMES

"""
    HEATMAP_PROBLEM_DIM(p, requested_d) -> Int

The dimension at which problem `p` is reported in the heatmaps. All
problems are reported at `requested_d` *except* `:eggbox`, which is
always at `d = 2` (its canonical setting per Mukherjee 2006 — at
`d ≥ 3` the lattice combinatorics dominate the metric and the
algorithms can no longer be discriminated).
"""
_heatmap_problem_dim(p::Symbol, requested_d::Integer) =
    p === :eggbox ? 2 : Int(requested_d)

"""
    fig_summary_heatmap(df, metric_col; transform=:negative_log10,
                          d=HEADLINE_DIMENSION, B=5e5, cap_dlogZ=10.0)

Render a single headline summary heatmap:

* **Rows** are problems in V5 canonical order (`HEATMAP_PROBLEM_ORDER`):
  six d=5 problems on top, then a thin horizontal rule, then eggbox at
  d=2 as a labelled sub-block. Y-tick labels reflect each row's
  reporting dimension (e.g. `Eggbox  /  d=2`).
* **Columns** are algorithms in canonical order (`ALG_ORDER`); for
  `dlogZ` the chain methods MH/NUTS are dropped entirely since
  they never produce an evidence estimate.
* **Grey cells** mark (problem, algorithm) combinations with no
  admissible run (budget-infeasible or all seeds flag-excluded).
* **Saturation glyphs** (lower-right of cell) flag protocol-allowed
  early termination: `★` for MW (`T_max` / `Neff_target`) and `▲`
  for NS (`dlogz`). Uniform white-on-black-outline style;
  legend lines render only for glyph/cell types actually present.
* **dlogZ colour cap**: `cap_dlogZ` caps the COLOUR scale only; the
  inscribed number is always the exact median.
* **Stray seeds** are dropped at the data-cleaning layer
  (`drop_v5_excluded_rows!` is called by `_do_catalogue_D` and the
  V5 driver), not here.
"""
function fig_summary_heatmap(df::DataFrame, metric_col::Symbol;
                                transform::Symbol = :negative_log10,
                                d::Integer = HEADLINE_DIMENSION,
                                B::Real = 5e5,
                                cap_dlogZ::Real = 10.0,
                                problems::Vector{Symbol} = collect(HEATMAP_PROBLEM_ORDER))
    set_pub_theme!(class = :wide)
    res = figure_resolution(:wide, :heatmap)
    fig = Figure(size = res)

    # dlogZ is undefined for the chain samplers MH and NUTS (they never
    # estimate the marginal likelihood), so the dlogZ heatmap drops those
    # columns entirely instead of rendering two permanently-grey stripes
    # (the caption explains the reduced column set).
    algs   = metric_col === :dlogZ ? Symbol[:is, :ns, :mw] : collect(ALG_ORDER)
    n_alg  = length(algs)
    n_prob = length(problems)
    raw        = fill(NaN, n_prob, n_alg)
    score      = fill(NaN, n_prob, n_alg)
    flagged    = falses(n_prob, n_alg)            # no admissible seeds
    saturation = fill(:none, n_prob, n_alg)

    for (i, prob) in enumerate(problems),
        (j, alg) in enumerate(algs)
        # Per-row dimension: d=2 for eggbox, requested_d otherwise.
        d_i = _heatmap_problem_dim(prob, d)
        ss = df[df.problem   .== String(prob) .&&
                df.algorithm .== String(alg)  .&&
                df.d         .== d_i          .&&
                df.B         .== Float64(B), :]
        if isempty(ss)
            flagged[i, j] = true
            continue
        end
        # Drop genuine failures before computing the median.
        keep = trues(nrow(ss))
        if hasproperty(ss, :notes)
            for (k, n) in enumerate(ss.notes)
                ns = String(coalesce(n, ""))
                if occursin("RHAT-FAIL", ns) ||
                   occursin("BUDGET-VIOLATION", ns) ||
                   occursin("FAILED-SANITY", ns)
                    keep[k] = false
                end
            end
        end
        v = filter(!isnan, Vector{Float64}(ss[keep, metric_col]))
        if isempty(v)
            flagged[i, j] = true
            continue
        end
        m = median(v)
        # The inscribed number is always the exact median; only the
        # COLOUR scale is capped for dlogZ, so that a single catastrophic
        # cell (shell x MW, median 25 nats) does not compress the colour
        # resolution of the rest of the map. An earlier revision capped
        # the printed value too and silently displayed "10" for 25.
        raw[i, j] = m
        m_col = metric_col === :dlogZ ? min(m, cap_dlogZ) : m
        score[i, j] = if transform === :negative_log10
            -log10(max(m_col, 1e-12))
        elseif transform === :log10
            log10(max(m_col, 1e-12))
        else
            m_col
        end
        # Saturation tag.
        if hasproperty(ss, :terminated_by)
            kept_idx = findall(keep)
            tag_counts = Dict{Symbol,Int}()
            for k in kept_idx
                isnan(ss[k, metric_col]) && continue
                tb = String(coalesce(ss.terminated_by[k], ""))
                isempty(tb) && continue
                if (alg === :mw && tb in ("T_max", "Neff_target")) ||
                   (alg === :ns && tb == "dlogz")
                    sym = Symbol(tb)
                    tag_counts[sym] = get(tag_counts, sym, 0) + 1
                end
            end
            tot = isempty(tag_counts) ? 0 : sum(values(tag_counts))
            tot > 0 && tot >= length(v) &&
                (saturation[i, j] = argmax(tag_counts))
        end
    end

    # Row labels: append `/  d=2` to the eggbox row; everything else
    # carries the requested d in the title rather than the row label.
    yticktext = String[]
    for prob in problems
        d_i = _heatmap_problem_dim(prob, d)
        push!(yticktext, d_i == Int(d) ? PROBLEM_LABEL[prob] :
            string(PROBLEM_LABEL[prob], "  /  d = ", d_i))
    end

    # V8-FIX-C2: budgets in thesis notation (5×10⁵, not "5e5").
    _title_math = string("(d = ", Int(d), ",\\; B = ", _budget_latex(B), ")")
    ax = Axis(fig[1, 1];
        xticks = (1:n_alg,  [ALG_LABEL[a] for a in algs]),
        yticks = (1:n_prob, yticktext),
        yreversed = true,                # MVN at top, eggbox at bottom
        title = haskey(METRIC_TITLE, metric_col) ?
            LaTeXString(string("\$", _strip_math(METRIC_TITLE[metric_col]),
                "\\;\\;", _title_math, "\$")) :
            _title_latex(String(metric_col); math = _title_math))
    standard_axis!(ax)
    # No minor ticks on a categorical axis.
    ax.xminorticksvisible = false
    ax.yminorticksvisible = false
    ax.xticksvisible = false
    ax.yticksvisible = false

    finite_score = filter(isfinite, vec(score))
    if isempty(finite_score)
        text!(ax, 0.5, 0.5; text = "no finite values",
            align = (:center, :center), space = :relative)
        return fig
    end
    cmin, cmax = extrema(finite_score)
    cmin == cmax && (cmin -= 0.5; cmax += 0.5)

    hm = heatmap!(ax, 1:n_alg, 1:n_prob, score';
        colormap = :viridis, nan_color = :transparent,
        colorrange = (cmin, cmax))

    # Light-grey fill for excluded cells: either no run reached this
    # (problem, budget) combination (budget-infeasible) or every seed was
    # dropped by a protocol flag (RHAT-FAIL / BUDGET-VIOLATION).
    for i in 1:n_prob, j in 1:n_alg
        if flagged[i, j]
            poly!(ax, [(j - 0.5, i - 0.5), (j + 0.5, i - 0.5),
                        (j + 0.5, i + 0.5), (j - 0.5, i + 0.5)];
                color = (:gray85, 0.35), strokecolor = :gray60,
                strokewidth = 0.4)
        end
    end

    Colorbar(fig[1, 2], hm; label = haskey(SCORE_LABEL, metric_col) ?
        SCORE_LABEL[metric_col] : LaTeXString(String(transform)),
        ticklabelsize = 9, labelsize = 10)

    # Inscribed numbers + saturation glyphs. V8-FIX-C3: text colour is
    # luminance-aware (white on dark viridis, black on bright yellow —
    # the old hard-coded white was invisible on the best cells).
    for i in 1:n_prob, j in 1:n_alg
        v = raw[i, j]
        flagged[i, j] && continue
        txt = if 0.01 <= abs(v) <= 100
            @sprintf("%.2g", v)
        else
            @sprintf("%.1e", v)
        end
        cell_col = _cell_text_color(score[i, j], cmin, cmax)
        text!(ax, j, i; text = txt, align = (:center, :center),
            color = cell_col, fontsize = 8)
        # Saturation glyphs: one consistent style everywhere — white fill
        # with a black outline stays legible on both ends of viridis.
        # (an earlier luminance-matched colour made the glyph black
        # on bright cells and white on dark ones, which read as two
        # different symbols.)
        sat = saturation[i, j]
        if sat in (:T_max, :Neff_target)
            text!(ax, j + 0.34, i - 0.34; text = "★",
                align = (:center, :center), color = :white,
                strokecolor = :black, strokewidth = 0.9, fontsize = 9)
        elseif sat === :dlogz
            text!(ax, j + 0.34, i - 0.34; text = "▲",
                align = (:center, :center), color = :white,
                strokecolor = :black, strokewidth = 0.9, fontsize = 9)
        end
    end

    # Thin horizontal rule above the eggbox (d=2) sub-block — V5 §2.
    eggbox_row = findfirst(==(:eggbox), problems)
    if eggbox_row !== nothing && eggbox_row > 1
        y_rule = eggbox_row - 0.5
        lines!(ax, [0.5, n_alg + 0.5], [y_rule, y_rule];
            color = :black, linewidth = 1.2)
    end

    # Caption-side legend for the saturation glyphs and grey cells.
    # Every legend line is conditional on the marker/cell type
    # actually appearing in THIS figure — the old unconditional legend
    # advertised an NS triangle and a dark-grey cell class that several
    # heatmaps never contain.
    # No hyphen in "evidence converged" — MathTeXEngine renders a hyphen
    # inside \text as a math minus. The MW glyph names only the T_max cap:
    # the ESS success stops are disabled under the benchmark protocol
    # (run_mw sets target_ess = Inf), so T_max is the only rule that can
    # fire under budget.
    has_star = any(s -> s in (:T_max, :Neff_target), saturation)
    has_tri  = any(==(:dlogz), saturation)
    has_gap  = any(flagged)
    row = 2
    if has_star || has_tri
        parts = String[]
        has_star && push!(parts,
            "\\bigstar\\;\\text{MW stopped early (iteration cap } T_{\\max}\\text{ reached, budget not exhausted)}")
        has_tri && push!(parts,
            "\\blacktriangle\\;\\text{NS stopped early (evidence converged)}")
        # One label per line: a single joined line overflows the canvas
        # when both glyph classes appear in the same figure.
        for p in parts
            Label(fig[row, :], LaTeXString("\$" * p * "\$");
                fontsize = 8, halign = :center)
            row += 1
        end
    end
    if has_gap
        Label(fig[row, :],
            LaTeXString("\$\\text{gray: no admissible run (mixing failure or infeasible budget)}\$");
            fontsize = 8, halign = :center)
    end
    return fig
end


# D.2 — Scaling  --------------------------------------------------------------

function fig_scaling(df::DataFrame, problem::Symbol; B::Real = 5e5,
                        class::Symbol = :wide)
    # `class = :narrow` renders at half-column width (190 pt) for
    # side-by-side subfigure use in the thesis.
    set_pub_theme!(class = class)
    res = figure_resolution(class, :scaling)
    narrow = class === :narrow
    fig = Figure(size = res)
    # V8-FIX-C2: prose in text mode (real hyphens/dashes), budget in
    # thesis notation. The old `LaTeXString(string(...))` rendered
    # "M − Ridges — scaling at B = 5e5" through math mode.
    # At half-column width the thesis subcaption already names the
    # problem and budget, so the in-figure title is dropped there.
    ax = Axis(fig[1, 1];
        xlabel = tex_label(:d),
        ylabel = tex_label(:W1),
        yscale = log10,
        title = narrow ? "" :
            _title_latex(string(PROBLEM_LABEL[problem], ", scaling at");
                         math = string("B = ", _budget_latex(B))),
        xticks = ([2, 5, 10], ["2", "5", "10"]))
    standard_axis!(ax)
    sub = df[df.problem .== String(problem) .&& df.B .== Float64(B), :]
    leg_lines = []
    leg_labels = String[]
    all_y = Float64[]
    for alg in ALG_ORDER
        sub_alg = sub[sub.algorithm .== String(alg), :]
        isempty(sub_alg) && continue
        per_d = combine(groupby(sub_alg, :d)) do s
            keep = trues(nrow(s))
            if hasproperty(s, :notes)
                for (k, n) in enumerate(s.notes)
                    ns = String(coalesce(n, ""))
                    if occursin("RHAT-FAIL", ns) || occursin("BUDGET-VIOLATION", ns) ||
                       occursin("FAILED-SANITY", ns)
                        keep[k] = false
                    end
                end
            end
            v = filter(!isnan, Vector{Float64}(s[keep, :W1_marginal_avg]))
            isempty(v) && return DataFrame(median = NaN, lower = NaN, upper = NaN)
            return DataFrame(median = median(v), lower = minimum(v),
                              upper = maximum(v))
        end
        sort!(per_d, :d)
        ds = Vector{Float64}(per_d.d)
        meds = Vector{Float64}(per_d.median)
        ok = .!isnan.(meds)
        any(ok) || continue
        ln = lines!(ax, ds[ok], meds[ok];
            color = ALG_COLOR[alg], linewidth = ALG_LINEWIDTH[alg],
            linestyle = ALG_LINESTYLE[alg])
        scatter!(ax, ds[ok], meds[ok];
            color = ALG_COLOR[alg], marker = ALG_MARKER[alg], markersize = 8)
        push!(leg_lines, ln)
        push!(leg_labels, ALG_LABEL[alg])
        append!(all_y, meds[ok])
    end
    _apply_log10_yticks!(ax, all_y)
    # At half-column width an in-axis legend covered the curves, so
    # it moves below the axis as a compact horizontal strip.
    if narrow
        isempty(leg_lines) || Legend(fig[2, 1], leg_lines, leg_labels;
            orientation = :horizontal, nbanks = 2, framevisible = false,
            labelsize = 7, patchsize = (14, 4), colgap = 8, rowgap = 0,
            padding = (0, 0, 0, 0))
    else
        axislegend(ax, leg_lines, leg_labels; position = :lt)
    end
    return fig
end


# D.3 — Grand Pareto (fix P2-8) ----------------------------------------------

function fig_pareto_grand(df::DataFrame, d::Integer = HEADLINE_DIMENSION;
                              B::Real = 5e5,
                              xclip::Tuple{Real,Real} = (1e-5, 1e0),
                              yclip::Tuple{Real,Real} = (1e0, 1e6))
    set_pub_theme!(class = :wide)
    res = figure_resolution(:wide, :pareto_grand)
    fig = Figure(size = res)
    ax = Axis(fig[1, 1];
        xlabel = tex_label(:time_ess),
        ylabel = tex_label(:Nlike_ess),
        xscale = log10, yscale = log10,
        title = _title_latex("Cross-problem Pareto";
            math = string("(d = ", Int(d), ",\\; B = ", _budget_latex(B), ")")))
    standard_axis!(ax)
    sub = df[df.d .== Int(d) .&& df.B .== Float64(B), :]
    # Algorithm-colour and problem-shape decoupled legends.
    alg_keys = MarkerElement[]
    alg_labels = String[]
    for alg in ALG_ORDER
        push!(alg_keys, MarkerElement(marker = :circle,
                                          color = ALG_COLOR[alg],
                                          markersize = 12,
                                          strokewidth = 0))
        push!(alg_labels, ALG_LABEL[alg])
    end
    prob_keys = MarkerElement[]
    prob_labels = String[]
    for prob in PROBLEM_ORDER
        push!(prob_keys, MarkerElement(marker = PROBLEM_MARKER[prob],
                                            color = :gray50,
                                            markersize = 12,
                                            strokewidth = 0))
        push!(prob_labels, PROBLEM_LABEL[prob])
    end

    n_clipped = 0
    for alg in ALG_ORDER
        sub_alg = sub[sub.algorithm .== String(alg), :]
        for prob in PROBLEM_ORDER
            sub_pp = sub_alg[sub_alg.problem .== String(prob), :]
            isempty(sub_pp) && continue
            x = sub_pp.wall_time_s ./ max.(sub_pp.Neff, 1.0)
            y = sub_pp.Nlike_used ./ max.(sub_pp.Neff, 1.0)
            xc = filter(isfinite, Vector{Float64}(x))
            yc = filter(isfinite, Vector{Float64}(y))
            (isempty(xc) || isempty(yc)) && continue
            x_med = median(xc)
            y_med = median(yc)
            isfinite(x_med) || continue
            isfinite(y_med) || continue
            xshow = clamp(x_med, xclip[1], xclip[2])
            yshow = clamp(y_med, yclip[1], yclip[2])
            clipped = xshow != x_med || yshow != y_med
            n_clipped += clipped ? 1 : 0
            scatter!(ax, [xshow], [yshow];
                color = ALG_COLOR[alg],
                marker = clipped ? :rtriangle : PROBLEM_MARKER[prob],
                markersize = clipped ? 14 : 12,
                strokecolor = ALG_COLOR[alg], strokewidth = 0.5)
        end
    end

    Legend(fig[1, 2],
        [alg_keys, prob_keys],
        [alg_labels, prob_labels],
        ["algorithm", "problem"]; framevisible = false)
    if n_clipped > 0
        text!(ax, 0.02, 0.02;
            text = @sprintf("▶ = %d clipped outlier(s)", n_clipped),
            space = :relative, fontsize = 8, align = (:left, :bottom),
            color = :gray30)
    end
    return fig
end


# =============================================================================
# V3 Catalogue extensions (Protocol §10.17, rev May 2026)
# =============================================================================
#
# Adds three new figure families and one CD-diagram tool that the v2
# pipeline did not have:
#
#   • V3-F1 — `fig_hardness_mridges_spiky`
#   • V3-F2 — `fig_recovery`
#   • V3-F4 — `fig_dim_grid`
#   • V3-F11 — `fig_tests_heatmap`
#   • V3-CD — `fig_cd_diagram` (Demšar 2006 critical-difference plot)
#
# All five honour the protocol's typography, palette, and filename
# conventions.

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

function _filter_keep(df::DataFrame; honour_flags::Bool = true)
    keep = trues(nrow(df))
    honour_flags || return keep
    if hasproperty(df, :notes) && hasproperty(df, :terminated_by) &&
       hasproperty(df, :algorithm)
        for k in 1:nrow(df)
            ns = String(coalesce(df.notes[k], ""))
            if occursin("RHAT-FAIL", ns) || occursin("FAILED-SANITY", ns)
                keep[k] = false
                continue
            end
            if occursin("BUDGET-VIOLATION", ns)
                # The aggregator now flags only non-exempt overshoots
                # (V2-FIX-1), so a literal `BUDGET-VIOLATION` after V2 is
                # unconditionally a real violation.
                keep[k] = false
            end
        end
    end
    return keep
end


# -----------------------------------------------------------------------------
# V3-F1 — `hardness__mridges-spiky`
# -----------------------------------------------------------------------------

"""
    fig_hardness_mridges_spiky(df; metric=:W1_marginal_avg, d=5, B=5e5)

Single-dial hardness sweep over `σ_spike`, two panels for `M ∈ {1, 4}`.
The frame requires a `σ_spike` column in `df` (added by the v3
runner via `extras.σ_spike`); cells that lack it are skipped.

The figure visualises MW's headline claim: at small `σ_spike`
(narrow spikes) MW remains close to the truth while every other
algorithm breaks down. Median over seeds with a paired-bootstrap
band; per-seed points overlaid for transparency.
"""
function fig_hardness_mridges_spiky(df::DataFrame;
                                       metric::Symbol = :W1_marginal_avg,
                                       d::Integer = HEADLINE_DIMENSION,
                                       B::Real = 5e5)
    set_pub_theme!(class = :wide)
    res = figure_resolution(:wide, :conv)
    fig = Figure(size = res)
    sub = df[df.problem .== "mridges_spiky" .&&
              df.d .== Int(d) .&& df.B .== Float64(B), :]
    if !hasproperty(sub, :sigma_spike) || !hasproperty(sub, :M_spikes)
        # V8-FIX-C1: never emit a placeholder PDF into the deliverable
        # set — the V6 placeholder ("run v3 grid first") shipped all the
        # way into the thesis figures folder. Raise instead; the driver
        # catches, logs, and produces NO file for this slot.
        error("fig_hardness_mridges_spiky: cells.csv has no " *
              "sigma_spike / M_spikes columns — the hardness sweep grid " *
              "was not run; refusing to emit a placeholder figure.")
    end
    Mvals = sort!(unique(sub.M_spikes))
    n_panels = length(Mvals)
    n_panels == 0 && (return fig)
    leg_lines = []
    leg_labels = String[]
    for (col, Mv) in enumerate(Mvals)
        ax = Axis(fig[1, col];
            xlabel = LaTeXString("\$\\sigma_\\mathrm{spike}\$"),
            ylabel = SCORE_LABEL[metric],
            xscale = log10,
            title = LaTeXString(string("M = ", Int(Mv))))
        standard_axis!(ax)
        sub_M = sub[sub.M_spikes .== Mv, :]
        for alg in ALG_ORDER
            sub_alg = sub_M[sub_M.algorithm .== String(alg), :]
            keep = _filter_keep(sub_alg)
            sub_alg = sub_alg[keep, :]
            isempty(sub_alg) && continue
            grouped = combine(groupby(sub_alg, :sigma_spike)) do s
                v = filter(isfinite, Vector{Float64}(s[!, metric]))
                isempty(v) && return DataFrame(σ = s.sigma_spike[1], median = NaN)
                return DataFrame(σ = s.sigma_spike[1], median = median(v))
            end
            sort!(grouped, :σ)
            isempty(grouped) && continue
            xs = Vector{Float64}(grouped.σ)
            ys = Vector{Float64}(grouped.median)
            ok = isfinite.(ys)
            any(ok) || continue
            score = -log10.(ys[ok])
            ln = lines!(ax, xs[ok], score;
                color = ALG_COLOR[alg], linewidth = ALG_LINEWIDTH[alg],
                linestyle = ALG_LINESTYLE[alg])
            scatter!(ax, xs[ok], score;
                color = ALG_COLOR[alg], marker = ALG_MARKER[alg],
                markersize = 9)
            if col == 1
                push!(leg_lines, ln)
                push!(leg_labels, ALG_LABEL[alg])
            end
        end
    end
    if !isempty(leg_lines)
        Legend(fig[1, n_panels + 1], leg_lines, leg_labels,
            "Algorithm"; framevisible = false, padding = (4, 4, 4, 4))
    end
    return fig
end


# -----------------------------------------------------------------------------
# V3-F2 — `recovery__<problem>`
# -----------------------------------------------------------------------------

"""
    fig_recovery(df, problem; d=5)

Mode-recovery rate vs. budget. One curve per algorithm. Skipped
gracefully (returns an empty Figure with a note) if `df` has no
`mode_recovery` column or `problem` has no modes.
"""
function fig_recovery(df::DataFrame, problem::Symbol;
                          d::Integer = HEADLINE_DIMENSION,
                          class::Symbol = :wide)
    # `class = :narrow` renders at half-column width (190 pt) for
    # side-by-side subfigure use in the thesis.
    set_pub_theme!(class = class)
    narrow = class === :narrow
    res = figure_resolution(narrow ? :narrow : :standard, :conv)
    fig = Figure(size = res)
    # V8-FIX-C2: prose title in text mode (the old math-mode string
    # rendered "M − Ridges — mode − recovery rate").
    # At half-column width the title overran the canvas and the
    # thesis subcaption already names problem and dimension, so the
    # in-figure title is dropped there.
    ax = Axis(fig[1, 1];
        xlabel = tex_label(:budget),
        # At half-column width the full label "(1.0 = all modes
        # recovered)" is taller than the axis and clips at the canvas
        # edge; the figure caption carries the explanation instead.
        # No hyphen — MathTeXEngine sets it as a math minus inside \text.
        ylabel = narrow ? L"R(\hat\pi)\;\;(\text{mode recovery rate})" :
                          SCORE_LABEL[:mode_recovery],
        xscale = log10,
        title = narrow ? "" :
            _title_latex(string(PROBLEM_LABEL[problem],
                                ", mode-recovery rate");
                         math = string("(d = ", Int(d), ")")))
    standard_axis!(ax)
    if !hasproperty(df, :mode_recovery)
        error("fig_recovery: cells.csv has no mode_recovery column")
    end
    sub = df[df.problem .== String(problem) .&& df.d .== Int(d), :]
    leg_lines = []
    leg_labels = String[]
    for alg in ALG_ORDER
        sub_alg = sub[sub.algorithm .== String(alg), :]
        keep = _filter_keep(sub_alg)
        sub_alg = sub_alg[keep, :]
        isempty(sub_alg) && continue
        grouped = combine(groupby(sub_alg, :B)) do s
            v = filter(isfinite, Vector{Float64}(s.mode_recovery))
            isempty(v) && return DataFrame(B = s.B[1], median = NaN)
            return DataFrame(B = s.B[1], median = median(v))
        end
        sort!(grouped, :B)
        xs = Vector{Float64}(grouped.B)
        ys = Vector{Float64}(grouped.median)
        ok = isfinite.(ys)
        any(ok) || continue
        ln = lines!(ax, xs[ok], ys[ok];
            color = ALG_COLOR[alg], linewidth = ALG_LINEWIDTH[alg],
            linestyle = ALG_LINESTYLE[alg])
        scatter!(ax, xs[ok], ys[ok];
            color = ALG_COLOR[alg], marker = ALG_MARKER[alg],
            markersize = 9)
        push!(leg_lines, ln)
        push!(leg_labels, ALG_LABEL[alg])
    end
    ylims!(ax, 0.0, 1.05)
    # V8-FIX-C2: ticks at the actual budgets in thesis notation instead
    # of Makie's fractional decades (10^{4.5}).
    Bs = sort!(unique(Float64.(sub.B)))
    if !isempty(Bs)
        ax.xticks = (Bs, [LaTeXString("\$" * _budget_latex(b) * "\$") for b in Bs])
        ax.xminorticksvisible = false
    end
    # At half-column width the five-entry in-axis legend covered the
    # curves, so it moves below the axis as a compact horizontal strip.
    if narrow
        isempty(leg_lines) || Legend(fig[2, 1], leg_lines, leg_labels;
            orientation = :horizontal, nbanks = 2, framevisible = false,
            labelsize = 7, patchsize = (14, 4), colgap = 8, rowgap = 0,
            padding = (0, 0, 0, 0))
    else
        isempty(leg_lines) || axislegend(ax, leg_lines, leg_labels; position = :rb)
    end
    return fig
end


# -----------------------------------------------------------------------------
# V3-F4 — `dim__<problem>__B1e5`
# -----------------------------------------------------------------------------

"""
    fig_dim_grid(df; metric=:W1_marginal_avg, B=1e5)

7-panel grid of `metric` vs `d` for every problem at the chosen B.
The grid is 2 rows × 4 columns; the eighth slot hosts a shared
legend.
"""
# Compact raw-metric y-axis label for the dim grid panels.
function _dim_metric_label(metric::Symbol)
    metric === :W1_marginal_avg && return tex_label(:W1)
    metric === :SWD && return tex_label(:SWD)
    metric === :eta_Nlike && return tex_label(:eta)
    metric === :dlogZ && return tex_label(:dlogZ)
    return LaTeXString(string(metric))
end

function fig_dim_grid(df::DataFrame;
                       metric::Symbol = :W1_marginal_avg,
                       B::Real = 1e5,
                       problems::Tuple = PROBLEM_NAMES)
    # Only problems that are actually swept over more than
    # one dimension get a panel. The old grid drew a 2-10 axis under the
    # fixed-dimension specialists (shell, spiky M-ridges, eggbox) whose
    # single column of markers carried no dimensional signal. With four
    # scaling targets the layout is a 2x2 grid plus a side legend.
    set_pub_theme!(class = :wide)
    multi_d_problems = Symbol[]
    for prob in problems
        sub_p = df[df.problem .== String(prob) .&& df.B .== Float64(B), :]
        length(unique(sub_p.d)) >= 2 && push!(multi_d_problems, prob)
    end
    isempty(multi_d_problems) && (multi_d_problems = collect(problems))
    n_p = length(multi_d_problems)
    cols = n_p <= 4 ? 2 : 4
    rows = ceil(Int, n_p / cols)
    res = figure_resolution(:wide, :dim)
    fig = Figure(size = res)
    axes = Axis[]
    leg_lines = []
    leg_labels = String[]
    for (idx, prob) in enumerate(multi_d_problems)
        r = div(idx - 1, cols) + 1
        c = mod(idx - 1, cols) + 1
        left_col = c == 1
        bottom_row = (r == rows) || (idx + cols > n_p)
        # V8-FIX-C3: the panels plot RAW metric medians on a log axis, so
        # the y-label is the plain metric symbol (lower = better) — the
        # old label claimed "−log₁₀ … (higher = closer to truth)", which
        # contradicted the plotted values. Compact label, left column only.
        ax = Axis(fig[r, c];
            xlabel = bottom_row ? tex_label(:d) : "",
            ylabel = left_col ? _dim_metric_label(metric) : "",
            yscale = log10,
            title = PROBLEM_LABEL[prob],
            xticks = ([2, 5, 10], ["2", "5", "10"]))
        standard_axis!(ax)
        xlims!(ax, 1, 11)
        push!(axes, ax)
        sub = df[df.problem .== String(prob) .&& df.B .== Float64(B), :]
        all_y = Float64[]
        for alg in ALG_ORDER
            sub_alg = sub[sub.algorithm .== String(alg), :]
            keep = _filter_keep(sub_alg)
            sub_alg = sub_alg[keep, :]
            isempty(sub_alg) && continue
            grouped = combine(groupby(sub_alg, :d)) do s
                v = filter(isfinite, Vector{Float64}(s[!, metric]))
                isempty(v) && return DataFrame(d = s.d[1], median = NaN)
                return DataFrame(d = s.d[1], median = median(v))
            end
            sort!(grouped, :d)
            xs = Vector{Float64}(grouped.d)
            ys = Vector{Float64}(grouped.median)
            ok = isfinite.(ys)
            any(ok) || continue
            ln = lines!(ax, xs[ok], ys[ok];
                color = ALG_COLOR[alg], linewidth = ALG_LINEWIDTH[alg],
                linestyle = ALG_LINESTYLE[alg])
            scatter!(ax, xs[ok], ys[ok];
                color = ALG_COLOR[alg], marker = ALG_MARKER[alg], markersize = 6)
            append!(all_y, ys[ok])
            if isempty(leg_labels) || !(ALG_LABEL[alg] in leg_labels)
                push!(leg_lines, ln)
                push!(leg_labels, ALG_LABEL[alg])
            end
        end
        _apply_log10_yticks!(ax, all_y)
    end
    if !isempty(leg_lines)
        # Side legend: keeps every grid slot for data panels.
        Legend(fig[1:rows, cols + 1], leg_lines, leg_labels, "Algorithm";
            framevisible = false)
    end
    return fig
end


# -----------------------------------------------------------------------------
# V3-F11 — `tests__heatmap` (Wilcoxon p-values, MW vs. comparators)
# -----------------------------------------------------------------------------

"""
    fig_tests_heatmap(df; metric=:W1_marginal_avg, d=5, B=5e5)

Effect-size heatmap of the seed-paired MW-vs-comparator tests. Cell colour encodes Cliff's δ oriented as a
*MoleWhacker advantage* (positive/blue = MW better, negative/red =
comparator better, diverging colormap, fixed range `[-1, 1]`); the δ
value is inscribed, and the Holm-adjusted seed-paired one-sided
Wilcoxon significance is overlaid as stars (`*` p ≤ .05, `**` ≤ .01,
`***` ≤ .001). Cells with fewer than three paired seeds render "n/a"
on grey. Each problem is tested at its own benchmark dimension
(`_heatmap_problem_dim`), so the eggbox column (d = 2) is populated —
the old strict d-filter left it entirely "n/a".
"""
function fig_tests_heatmap(df::DataFrame;
                              metric::Symbol = :W1_marginal_avg,
                              d::Integer = HEADLINE_DIMENSION,
                              B::Real = 5e5,
                              tail::Symbol = :left,
                              tests::Union{DataFrame,Nothing} = nothing)
    # When `tests` (headline_tests.csv) is provided, δ and the adjusted
    # p are taken from it verbatim, so the stars carry the SAME Holm
    # family as the numbers quoted in the thesis text (all headline
    # tests), instead of a within-figure family that is less
    # conservative. Without `tests` the figure computes its own
    # seed-paired statistics as a fallback.
    set_pub_theme!(class = :wide)
    res = figure_resolution(:wide, :heatmap)
    fig = Figure(size = res)
    comparators = [a for a in ALG_ORDER if a != :mw]
    probs = collect(HEATMAP_PROBLEM_ORDER)
    n_alg = length(comparators)
    n_prob = length(probs)
    pgrid  = fill(NaN, n_alg, n_prob)
    dgrid  = fill(NaN, n_alg, n_prob)   # Cliff's δ, oriented as MW advantage
    n_used = fill(0, n_alg, n_prob)
    raw_p = Float64[]
    raw_idx = Tuple{Int,Int}[]
    # For "lower is better" metrics MW ahead means δ(mw, cm) < 0, so the
    # sign is flipped for display; for a "higher is better" metric like
    # mode_recovery the raw δ is already the advantage.
    smaller_better = metric in (:W1_marginal_avg, :SWD, :mmd_rbf, :dlogZ)
    for (i, alg) in enumerate(comparators), (j, prob) in enumerate(probs)
        d_j = _heatmap_problem_dim(prob, d)
        if tests !== nothing
            row = tests[tests.problem    .== String(prob)  .&&
                        tests.d          .== d_j            .&&
                        tests.B          .== Float64(B)     .&&
                        tests.metric     .== String(metric) .&&
                        tests.comparator .== String(alg), :]
            isempty(row) && continue      # no admissible test -> n/a
            n_used[i, j] = Int(row.n_pairs[1])
            n_used[i, j] < 3 && continue
            delta = Float64(row.cliffs_delta[1])
            dgrid[i, j] = smaller_better ? -delta : delta
            pgrid[i, j] = Float64(row.p_adj[1])
            continue
        end
        ssp = df[df.problem .== String(prob) .&&
                 df.d       .== d_j          .&&
                 df.B       .== Float64(B), :]
        mw = ssp[ssp.algorithm .== "mw", :]
        cm = ssp[ssp.algorithm .== String(alg), :]
        # Match by seed for paired test. Flag-failed seeds
        # (RHAT-FAIL / FAILED-SANITY / BUDGET-VIOLATION) are dropped from
        # the pairing exactly as in 02b_tests.jl, so the figure agrees
        # with headline_tests.csv — the old unfiltered pairing let
        # inadmissible chains (e.g. M-ridges x NUTS) enter the test.
        common_seeds = intersect(Set(mw.seed), Set(cm.seed))
        isempty(common_seeds) && continue
        a = Float64[]
        b = Float64[]
        for s in collect(common_seeds)
            mw_row = mw[mw.seed .== s, :]
            cm_row = cm[cm.seed .== s, :]
            isempty(mw_row) && continue
            isempty(cm_row) && continue
            bad(r) = hasproperty(r, :notes) &&
                (occursin("RHAT-FAIL",        String(coalesce(r.notes[1], ""))) ||
                 occursin("FAILED-SANITY",    String(coalesce(r.notes[1], ""))) ||
                 occursin("BUDGET-VIOLATION", String(coalesce(r.notes[1], ""))))
            (bad(mw_row) || bad(cm_row)) && continue
            push!(a, Float64(mw_row[1, metric]))
            push!(b, Float64(cm_row[1, metric]))
        end
        finite = isfinite.(a) .& isfinite.(b)
        a = a[finite]
        b = b[finite]
        n_used[i, j] = length(a)
        length(a) < 3 && continue
        _, p, n = wilcoxon_signed_rank(a, b; tail = tail)
        delta = cliffs_delta(a, b)
        dgrid[i, j] = smaller_better ? -delta : delta
        pgrid[i, j] = p
        push!(raw_p, p)
        push!(raw_idx, (i, j))
    end
    # Holm-Bonferroni adjust.
    if !isempty(raw_p)
        adj = holm_bonferroni(raw_p)
        for (k, (i, j)) in enumerate(raw_idx)
            pgrid[i, j] = adj[k]
        end
    end
    # V8-FIX-C2: metric rendered through a readable plain-text name — the
    # old title interpolated the raw column name (`W1_marginal_avg`),
    # which MathTeXEngine mangled into subscript soup.
    metric_name = get(Dict(
            :W1_marginal_avg => "marginal Wasserstein-1",
            :SWD => "sliced Wasserstein",
            :mmd_rbf => "kernel MMD",
            :mode_recovery => "mode recovery"),
        metric, String(metric))
    # Column labels carry the per-problem dimension where it deviates
    # from the headline d (eggbox / d = 2), matching the summary heatmaps.
    xticktext = String[]
    for prob in probs
        d_j = _heatmap_problem_dim(prob, d)
        push!(xticktext, d_j == Int(d) ? PROBLEM_LABEL[prob] :
            string(PROBLEM_LABEL[prob], "  /  d = ", d_j))
    end
    ax = Axis(fig[1, 1];
        xlabel = "Problem",
        ylabel = "Comparator (vs. MW)",
        title = string("MoleWhacker advantage, Cliff's δ  (", metric_name, ")"))
    standard_axis!(ax)
    ax.xminorticksvisible = false
    ax.yminorticksvisible = false
    ax.xticksvisible = false
    ax.yticksvisible = false
    hm = heatmap!(ax, 1:n_prob, 1:n_alg, dgrid';
        colormap = :RdBu, colorrange = (-1, 1), nan_color = :transparent)
    Colorbar(fig[1, 2], hm;
        label = LaTeXString("\$\\text{Cliff's } \\delta \\;\\text{ (MW advantage)}\$"),
        labelsize = 9, ticklabelsize = 9)
    ax.xticks = (1:n_prob, xticktext)
    ax.yticks = (1:n_alg, [ALG_LABEL[a] for a in comparators])
    ax.xticklabelrotation = π / 4
    for i in 1:n_alg, j in 1:n_prob
        dv = dgrid[i, j]
        if isnan(dv)
            poly!(ax, [(j - 0.5, i - 0.5), (j + 0.5, i - 0.5),
                        (j + 0.5, i + 0.5), (j - 0.5, i + 0.5)];
                color = (:gray85, 0.35), strokecolor = :gray60,
                strokewidth = 0.4)
            text!(ax, j, i; text = "n/a", align = (:center, :center),
                fontsize = 8, color = :gray40)
            continue
        end
        p = pgrid[i, j]
        stars = p <= 1e-3 ? "***" : (p <= 1e-2 ? "**" : (p <= 5e-2 ? "*" : ""))
        # RdBu is light near zero and saturated at the extremes; switch
        # the inscribed text colour on |δ|.
        txtcol = abs(dv) >= 0.62 ? :white : :black
        label = @sprintf("%+.2f", dv)
        isempty(stars) || (label *= "\n" * stars)
        text!(ax, j, i; text = label, align = (:center, :center),
            fontsize = 8, color = txtcol)
    end
    # In-figure key (caption independence), two stacked lines so the
    # text fits the 396 pt canvas.
    Label(fig[2, :],
        LaTeXString("\$\\text{blue: MW better} \\qquad \\text{red: comparator better}" *
            "\\qquad \\text{gray: fewer than three admissible seed pairs}\$");
        fontsize = 8, halign = :center)
    Label(fig[3, :],
        LaTeXString("\${}^{*}\\; p_{\\mathrm{adj}} \\leq 0.05" *
            "\\qquad {}^{**}\\; p_{\\mathrm{adj}} \\leq 0.01" *
            "\\qquad {}^{***}\\; p_{\\mathrm{adj}} \\leq 0.001" *
            "\\quad\\text{(Holm adjusted, seed paired one sided Wilcoxon)}\$");
        fontsize = 8, halign = :center)
    return fig
end

# Map cells.csv metric columns to `_TEX_LABELS` keys for the tests heatmap.
function _tests_metric_texkey(metric::Symbol)
    metric === :W1_marginal_avg && return :W1
    metric === :SWD && return :SWD
    metric === :mmd_rbf && return :mmd
    metric === :mode_recovery && return :recovery
    return metric
end


# -----------------------------------------------------------------------------
# V3-CD — Friedman/Nemenyi critical-difference diagram
# -----------------------------------------------------------------------------

"""
    fig_cd_diagram(scores, alg_labels; α=0.05)

`scores` is an `n_blocks × k` matrix, lower = better. Each block
must be a (problem, d, B, seed) cell where every algorithm has a
score. Returns a Demšar-style critical-difference diagram showing
average ranks and connecting non-significantly-different groups.
"""
function fig_cd_diagram(scores::AbstractMatrix{<:Real},
                            alg_labels::AbstractVector{<:AbstractString};
                            α::Real = 0.05)
    set_pub_theme!(class = :wide)
    n, k = size(scores)
    @assert length(alg_labels) == k "fig_cd_diagram: label count mismatch"
    # Average ranks per algorithm (lower = better).
    ranks = zeros(Float64, n, k)
    @inbounds for i in 1:n
        row = collect(Float64, scores[i, :])
        finite = isfinite.(row)
        if !any(finite)
            ranks[i, :] .= NaN
            continue
        end
        perm = sortperm(row)
        rk = Vector{Float64}(undef, k)
        idx = 1
        while idx <= k
            j = idx
            while j < k && row[perm[j + 1]] == row[perm[idx]]
                j += 1
            end
            avg = (idx + j) / 2
            for s in idx:j
                rk[perm[s]] = avg
            end
            idx = j + 1
        end
        ranks[i, :] = rk
    end
    valid = .!any(isnan, ranks; dims = 2)[:]
    n_eff = sum(valid)
    avg_ranks = vec(mean(ranks[valid, :]; dims = 1))
    cd = nemenyi_critical_difference(k, n_eff; α = α)

    res = figure_resolution(:wide, :scaling)
    fig = Figure(size = res)
    ax = Axis(fig[1, 1];
        xlabel = "Average rank (lower = better)",
        title = LaTeXString(@sprintf("Friedman + Nemenyi CD = %.2f, n = %d, α = %.2f",
                                       cd, n_eff, α)))
    hidedecorations!(ax; label = false)
    hidespines!(ax)
    xlims!(ax, 0.5, k + 0.5)
    ylims!(ax, 0, 1)

    # Top axis with rank ticks.
    rank_ticks = collect(1:k)
    for r in rank_ticks
        lines!(ax, [r, r], [0.78, 0.82]; color = :black, linewidth = 1)
        text!(ax, r, 0.85; text = string(r), align = (:center, :bottom),
            fontsize = 9)
    end
    lines!(ax, [1, k], [0.80, 0.80]; color = :black, linewidth = 1)

    # Place algorithm labels on left and right of the rank axis,
    # alternating bottom positions.
    perm = sortperm(avg_ranks)
    half = ceil(Int, k / 2)
    # Left half (lowest ranks) listed on the left side, right half on
    # the right. Each one has a connector line down from the rank.
    for (i, idx) in enumerate(perm)
        side = i <= half ? :left : :right
        x_anchor = side === :left ? 0.7 : k + 0.3
        y_anchor = 0.6 - 0.07 * (side === :left ? i : (i - half))
        r = avg_ranks[idx]
        # Connector segments.
        lines!(ax, [r, r], [0.78, y_anchor + 0.02];
            color = :black, linewidth = 0.7)
        lines!(ax, [r, x_anchor], [y_anchor + 0.02, y_anchor + 0.02];
            color = :black, linewidth = 0.7)
        text!(ax, x_anchor, y_anchor + 0.02;
            text = LaTeXString(@sprintf("%s  (%.2f)", alg_labels[idx], r)),
            align = (side === :left ? :right : :left, :center),
            fontsize = 9)
    end

    # Draw CD-bar connecting non-significantly-different groups.
    sorted_ranks = sort(avg_ranks)
    groups = Vector{Tuple{Float64,Float64}}()
    i = 1
    while i <= k
        j = i
        while j < k && sorted_ranks[j + 1] - sorted_ranks[i] <= cd
            j += 1
        end
        if j > i
            push!(groups, (sorted_ranks[i], sorted_ranks[j]))
        end
        i = j + 1
    end
    for (g, (lo, hi)) in enumerate(groups)
        y = 0.92 + 0.012 * g
        lines!(ax, [lo, hi], [y, y]; color = :black, linewidth = 2.4)
    end

    # CD reference bar (top right).
    lines!(ax, [k - cd, k], [0.97, 0.97]; color = :black, linewidth = 1.4)
    text!(ax, k - cd / 2, 0.985;
        text = LaTeXString(@sprintf("CD = %.2f", cd)),
        align = (:center, :bottom), fontsize = 9)

    return fig
end


# =============================================================================
# LaTeX tables (§10.16.8)
# =============================================================================

function write_summary_table(df::DataFrame, metric_col::Symbol, path::AbstractString;
                                d::Integer = HEADLINE_DIMENSION, B::Real = 5e5)
    # V5 §2: rows are problems (canonical heatmap order, eggbox at the
    # bottom always at d=2), columns are algorithms. dlogZ × {MH, NUTS}
    # cells render as `--` (chain methods do not produce a log-evidence
    # estimate; documented in the chapter caption).
    n_alg  = length(ALG_ORDER)
    n_prob = length(HEATMAP_PROBLEM_ORDER)
    chain_algs = (:mh, :nuts)
    table   = fill(NaN, n_prob, n_alg)
    blocked = falses(n_prob, n_alg)
    for (i, prob) in enumerate(HEATMAP_PROBLEM_ORDER),
        (j, alg) in enumerate(ALG_ORDER)
        d_i = _heatmap_problem_dim(prob, d)
        if metric_col === :dlogZ && alg in chain_algs
            blocked[i, j] = true
            continue
        end
        ss = df[df.problem   .== String(prob) .&&
                df.algorithm .== String(alg)  .&&
                df.d         .== d_i          .&&
                df.B         .== Float64(B), :]
        keep = trues(nrow(ss))
        if hasproperty(ss, :notes)
            for (k, n) in enumerate(ss.notes)
                ns = String(coalesce(n, ""))
                if occursin("RHAT-FAIL", ns) || occursin("BUDGET-VIOLATION", ns) ||
                   occursin("FAILED-SANITY", ns)
                    keep[k] = false
                end
            end
        end
        v = filter(!isnan, Vector{Float64}(ss[keep, metric_col]))
        table[i, j] = isempty(v) ? NaN : median(v)
    end
    isdir(dirname(path)) || mkpath(dirname(path))
    open(path, "w") do io
        println(io, "% LaTeX table — $(metric_col) at d=$(Int(d)), B=$(_budget_token(B))")
        println(io, "\\begin{tabular}{l$(repeat("r", n_alg))}")
        println(io, "\\toprule")
        println(io, "Problem & " *
            join([ALG_LABEL[a] for a in ALG_ORDER], " & ") * " \\\\")
        println(io, "\\midrule")
        for (i, prob) in enumerate(HEATMAP_PROBLEM_ORDER)
            d_i = _heatmap_problem_dim(prob, d)
            label = d_i == Int(d) ? PROBLEM_LABEL[prob] :
                string(PROBLEM_LABEL[prob], " (\$d=", d_i, "\$)")
            # V5 §2: insert a midrule above the eggbox sub-block to mark
            # the d-block boundary, matching the thin horizontal rule in
            # the heatmap figure.
            if prob === :eggbox && i > 1
                println(io, "\\midrule")
            end
            row = label
            for j in 1:n_alg
                row *= " & "
                if blocked[i, j]
                    row *= "--"
                else
                    v = table[i, j]
                    row *= isnan(v) ? "n/a" : @sprintf("%.3g", v)
                end
            end
            println(io, row * " \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
    return path
end

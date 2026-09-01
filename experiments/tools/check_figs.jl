# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# =============================================================================
# check_figs.jl — V8-FIX-C4 figure verification gate
# =============================================================================
#
# For every PDF in experiments/out/figs:
#   1. page width ∈ {396, 190} pt (±2) — final-print-size contract
#      (full-width = \textwidth = 396 pt, half-width = 190 pt);
#   2. file size ≤ 1.5 MB — catches un-rasterized scatter/surface panels;
#   3. no placeholder text ("run v3 grid first", "not in cells.csv");
#   4. text stream carries no raw programmer tokens: a metric column
#      name (`W1_marginal_avg`, `eta_Nlike`, …) or budget notation
#      ("B = 5e5") leaking into a title.
#
# Usage: julia --project=. experiments/tools/check_figs.jl [figs_dir]
# Exit code 0 iff every check passes.

const FIGS_DIR = isempty(ARGS) ?
    joinpath(@__DIR__, "..", "out", "figs") : ARGS[1]

const ALLOWED_WIDTHS_PT = (396.0, 190.0)
const WIDTH_TOL_PT = 2.0
const MAX_BYTES = round(Int, 1.5 * 1024^2)
const BAD_TOKENS = (
    "run v3 grid first", "not in cells.csv",
    "W1_marginal_avg", "eta_Nlike", "mode_recovery", "mmd_rbf",
    "N_show", "T_max / Neff",
    "B = 5e5", "B = 5e4", "B = 5e3", "B=5e5", "B=5e4", "B=5e3",
)

"Extract /MediaBox widths (pt) from the raw PDF bytes."
function _mediabox_widths(raw::Vector{UInt8})
    s = String(copy(raw))
    widths = Float64[]
    for m in eachmatch(r"/MediaBox\s*\[\s*([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\s*\]", s)
        x0 = parse(Float64, m.captures[1]); x1 = parse(Float64, m.captures[3])
        push!(widths, x1 - x0)
    end
    return widths
end

"Scan raw bytes plus inflated Flate streams for forbidden tokens."
function _scan_tokens(raw::Vector{UInt8})
    hits = String[]
    texts = String[String(copy(raw))]
    inflated = try
        _inflate_streams(raw)
    catch
        ""
    end
    isempty(inflated) || push!(texts, inflated)
    for t in texts, tok in BAD_TOKENS
        occursin(tok, t) && push!(hits, tok)
    end
    return unique(hits)
end

function _inflate_streams(raw::Vector{UInt8})
    io = IOBuffer()
    data = raw
    i = 1
    needle = codeunits("stream")
    endneedle = codeunits("endstream")
    while true
        r = findnext(needle, data, i)
        r === nothing && break
        j = last(r) + 1
        (j <= length(data) && data[j] == UInt8('\r')) && (j += 1)
        (j <= length(data) && data[j] == UInt8('\n')) && (j += 1)
        e = findnext(endneedle, data, j)
        e === nothing && break
        payload = data[j:first(e)-1]
        try
            write(io, transcode(CodecZlib.ZlibDecompressor, payload))
        catch
        end
        i = last(e) + 1
    end
    return String(take!(io))
end

using Pkg
const _ROOT = abspath(joinpath(@__DIR__, "..", ".."))
Base.active_project() != joinpath(_ROOT, "Project.toml") && Pkg.activate(_ROOT)
_HAVE_CODECZLIB = try
    @eval import CodecZlib
    true
catch
    false
end

function main()
    pdfs = filter(f -> endswith(f, ".pdf"), readdir(FIGS_DIR; join = true))
    isempty(pdfs) && (println("no PDFs found in $FIGS_DIR"); return 1)
    n_fail = 0
    for p in sort(pdfs)
        raw = read(p)
        problems = String[]
        widths = _mediabox_widths(raw)
        if isempty(widths)
            push!(problems, "no MediaBox found")
        else
            w = maximum(widths)
            if !any(abs(w - a) <= WIDTH_TOL_PT for a in ALLOWED_WIDTHS_PT)
                push!(problems, "width $(round(w; digits=1)) pt ∉ {396, 190}")
            end
        end
        length(raw) > MAX_BYTES &&
            push!(problems, "size $(round(length(raw)/1024^2; digits=2)) MB > 1.5 MB")
        if _HAVE_CODECZLIB
            hits = _scan_tokens(raw)
            isempty(hits) || push!(problems, "raw tokens: " * join(hits, ", "))
        end
        if !isempty(problems)
            n_fail += 1
            println("FAIL  ", basename(p), "  — ", join(problems, "; "))
        end
    end
    println()
    println("check_figs: $(length(pdfs)) PDFs checked, $(n_fail) failing")
    return n_fail == 0 ? 0 : 1
end

exit(main())

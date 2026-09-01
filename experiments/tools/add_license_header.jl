#!/usr/bin/env julia
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
# =============================================================================
# add_license_header.jl — V5 §5, attribution reconciled in V6 §P1
# =============================================================================
#
# Walk every `.jl` file under `experiments/{src,scripts,tools,tests}` and
#   1. prepend a short MIT/SPDX header if the file does not already carry one;
#   2. reconcile the copyright author in any existing header (V6 P1:
#      `Valentin Reindel` → `Valentin Reindel`).
# Idempotent: re-running is a no-op.
#
# Usage:
#   julia --project=. experiments/tools/add_license_header.jl
#
# By default the script writes in-place. Pass `--dry-run` to list the files
# that would be modified without changing them.

const AUTHOR = "Valentin Reindel"
const OLD_AUTHORS = ("Valentin Reindel",)   # superseded names to reconcile

const HEADER_LINES = String[
    "# SPDX-License-Identifier: MIT",
    "# Copyright (c) 2026 $(AUTHOR) (MoleWhacker thesis).",
    "# See `LICENSE` at the repository root for the full MIT license text.",
]
const HEADER_BLOCK = join(HEADER_LINES, "\n") * "\n"

function _has_header(path::AbstractString)
    isfile(path) || return false
    open(path) do io
        head = read(io, 1024)
        return occursin("SPDX-License-Identifier", String(head))
    end
end

function _prepend_header(path::AbstractString; dry::Bool = false)
    body = read(path, String)
    new_body = HEADER_BLOCK * body
    dry && return false
    open(path, "w") do io
        write(io, new_body)
    end
    return true
end

function main(args::Vector{String} = ARGS)
    dry = any(==(  "--dry-run" ), args)
    root = abspath(joinpath(@__DIR__, "..", ".."))
    targets = String[]
    for sub in ("experiments/src", "experiments/scripts",
                    "experiments/tools", "experiments/tests")
        d = joinpath(root, sub)
        isdir(d) || continue
        for (dir, _, files) in walkdir(d)
            for f in files
                endswith(f, ".jl") || continue
                push!(targets, joinpath(dir, f))
            end
        end
    end
    n_skipped = 0
    n_patched = 0
    n_reattributed = 0
    for f in targets
        if _has_header(f)
            n_skipped += 1
        else
            _prepend_header(f; dry = dry)
            n_patched += 1
            println(dry ? "would patch: " : "patched: ",
                    relpath(f, root))
        end
        # V6 P1: reconcile a superseded author name in existing headers.
        body = read(f, String)
        new_body = body
        for old in OLD_AUTHORS
            new_body = replace(new_body, old => AUTHOR)
        end
        if new_body != body
            n_reattributed += 1
            println(dry ? "would re-attribute: " : "re-attributed: ",
                    relpath(f, root))
            dry || open(f, "w") do io
                write(io, new_body)
            end
        end
    end
    println("\nSummary: ", n_patched, " files ",
            dry ? "to be patched" : "patched", "; ",
            n_reattributed, " re-attributed; ",
            n_skipped, " already had a header.")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end

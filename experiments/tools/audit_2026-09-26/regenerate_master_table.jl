# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valentin Reindel (MoleWhacker thesis).
# See `LICENSE` at the repository root for the full MIT license text.
#!/usr/bin/env julia
# =============================================================================
# regenerate_master_table.jl — audit wrapper of 2026-09-26.
#
# Runs the unchanged generator experiments/tools/table_master_results.jl (the
# caption of which was corrected on 2026-09-26, review finding V5-B06) with its
# single output path redirected to experiments/out/audit_2026-09-26/, so that
# the archived experiments/out/tables/master_results_table.tex is not touched.
# Inputs are the archived experiments/out/tables/cells.csv and
# experiments/out/revision_2026-09-25/partition_mass.csv, read only. No
# sampler runs. The regenerated file is then copied by hand into the thesis
# (appendices/B-master-table.tex) after a line-by-line diff confirming that
# only the caption line changed.
#
# Usage (bench repository root):
#   julia --project=. experiments/tools/audit_2026-09-26/regenerate_master_table.jl
# =============================================================================

const GEN = normpath(joinpath(@__DIR__, "..", "table_master_results.jl"))
const AUDIT_OUT = normpath(joinpath(@__DIR__, "..", "..", "out", "audit_2026-09-26"))
mkpath(AUDIT_OUT)

src = read(GEN, String)
const OLD_LINE = "out_tex = joinpath(OUT, \"tables\", \"master_results_table.tex\")"
const NEW_LINE = "out_tex = joinpath(OUT, \"audit_2026-09-26\", \"master_results_table.tex\")"
occursin(OLD_LINE, src) || error("output line of the generator not found; update this wrapper")
src = replace(src, OLD_LINE => NEW_LINE)

t0 = time()
# Passing GEN as the file name makes @__DIR__/@__FILE__ inside the generator
# resolve to experiments/tools/, so its relative include of 03_plots.jl works;
# its `abspath(PROGRAM_FILE) == @__FILE__` guard is false here, so main() is
# called explicitly below.
include_string(Main, src, GEN)
t1 = time()
rc = main()
t2 = time()
println("regenerate_master_table.jl: load $(round(t1 - t0, digits = 1)) s, generate $(round(t2 - t1, digits = 1)) s, exit code $rc")
println("output: ", joinpath(AUDIT_OUT, "master_results_table.tex"))
println("julia ", VERSION)

# =============================================================================
# run_all.ps1 — PowerShell wrapper around 04_run_all.jl (Windows-friendly)
# =============================================================================
# Usage from the MoleWhacker repository root:
#
#     pwsh experiments/scripts/run_all.ps1                 # full headline run
#     pwsh experiments/scripts/run_all.ps1 --scaling       # + scaling subset
#     pwsh experiments/scripts/run_all.ps1 --dryrun        # plan only
#
# Env overrides:
#     JULIA_BIN       override julia executable (default: julia)
#     JULIA_THREADS   override -t (default: auto)
# =============================================================================

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = 'Stop'

$JuliaBin     = $env:JULIA_BIN
if (-not $JuliaBin) { $JuliaBin = 'julia' }

$JuliaThreads = $env:JULIA_THREADS
if (-not $JuliaThreads) { $JuliaThreads = 'auto' }

$ScriptDir = Split-Path -Parent $PSCommandPath
$RootDir   = Resolve-Path (Join-Path $ScriptDir '..' '..')
$Script    = Join-Path $RootDir 'experiments\scripts\04_run_all.jl'

Set-Location $RootDir

Write-Host "[run_all.ps1] julia=$JuliaBin  threads=$JuliaThreads  project=." -ForegroundColor Cyan
Write-Host "[run_all.ps1] script=$Script" -ForegroundColor Cyan
Write-Host "[run_all.ps1] args=$($Args -join ' ')" -ForegroundColor Cyan

& $JuliaBin --project=. -t $JuliaThreads $Script @Args
exit $LASTEXITCODE

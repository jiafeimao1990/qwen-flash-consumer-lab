param(
    [Parameter(Mandatory = $true)][string]$ExllamaRoot
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent
$patch = Join-Path $repoRoot "patches\exllamav3-v1.5.0-consumer-moe-lab.patch"
$ExllamaRoot = (Resolve-Path -LiteralPath $ExllamaRoot).Path

$commit = git -C $ExllamaRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw "Not a Git checkout: $ExllamaRoot" }
if ($commit -ne "0740edc2da569fb99174023c1d2988b1e98cb41e") {
    throw "Expected ExLlamaV3 v1.5.0 commit 0740edc..., found $commit"
}

git -C $ExllamaRoot apply --check $patch
if ($LASTEXITCODE -ne 0) { throw "Patch preflight failed; the checkout may be modified" }
git -C $ExllamaRoot apply $patch
if ($LASTEXITCODE -ne 0) { throw "Patch application failed" }
Write-Host "Applied consumer MoE lab patch to $ExllamaRoot" -ForegroundColor Green

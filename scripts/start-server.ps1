param(
    [Parameter(Mandatory = $true)][string]$TabbyApiRoot,
    [Parameter(Mandatory = $true)][string]$ExllamaRoot,
    [Parameter(Mandatory = $true)][string]$ModelRoot,
    [string]$ModelName = "Qwen3.8-Flash-Next-EXL3-4.05bpw",
    [int]$Port = 5002,
    [int]$ContextTokens = 98304,
    [int]$CpuExperts = 364,
    [int]$CpuThreads = 16,
    [double[]]$GpuSplit = @(14.5, 14.5),
    [int]$ReasoningBudget = 1024,
    [string]$StatsPath = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent
$TabbyApiRoot = (Resolve-Path -LiteralPath $TabbyApiRoot).Path
$ExllamaRoot = (Resolve-Path -LiteralPath $ExllamaRoot).Path
$ModelRoot = (Resolve-Path -LiteralPath $ModelRoot).Path

$python = Join-Path $TabbyApiRoot ".venv\Scripts\python.exe"
$main = Join-Path $TabbyApiRoot "main.py"
$templatePath = Join-Path $repoRoot "configs\tabbyapi-96k-thinking.yml.template"
if (!(Test-Path -LiteralPath $python)) { throw "TabbyAPI venv Python not found: $python" }
if (!(Test-Path -LiteralPath $main)) { throw "TabbyAPI main.py not found: $main" }
if (!(Test-Path -LiteralPath (Join-Path $ExllamaRoot "exllamav3"))) {
    throw "ExLlamaV3 package directory not found under: $ExllamaRoot"
}
if (!(Test-Path -LiteralPath (Join-Path $ModelRoot $ModelName))) {
    throw "Model directory not found: $(Join-Path $ModelRoot $ModelName)"
}

if ([string]::IsNullOrWhiteSpace($StatsPath)) {
    $StatsPath = Join-Path $repoRoot "profiles\mixed-placement-identity.json"
}
$StatsPath = (Resolve-Path -LiteralPath $StatsPath).Path

function YamlPath([string]$value) {
    $escaped = ($value -replace '\\', '/') -replace '"', '\"'
    return '"' + $escaped + '"'
}

$runtime = [IO.File]::ReadAllText($templatePath)
$replacements = [ordered]@{
    "__PORT__" = $Port
    "__MODEL_ROOT__" = (YamlPath $ModelRoot)
    "__MODEL_NAME__" = (YamlPath $ModelName)
    "__CONTEXT_TOKENS__" = $ContextTokens
    "__GPU_SPLIT__" = (($GpuSplit | ForEach-Object { $_.ToString([Globalization.CultureInfo]::InvariantCulture) }) -join ', ')
    "__CPU_EXPERTS__" = $CpuExperts
    "__CPU_THREADS__" = $CpuThreads
    "__REASONING_BUDGET__" = $ReasoningBudget
}
foreach ($pair in $replacements.GetEnumerator()) {
    $runtime = $runtime.Replace([string]$pair.Key, [string]$pair.Value)
}

$runtimeConfig = Join-Path $env:TEMP "qwen-flash-consumer-lab-$PID.yml"
[IO.File]::WriteAllText($runtimeConfig, $runtime, [Text.UTF8Encoding]::new($false))

if ($DryRun) {
    try {
        Write-Output $runtime
    }
    finally {
        Remove-Item -LiteralPath $runtimeConfig -Force -ErrorAction SilentlyContinue
    }
    return
}

$env:PYTHONPATH = $ExllamaRoot
$env:EXL3_MOE_CPU_SPLIT_STATS = $StatsPath
$env:EXL3_MOE_CPU_SWAP = "1"
$env:EXL3_MOE_CPU_SWAP_PROFILE = "1"
$env:EXL3_MOE_CPU_SWAP_INTERVAL = "128"
$env:EXL3_MOE_CPU_SWAP_MAX = "96"
$env:EXL3_MOE_CPU_SWAP_PER_LAYER = "2"
$env:EXL3_INT8_GEMV = "0"
Remove-Item Env:EXL3_MOE_COOP_WIDE -ErrorAction SilentlyContinue

try {
    Push-Location $TabbyApiRoot
    & $python $main --config $runtimeConfig
    if ($LASTEXITCODE -ne 0) { throw "TabbyAPI exited with code $LASTEXITCODE" }
}
finally {
    Pop-Location -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $runtimeConfig -Force -ErrorAction SilentlyContinue
}

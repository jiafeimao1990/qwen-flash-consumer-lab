$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent
$patterns = @(
    'C:\\Users\\',
    'D:\\AI\\',
    'gho_[A-Za-z0-9_]+',
    'hf_[A-Za-z0-9]+'
)
$files = Get-ChildItem -LiteralPath $repoRoot -File -Recurse | Where-Object {
    $_.FullName -notmatch '[\\/]\.git[\\/]' -and
    $_.FullName -notmatch '[\\/]__pycache__[\\/]' -and
    $_.Extension -notin @('.pyc', '.pyo')
}
$hits = @()
foreach ($file in $files) {
    foreach ($pattern in $patterns) {
        $match = Select-String -LiteralPath $file.FullName -Pattern $pattern -ErrorAction SilentlyContinue
        if ($match) { $hits += $match }
    }
}
if ($hits.Count) {
    $hits | ForEach-Object { "{0}:{1}: {2}" -f $_.Path, $_.LineNumber, $_.Line.Trim() }
    throw "Public-tree check failed with $($hits.Count) path/credential hit(s)."
}
Write-Host "Public-tree check passed." -ForegroundColor Green

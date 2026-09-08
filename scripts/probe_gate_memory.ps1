<# Isolated WPR capture. No model execution, no service changes, no global cache flush. #>
param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$gateIdentity=[Security.Principal.WindowsIdentity]::GetCurrent()
if (-not ([Security.Principal.WindowsPrincipal]::new($gateIdentity)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Administrator token required' }
$gateRoot=[IO.Path]::GetFullPath($Root)
$gateCheckout=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
if ($gateRoot.StartsWith($gateCheckout.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Evidence must be outside checkout'}
if (Test-Path -LiteralPath $gateRoot) {throw 'New probe directory required'}
New-Item -ItemType Directory -Path $gateRoot | Out-Null
$gateStatus=& wpr.exe -status 2>&1 | Out-String
$gateStatus | Out-File (Join-Path $gateRoot 'before.txt')
if (-not $gateStatus.Contains('WPR is not recording')) {throw 'Existing or unknown WPR session; leave it untouched'}
$gateProfile=Join-Path $PSScriptRoot 'gate_memory.wprp'
$gateOwned=$false
try {
    & wpr.exe -start "$gateProfile!GateMemory" -filemode 2>&1 | Out-File (Join-Path $gateRoot 'start.txt')
    if ($LASTEXITCODE -ne 0) {throw 'WPR start failed'}
    $gateOwned=$true
    Push-Location ([IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')))
    try {
        $env:CLOUDRAG_SYNTHETIC_FILE=Join-Path $gateRoot 'synthetic-uncached.bin'
        & '.venv-app/Scripts/python.exe' -c "import os,json; from scripts.gate_memory import synthetic_hard_faults; print(json.dumps(synthetic_hard_faults(os.environ['CLOUDRAG_SYNTHETIC_FILE'])))" | Out-File (Join-Path $gateRoot 'work.json')
        if ($LASTEXITCODE -ne 0) {throw 'Synthetic work failed'}
    } finally {Pop-Location}
} finally {
    if ($gateOwned) {
        & wpr.exe -stop (Join-Path $gateRoot 'trace.etl') 2>&1 | Out-File (Join-Path $gateRoot 'stop.txt')
        if ($LASTEXITCODE -ne 0) {throw 'Owned WPR stop failed; inspect session'}
    }
}
& tracerpt.exe (Join-Path $gateRoot 'trace.etl') -of XML -o (Join-Path $gateRoot 'trace.xml') -summary (Join-Path $gateRoot 'summary.txt') 2>&1 | Out-File (Join-Path $gateRoot 'decode.txt')
if ($LASTEXITCODE -ne 0) {throw 'ETW decode failed'}
@{at=[DateTime]::UtcNow.ToString('o');profile_sha256=(Get-FileHash -LiteralPath $gateProfile -Algorithm SHA256).Hash;script_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash} | ConvertTo-Json | Out-File (Join-Path $gateRoot 'identity.json')

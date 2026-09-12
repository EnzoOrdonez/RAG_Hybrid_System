<# Human entry point. DryRun never invokes the privileged manager or models. #>
param([string]$Resume, [switch]$AuthorizeNewWindow, [switch]$DryRun, [string]$Output)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $project
$python = Join-Path $project '.venv-app/Scripts/python.exe'
$env:PYTHONUTF8='1'
$env:PYTHONHASHSEED='42'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:CUDA_VISIBLE_DEVICES=''
if ($DryRun -and ($Resume -or $AuthorizeNewWindow)) { throw 'DryRun cannot open/resume a real window' }
if ($Resume -and $Output) { throw 'Use Resume or Output, not both' }
$root = if ($Resume) { [IO.Path]::GetFullPath($Resume) } elseif ($Output) { [IO.Path]::GetFullPath($Output) } else {
    Join-Path 'C:/CloudRAG' (('diag-{0}-' -f $(if($DryRun){'synthetic'}else{'run'})) + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ'))
}
if ($DryRun) {
    & $python scripts/unattended_diagnostic.py dry-run --root $root
    exit $LASTEXITCODE
}
$admin = ([Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw 'Open PowerShell as Administrator and run the same command. Nothing changed.' }
$key=[BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($root.ToLowerInvariant()))).Replace('-','')
$lock=$null
try {
    $lock=[IO.File]::Open((Join-Path ([IO.Path]::GetTempPath()) ('CloudRAG-launch-'+$key+'.lock')),'OpenOrCreate','ReadWrite','None')
$argsPrepare = @('scripts/unattended_diagnostic.py','prepare','--root',$root)
if ($Resume) { $argsPrepare += '--resume' }
if ($AuthorizeNewWindow) { $argsPrepare += '--authorize-new-window' }
# Native stderr is diagnostic output, not a PowerShell exception on Windows 5.1.
$ErrorActionPreference='Continue'
$outputLines = @(& $python @argsPrepare 2>&1)
$prepareExit=$LASTEXITCODE
$ErrorActionPreference='Stop'
if ($prepareExit) { $outputLines | ForEach-Object { Write-Host $_ }; exit $prepareExit }
$window = [string]$outputLines[-1]
if ($window -eq 'COMPLETE') {
    Write-Host "40 posiciones consumidas; sin nueva intervencion. Revise $root/summary.json; NO-GO vigente."
    $done=Get-Content -LiteralPath (Join-Path $root 'summary.json') -Raw | ConvertFrom-Json
    if (-not $done.confirmation_ready -or $done.cleanup_pending.Count) { exit 2 }
    exit 0
}
if (-not $window.StartsWith($root.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid prepared window path' }
Write-Host "CloudRAG | paquete: $root | ventana: $window | limite 120 minutos"
try {
    $proof=Join-Path $window 'proof'
    & $PSScriptRoot/manage_gate_window.ps1 -Mode SelfTest -LexicalDiagnostic -Unattended -Root (Join-Path $proof 'deadline')
    & $PSScriptRoot/manage_gate_window.ps1 -Mode SelfTestController -LexicalDiagnostic -Unattended -Root (Join-Path $proof 'controller')
    foreach($kind in @('deadline','controller')) {
        if (-not (Test-Path -LiteralPath (Join-Path $proof "$kind/selftest-passed.json"))) { throw "Supervisor proof failed: $kind. No cut." }
    }
    & $PSScriptRoot/manage_gate_window.ps1 -Mode Run -LexicalDiagnostic -Unattended -KeepAnyDesk -Root $window -Cohort (Join-Path $root 'cohort') -SupervisorProof $proof
} catch {
    Write-Host "FALLO: $($_.Exception.Message) | conserve $root"
    $failure=$_.Exception.Message
} finally {
    if (Test-Path -LiteralPath (Join-Path $window 'window.json')) {
        & $PSScriptRoot/manage_gate_window.ps1 -Mode Restore -Root $window
    }
    & $python scripts/unattended_diagnostic.py package --root $root
    if ($LASTEXITCODE) { $failure='Paquete incompleto: conserve la carpeta y revise restauracion.'; Write-Host $failure }
}
if (Get-Variable failure -ErrorAction SilentlyContinue) { exit 1 }
$summary=Get-Content -LiteralPath (Join-Path $root 'summary.json') -Raw | ConvertFrom-Json
Write-Host "Fin: $root/summary.json | confirmation_ready=$($summary.confirmation_ready) | NO-GO vigente"
if (-not $summary.confirmation_ready -or $summary.cleanup_pending.Count) { exit 2 }
} finally { if ($lock) { $lock.Dispose() } }

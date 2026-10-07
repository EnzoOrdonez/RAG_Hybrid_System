param([Parameter(ValueFromRemainingArguments=$true)][string[]]$OperatorArguments)
$ErrorActionPreference='Stop'
$identity=[Security.Principal.WindowsIdentity]::GetCurrent()
$principal=New-Object Security.Principal.WindowsPrincipal($identity)
if ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Abre PowerShell sin administrador para usar este operador.' }
$settings=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'installation.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$previousPath=$env:PYTHONPATH
$previousBytecode=$env:PYTHONDONTWRITEBYTECODE
$previousUtf8=$env:PYTHONUTF8
$previousLocation=Get-Location
try {
    $env:PYTHONPATH=Join-Path $PSScriptRoot 'bundle'
    $env:PYTHONDONTWRITEBYTECODE='1'
    $env:PYTHONUTF8='1'
    Set-Location -LiteralPath $PSScriptRoot
    & $settings.python -B -m scripts.study_operator.cli --root $PSScriptRoot @OperatorArguments
    $operatorExit=$LASTEXITCODE
} finally {
    Set-Location -LiteralPath $previousLocation.Path
    $env:PYTHONPATH=$previousPath
    $env:PYTHONDONTWRITEBYTECODE=$previousBytecode
    $env:PYTHONUTF8=$previousUtf8
}
exit $operatorExit

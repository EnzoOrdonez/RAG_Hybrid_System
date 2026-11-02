# Lanzador del cierre por inventario I7 (auditoría 2026-11-02). Toda la lógica está en closeout.py,
# probada con un gcloud falso en tests/test_review_2026_11_02_closeout.py.
# ESTE .ps1 NO SE EJECUTÓ en la auditoría (no había PowerShell): revísalo antes del primer uso.
#
# Solo lectura (por defecto):
#   .\closeout.ps1 -Sdk $sdk -Project <id> -Root $run -Out C:\CloudRAG\closeout-AAAAMMDD
# Borrado (pide una palabra distinta por recurso):
#   .\closeout.ps1 -Sdk $sdk -Project <id> -Root $run -Out C:\CloudRAG\closeout-AAAAMMDD -Delete -Inventory C:\CloudRAG\closeout-AAAAMMDD\inventory.json
param(
    [Parameter(Mandatory = $true)][string]$Sdk,
    [Parameter(Mandatory = $true)][string]$Project,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$Root,
    [switch]$Delete,
    [string]$Inventory,
    [switch]$IncludeRetained,
    [string]$Python = (Join-Path $PSScriptRoot '..\..\..\..\.venv-app\Scripts\python.exe')
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python)) { throw "No existe el intérprete: $Python (usa -Python)" }
$arguments = @((Join-Path $PSScriptRoot 'closeout.py'), '--sdk', $Sdk, '--project', $Project, '--out', $Out)
if ($Root) { $arguments += @('--root', $Root) }
if ($Delete) {
    if (-not $Inventory) { throw '-Delete requiere -Inventory de una lectura previa' }
    $arguments += @('--delete', '--inventory', $Inventory)
}
if ($IncludeRetained) { $arguments += '--include-retained' }
& $Python @arguments
exit $LASTEXITCODE

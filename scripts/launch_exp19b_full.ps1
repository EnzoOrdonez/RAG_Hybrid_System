<#
    exp19b — lanzador para la terminal de Enzo.

    Seccion de Claude Code — 2026-08-21 14:15 (hora local).

    La corrida son ~6,6 h de generacion y TIENEN que caer dentro de una sola sesion de Ollama
    ya calentada: medido el 2026-08-21, reiniciar el servidor cambia la respuesta a un prompt
    byte-identico (q001 vs checkpoint, jaccard-5grama 0,0705). Un brazo partido entre dos
    estados del generador produce una diferencia que se leeria como efecto del selector.

    Este script hace el preflight y delega TODA la orquestacion en
    scripts/run_exp19b_pipeline.py, que es la parte con tests.

    Uso:  pwsh -File scripts\launch_exp19b_full.ps1
          pwsh -File scripts\launch_exp19b_full.ps1 -DryRun
    Exit: 0 todo paso · 2 una etapa fallo · 3 RUNTIME_STATE_CHANGED (no se puntuo nada)
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [int]$MaxQueries = 0,
    [string]$Python = "C:\Users\enziz\AppData\Local\Python\pythoncore-3.14-64\python.exe"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$env:PYTHONUTF8 = 1          # receta determinista en cualquier consola (REPRODUCE.md §0)
$env:HF_HUB_OFFLINE = 1
$env:TRANSFORMERS_OFFLINE = 1
$env:PYTHONHASHSEED = 42

if (-not (Test-Path $Python)) { Write-Error "Interprete no encontrado: $Python"; exit 2 }

# ---------------------------------------------------------------- preflight: Ollama
$ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
function Get-OllamaVersion {
    try { (Invoke-WebRequest -Uri "http://localhost:11434/api/version" -TimeoutSec 4).Content }
    catch { $null }
}

$ver = Get-OllamaVersion
if (-not $ver) {
    if (-not (Test-Path $ollama)) { Write-Error "Ollama no encontrado en $ollama"; exit 2 }
    Write-Host "Ollama no responde; levantando el servidor..." -ForegroundColor Yellow
    Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
    foreach ($i in 1..20) {
        Start-Sleep -Seconds 3
        $ver = Get-OllamaVersion
        if ($ver) { break }
    }
}
if (-not $ver) { Write-Error "Ollama sigue sin responder en localhost:11434"; exit 2 }
Write-Host "Ollama OK: $ver"

$tags = (Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 10).Content | ConvertFrom-Json
if (-not ($tags.models.name -contains "granite4.1:8b")) {
    Write-Error "granite4.1:8b no esta en Ollama. Es el generador de TODA la evidencia de la fase."
    exit 2
}
Write-Host "granite4.1:8b OK"

# ---------------------------------------------------------------- el aviso que hay que leer
Write-Host ""
Write-Host ("=" * 72) -ForegroundColor Cyan
Write-Host " ANTES DE SEGUIR — la corrida son ~6,6 h y no admite interrupcion" -ForegroundColor Cyan
Write-Host ("=" * 72) -ForegroundColor Cyan
Write-Host "  1. Portatil ENCHUFADO a la corriente."
Write-Host "  2. Suspension e hibernacion DESACTIVADAS:"
Write-Host "       powercfg /change standby-timeout-ac 0"
Write-Host "       powercfg /change hibernate-timeout-ac 0"
Write-Host "  3. NO reinicies Ollama ni cierres esta terminal hasta que termine TODO."
Write-Host "     Un reinicio a mitad invalida el pareado: el script lo detecta antes de"
Write-Host "     regenerar y aborta con RUNTIME_STATE_CHANGED sin puntuar nada."
Write-Host "  4. El log queda en logs\exp19b_full_<fecha>.log"
Write-Host ("=" * 72) -ForegroundColor Cyan
Write-Host ""

$argv = @("$repo\scripts\run_exp19b_pipeline.py")
if ($DryRun)        { $argv += "--dry-run" }
if ($MaxQueries -gt 0) { $argv += @("--max-queries", "$MaxQueries") }

& $Python @argv
$code = $LASTEXITCODE

switch ($code) {
    0 { Write-Host "exp19b: pipeline completo." -ForegroundColor Green }
    3 { Write-Host "exp19b: RUNTIME_STATE_CHANGED — no se puntuo nada. Relanza el pipeline entero." -ForegroundColor Red }
    default { Write-Host "exp19b: una etapa fallo (exit $code). Revisa el log." -ForegroundColor Red }
}
exit $code

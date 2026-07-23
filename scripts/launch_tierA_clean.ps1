# Tier A overnight launcher (summer phase) — run AFTER a clean reboot.
#
# Why: granite4.1:8b @ num_ctx 4096 needs ~5.4 GB fully-on-GPU; with the normal
# desktop stack (~1.7 GB VRAM baseline) Ollama splits CPU/GPU and generation is
# NOT deterministic (cold-vs-warm prompt-cache divergence; ledger entrada 3).
# On a clean boot (~0.4 GB baseline) the model fits 100 % GPU like the June
# exp12 runs, generation is bit-deterministic and ~2-4x faster.
#
# Protocol (Enzo):
#   1. Reboot. Do NOT open browsers/apps. Open one PowerShell.
#   2. cd C:\Users\enziz\projects\hybrid-rag-system
#   3. powershell -ExecutionPolicy Bypass -File scripts\launch_tierA_clean.ps1
#   4. Leave the machine alone (screen can lock; do not sleep/hibernate).
# The run is checkpointed: re-running this script resumes where it stopped.
# Ollama defaults are kept (no KV quantization) = June-canonical conditions.

$ErrorActionPreference = "Stop"
$repo = "C:\Users\enziz\projects\hybrid-rag-system"
$py = "C:\Users\enziz\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$log = "$repo\output\audit\tierA_run_$(Get-Date -Format 'yyyy-MM-dd_HHmm')"

$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"; $env:PYTHONHASHSEED = "42"

# keep the tray app out of the way; plain `ollama serve` with defaults
Get-Process | Where-Object { $_.Name -like "*ollama*" } |
    Stop-Process -Force -Confirm:$false -ErrorAction SilentlyContinue
Start-Sleep 3
Start-Process -FilePath "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList "serve" `
    -RedirectStandardOutput "$log.ollama.out.log" -RedirectStandardError "$log.ollama.err.log" `
    -WindowStyle Hidden
Start-Sleep 6

# preload + placement check: refuse to start if the model is not 100% GPU
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" run granite4.1:8b "say OK" --keepalive 60m | Out-Null
$ps = & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" ps | Out-String
Write-Host $ps
if ($ps -notmatch "100% GPU") {
    Write-Host "ABORT: granite is NOT 100% GPU (split detected). Free VRAM (clean boot?) and retry." -ForegroundColor Red
    exit 1
}

# Pass G — 5 arms x 60 queries, determinism probe per arm, checkpoint/resume
& $py "$repo\scripts\run_exp15_ablation.py" --exp-id exp15_ablation_tierA --pass G `
    *>> "$log.passG.log"
if ($LASTEXITCODE -ne 0) { Write-Host "Pass G FAILED - see $log.passG.log" -ForegroundColor Red; exit 1 }

# Pass N — NLI scoring, both verifiers (GPU free of Ollama pressure by now is
# irrelevant: Pass N runs after generation completes)
& $py "$repo\scripts\run_exp15_ablation.py" --exp-id exp15_ablation_tierA --pass N --verifier small `
    *>> "$log.passN_small.log"
& $py "$repo\scripts\run_exp15_ablation.py" --exp-id exp15_ablation_tierA --pass N --verifier base `
    *>> "$log.passN_base.log"

Write-Host "Tier A COMPLETE. Logs: $log.*" -ForegroundColor Green

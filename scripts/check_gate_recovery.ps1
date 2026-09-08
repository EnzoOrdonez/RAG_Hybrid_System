<# Verify independent SYSTEM task execution. Never stops or configures a service. #>
param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Write-NewJson([string]$Path, $Value) {
    $bytes = [Text.Encoding]::UTF8.GetBytes(($Value | ConvertTo-Json -Depth 8))
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    try { $stream.Write($bytes, 0, $bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
}

$rootFull = [IO.Path]::GetFullPath($Root)
$checkout = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
if ($rootFull.StartsWith($checkout.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase) -or $rootFull -eq $checkout) {
    throw 'Evidence must be outside checkout'
}
[IO.Directory]::CreateDirectory($rootFull) | Out-Null
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$administrator = ([Security.Principal.WindowsPrincipal]::new($identity)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
Write-NewJson (Join-Path $rootFull 'invocation.json') @{
    at=[DateTime]::UtcNow.ToString('o'); pid=$PID; administrator=$administrator
    script_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
}
if (-not $administrator) { throw 'Administrator token required; no task or service changed' }

$taskName = 'CloudRAG-RecoveryProbe-' + [guid]::NewGuid().ToString('N')
$marker = Join-Path $rootFull 'system-marker.json'
$escapedMarker = $marker.Replace("'", "''")
$actionCode = @"
`$ErrorActionPreference='Stop'
`$value=@{at=[DateTime]::UtcNow.ToString('o');pid=`$PID;sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value}
`$bytes=[Text.Encoding]::UTF8.GetBytes((`$value | ConvertTo-Json))
`$stream=[IO.File]::Open('$escapedMarker',[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try {`$stream.Write(`$bytes,0,`$bytes.Length);`$stream.Flush(`$true)} finally {`$stream.Dispose()}
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($actionCode))
$created = $false
try {
    $action = New-ScheduledTaskAction -Execute "$env:SystemRoot/System32/WindowsPowerShell/v1.0/powershell.exe" -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand $encoded"
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 2)
    Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -User 'SYSTEM' -RunLevel Highest | Out-Null
    $created = $true
    Start-ScheduledTask -TaskName $taskName
    $deadline = [DateTime]::UtcNow.AddSeconds(60)
    while (-not (Test-Path -LiteralPath $marker) -and [DateTime]::UtcNow -lt $deadline) { Start-Sleep -Milliseconds 250 }
    if (-not (Test-Path -LiteralPath $marker)) { throw 'Independent task did not publish its marker within 60 seconds' }
    $result = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
    if ($result.sid -ne 'S-1-5-18') { throw 'Independent task did not execute as SYSTEM' }
    Write-NewJson (Join-Path $rootFull 'result.json') @{at=[DateTime]::UtcNow.ToString('o');passed=$true;task=$taskName;marker=$result}
} catch {
    Write-NewJson (Join-Path $rootFull 'failure.json') @{at=[DateTime]::UtcNow.ToString('o');error=$_.Exception.Message;task=$taskName}
    throw
} finally {
    if ($created) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-NewJson (Join-Path $rootFull 'cleanup.json') @{at=[DateTime]::UtcNow.ToString('o');removed_task=$taskName}
    }
}

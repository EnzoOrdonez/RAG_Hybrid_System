<# Bounded reversible window. Run elevated only after SelfTest and code audit pass. #>
param(
    [ValidateSet('Run','Watch','Restore','SelfTest','SelfTestController','NoticeTest')][string]$Mode,
    [Parameter(Mandatory=$true)][string]$Root,
    [string]$Cohort,
    [ValidateSet('hybrid','lexical','semantic')][string]$System,
    [ValidateSet('cold','warm')][string]$Phase,
    [switch]$KeepAnyDesk
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$rootFull = [IO.Path]::GetFullPath($Root)
$checkout = [IO.Path]::GetFullPath((Join-Path $project '../..'))
if ($rootFull -eq $checkout -or $rootFull.StartsWith($checkout.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Use external evidence root' }
$admin = ([Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw 'Administrator token required; nothing changed' }

function Save-New([string]$Path, $Value) {
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Path)) | Out-Null
    $bytes = [Text.Encoding]::UTF8.GetBytes(($Value | ConvertTo-Json -Depth 12))
    $stream = [IO.File]::Open($Path, 'CreateNew', 'Write', 'Read')
    try { $stream.Write($bytes, 0, $bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
}
function Event($Kind, $Data) {
    Save-New (Join-Path $rootFull ('events/' + [guid]::NewGuid().ToString('N') + '.json')) @{
        at=[DateTime]::UtcNow.ToString('o'); kind=$Kind; data=$Data; pid=$PID
    }
}
function Read-Json($Name) { Get-Content -LiteralPath (Join-Path $rootFull $Name) -Raw | ConvertFrom-Json }
function Identity($Process) { @{pid=$Process.Id; creation_filetime=$Process.StartTime.ToUniversalTime().ToFileTimeUtc()} }
function Same-Process($Identity) {
    $candidate = Get-Process -Id $Identity.pid -ErrorAction SilentlyContinue
    return ($null -ne $candidate -and $candidate.StartTime.ToUniversalTime().ToFileTimeUtc() -eq $Identity.creation_filetime)
}
function Interactive-Snapshot($Processes, $SessionId) {
    @($Processes | Where-Object { $_.session -eq $SessionId -and $_.name -in @('AnyDesk','EpicGamesLauncher') } |
        ForEach-Object { [pscustomobject]@{path=$_.path} } | Sort-Object path -Unique)
}
function Show-Notice([string]$Message, [int]$Seconds=10) {
    if ((Get-Process -Id $PID).SessionId -eq 0) { throw 'Notice requires an interactive session' }
    $shell = New-Object -ComObject WScript.Shell
    try {
        $code = $shell.Popup($Message, $Seconds, 'CloudRAG - aviso tecnico', 4160)
        if ($code -notin @(-1,1)) { throw 'Notice failed or returned an unexpected result' }
        return @{mechanism='WScript.Shell.Popup'; return_code=$code; seconds=$Seconds; read_by_user='unverified'}
    } finally { [Runtime.InteropServices.Marshal]::FinalReleaseComObject($shell) | Out-Null }
}
function Restore-Service($Before, [bool]$Simulated) {
    if ($Before.name -notin @('AnyDesk','NvContainerLocalSystem')) { throw 'Service outside allowlist' }
    Event 'restore-service-intent' $Before
    if (-not $Simulated) {
        $start = @{Auto='Automatic'; Manual='Manual'; Disabled='Disabled'}[$Before.start_mode]
        if (-not $start) { throw 'Unknown original service start mode' }
        Set-Service -Name $Before.name -StartupType $start
        if ($Before.state -eq 'Running') {
            Start-Service -Name $Before.name
            (Get-Service -Name $Before.name).WaitForStatus('Running', [TimeSpan]::FromSeconds(30))
        } else {
            Stop-Service -Name $Before.name
        }
        $after = Get-CimInstance Win32_Service -Filter "Name='$($Before.name)'"
        if ($after.State -ne $Before.state -or $after.StartMode -ne $Before.start_mode) { throw 'Service restoration mismatch' }
        # Set-Service must not silently change the delayed auto-start flag.
        $reg = Get-ItemProperty -LiteralPath ('HKLM:/SYSTEM/CurrentControlSet/Services/' + $Before.name)
        $delayed = if ($reg.PSObject.Properties['DelayedAutoStart']) { $reg.DelayedAutoStart } else { $null }
        if ($delayed -ne $Before.delayed_auto_start) { throw 'Delayed auto-start changed unexpectedly' }
    }
    Event 'restore-service-verified' @{name=$Before.name; simulated=$Simulated}
}
function Restore-Window {
    $window = Read-Json 'window.json'
    if (Test-Path -LiteralPath (Join-Path $rootFull 'restored.json')) { return }
    $lock = $null
    try { $lock = [IO.File]::Open((Join-Path $rootFull 'restore.lock'), 'OpenOrCreate', 'ReadWrite', 'None') } catch { return }
    try {
        if (Test-Path -LiteralPath (Join-Path $rootFull 'restored.json')) { return }
        # Remote access is always the first restoration action, even if worker cleanup fails.
        if (-not $window.PSObject.Properties['manage_anydesk'] -or $window.manage_anydesk) {
            Restore-Service ($window.services | Where-Object name -eq 'AnyDesk') $window.simulated
        }
        $failures = [Collections.Generic.List[string]]::new()
        try {
            if (Test-Path -LiteralPath (Join-Path $rootFull 'worker-identity.json')) {
                $worker = Read-Json 'worker-identity.json'
                if (Same-Process $worker) { Stop-Process -Id $worker.pid -Force; Event 'worker-stopped' $worker }
            }
        } catch { $failures.Add($_.Exception.Message) }
        if (-not $window.simulated) {
            # Stop only named traces belonging to this window; never global wpr -cancel.
            $traceRoots = @($rootFull)
            if ($window.PSObject.Properties['cohort']) { $traceRoots += $window.cohort }
            foreach ($file in @(Get-ChildItem -LiteralPath $traceRoots -Filter trace-identity.json -Recurse)) {
                if (Test-Path -LiteralPath (Join-Path $file.DirectoryName 'trace-stopped.json')) { continue }
                try {
                    $trace = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
                    if (-not $file.FullName.StartsWith($rootFull.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
                        if (-not $trace.PSObject.Properties['window_id'] -or $trace.window_id -ne $window.id) { continue }
                    }
                    if ($trace.instance -notmatch '^CloudRAG-[a-f0-9]{32}$') { throw 'Unrecognized trace identity' }
                    $destination = Join-Path $file.DirectoryName ('recovered-' + [guid]::NewGuid().ToString('N') + '.etl')
                    $output = & wpr -stop $destination -skipPdbGen -instancename $trace.instance 2>&1
                    Event 'trace-recovery' @{instance=$trace.instance; exit=$LASTEXITCODE; output=@($output | ForEach-Object { "$_" })}
                    # A failed start has no active recorder; retain the command result for audit.
                } catch { $failures.Add($_.Exception.Message) }
            }
        }
        try { Restore-Service ($window.services | Where-Object name -eq 'NvContainerLocalSystem') $window.simulated } catch { $failures.Add($_.Exception.Message) }
        foreach ($task in $window.tasks) {
            try {
                if ($task.name -notlike 'NVIDIA App SelfUpdate_*' -or $task.path -ne '\') { throw 'Task outside allowlist' }
                if (-not $window.simulated -and $task.enabled) { Enable-ScheduledTask -TaskName $task.name -TaskPath $task.path | Out-Null }
                if (-not $window.simulated -and (Get-ScheduledTask -TaskName $task.name -TaskPath $task.path).Settings.Enabled -ne $task.enabled) { throw 'Task restore mismatch' }
                Event 'task-restored' $task
            } catch { $failures.Add($_.Exception.Message) }
        }
        foreach ($launch in $window.interactive) {
            try {
                if ($window.simulated) { continue }
                if ([IO.Path]::GetFileName($launch.path) -notin @('AnyDesk.exe','EpicGamesLauncher.exe','EpicWebHelper.exe')) { throw 'Interactive app outside allowlist' }
                if ([IO.Path]::GetFileName($launch.path) -eq 'EpicWebHelper.exe') { continue } # launcher owns helpers
                $present = @(Get-Process | Where-Object { $_.SessionId -eq $window.session_id -and $_.Path -eq $launch.path })
                if ($present.Count) { continue }
                $name = $window.task_name + '-Interactive-' + [guid]::NewGuid().ToString('N')
                $principal = New-ScheduledTaskPrincipal -UserId $window.user -LogonType Interactive -RunLevel Limited
                $action = New-ScheduledTaskAction -Execute $launch.path
                Register-ScheduledTask -TaskName $name -Action $action -Principal $principal | Out-Null
                try {
                    Start-ScheduledTask -TaskName $name
                    Start-Sleep -Seconds 5
                    $present = @(Get-Process | Where-Object { $_.SessionId -eq $window.session_id -and $_.Path -eq $launch.path })
                    if (-not $present.Count) { throw ('Interactive restoration unverified: ' + $launch.path) }
                    Event 'interactive-restored' @{path=$launch.path; pids=@($present.Id)}
                } finally { Unregister-ScheduledTask -TaskName $name -Confirm:$false }
            } catch { $failures.Add($_.Exception.Message) }
        }
        if ($failures.Count) { Event 'restore-incomplete' @($failures); throw ($failures -join '; ') }
        Save-New (Join-Path $rootFull 'restored.json') @{at=[DateTime]::UtcNow.ToString('o'); simulated=$window.simulated; anydesk_first=$true}
        if (Get-ScheduledTask -TaskName $window.task_name -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $window.task_name -Confirm:$false
            Event 'watchdog-unregistered' $window.task_name
        }
    } finally { $lock.Dispose() }
}
function Watch-Window {
    $window = Read-Json 'window.json'
    Event 'watchdog-ready' @{sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value}
    while (-not (Test-Path -LiteralPath (Join-Path $rootFull 'restored.json'))) {
        $expired = [DateTime]::UtcNow -ge [DateTime]::Parse($window.deadline_utc).ToUniversalTime()
        $beat = Get-Item -LiteralPath (Join-Path $rootFull 'heartbeat') -ErrorAction SilentlyContinue
        $stale = $null -eq $beat -or ([DateTime]::UtcNow - $beat.LastWriteTimeUtc).TotalSeconds -gt 60
        if ($expired -or $stale -or -not (Same-Process $window.controller)) {
            Event 'watchdog-trigger' @{expired=$expired; stale=$stale; controller_alive=(Same-Process $window.controller)}
            try { Restore-Window } catch { Event 'watchdog-restore-error' $_.Exception.Message }
        }
        Start-Sleep -Seconds 2
    }
}
if ($Mode -eq 'NoticeTest') {
    $result = Show-Notice 'CloudRAG: prueba del aviso visible. NO se cerrara AnyDesk ni se ejecutaran modelos en esta prueba.' 5
    Save-New (Join-Path $rootFull 'notice-test.json') @{at=[DateTime]::UtcNow.ToString('o'); result=$result}
    exit
}
if ($Mode -eq 'Restore') { Restore-Window; exit }
if ($Mode -eq 'Watch') { Watch-Window; exit }
if (Test-Path -LiteralPath (Join-Path $rootFull 'window.json')) { throw 'Window already exists; use Restore, never replay Run' }
if ($Mode -eq 'Run' -and (-not $Cohort -or -not $System -or -not $Phase)) { throw 'Run requires Cohort, System and Phase' }
if ($KeepAnyDesk -and $Mode -ne 'Run') { throw 'KeepAnyDesk applies only to Run' }
if ($Mode -eq 'Run') {
    $cohortFull = [IO.Path]::GetFullPath($Cohort)
    if ($cohortFull -eq $checkout -or $cohortFull.StartsWith($checkout.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Use external cohort root' }
    $cohortProtocol = (Get-Content -LiteralPath (Join-Path $cohortFull 'source-manifest.json') -Raw | ConvertFrom-Json).protocol
    if ($cohortProtocol.protocol_version -ne 2 -or $cohortProtocol.systems.Count -ne 3 -or $System -notin $cohortProtocol.systems) { throw 'Unregistered bounded cohort' }
    if ($cohortProtocol.build_id -ne (& git -C $project rev-parse HEAD)) { throw 'Cohort build differs before process changes' }
}
[IO.Directory]::CreateDirectory($rootFull) | Out-Null
$simulated = $Mode -in @('SelfTest','SelfTestController')
if (-not $simulated) {
    if ((& git -C $project branch --show-current) -ne 'fix/interview-readiness') { throw 'Wrong branch' }
    if (@(& git -C $project status --porcelain).Count) { throw 'Commit audited code before measurement' }
    $version = Invoke-RestMethod http://localhost:11434/api/version -TimeoutSec 10
    $tags = Invoke-RestMethod http://localhost:11434/api/tags -TimeoutSec 10
    $model = @($tags.models | Where-Object name -eq 'granite4.1:8b')
    if ($version.version -ne '0.22.1' -or $model.Count -ne 1 -or $model[0].digest -ne '444af1c4b2fedd6b54041aca558e7300b0b3d5c0468c44619126240323ba2852') { throw 'Ollama version/digest mismatch; stop and ask' }
    Save-New (Join-Path $rootFull 'ollama-identity.json') @{at=[DateTime]::UtcNow.ToString('o'); version=$version; models=$tags}
}
$services = foreach ($name in @('AnyDesk','NvContainerLocalSystem')) {
    $service = Get-CimInstance Win32_Service -Filter "Name='$name'"
    if (-not $service) { throw ('Missing authorized service: ' + $name) }
    $reg = Get-ItemProperty -LiteralPath ('HKLM:/SYSTEM/CurrentControlSet/Services/' + $name)
    @{name=$name; state=$service.State; start_mode=$service.StartMode; pid=$service.ProcessId;
      delayed_auto_start= $(if ($reg.PSObject.Properties['DelayedAutoStart']) { $reg.DelayedAutoStart } else { $null })}
}
$tasks = @(Get-ScheduledTask | Where-Object { $_.TaskName -like 'NVIDIA App SelfUpdate_*' -and $_.TaskPath -eq '\' } | ForEach-Object {
    @{name=$_.TaskName; path=$_.TaskPath; enabled=$_.Settings.Enabled}
})
$sessionId = (Get-Process -Id $PID).SessionId
$processes = @(Get-Process | Where-Object { $_.ProcessName -in @('AnyDesk','NVIDIA Overlay','EpicGamesLauncher','EpicWebHelper') } | ForEach-Object {
    @{name=$_.ProcessName; path=$_.Path; identity=(Identity $_); session=$_.SessionId}
})
if ($KeepAnyDesk) { $processes = @($processes | Where-Object name -ne 'AnyDesk') }
$interactive = @(Interactive-Snapshot $processes $sessionId)
$windowId = [guid]::NewGuid().ToString('N')
$deadline = [DateTime]::UtcNow.AddMinutes(120)
if ($simulated) { $deadline = [DateTime]::UtcNow.AddSeconds(15) }
$controller = Get-Process -Id $PID
if ($Mode -eq 'SelfTestController') {
    $controller = Start-Process powershell.exe -ArgumentList @('-NoProfile','-Command','Start-Sleep -Seconds 60') -WindowStyle Hidden -PassThru
    $deadline = [DateTime]::UtcNow.AddSeconds(90)
}
$window = @{
    id=$windowId; at=[DateTime]::UtcNow.ToString('o'); deadline_utc=$deadline.ToString('o'); simulated=$simulated
    controller=(Identity $controller); services=@($services); tasks=$tasks; processes=$processes
    interactive=$interactive; session_id=$sessionId; user=[Security.Principal.WindowsIdentity]::GetCurrent().Name
    task_name=('CloudRAG-Gate-' + $windowId); build_id=(& git -C $project rev-parse HEAD)
    model_digest='444af1c4b2fedd6b54041aca558e7300b0b3d5c0468c44619126240323ba2852'
    trusted_manifest='C:/CloudRAG/operational-20260905T1428Z/deployment-manifest.json'
    script_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    manage_anydesk=(-not $KeepAnyDesk)
}
if ($Mode -eq 'Run') { $window.cohort=$cohortFull; $window.system=$System; $window.phase=$Phase }
Save-New (Join-Path $rootFull 'window.json') $window
[IO.File]::WriteAllText((Join-Path $rootFull 'heartbeat'), [DateTime]::UtcNow.ToString('o'))
$safeScript = $PSCommandPath.Replace("'", "''")
$safeRoot = $rootFull.Replace("'", "''")
$watchCode = @"
`$ErrorActionPreference='Stop'
try { & '$safeScript' -Mode Watch -Root '$safeRoot' } catch {
    `$errorPath=Join-Path '$safeRoot' ('watchdog-start-error-' + [guid]::NewGuid().ToString('N') + '.json')
    @{at=[DateTime]::UtcNow.ToString('o');error=`$_.ToString();detail=`$_.ScriptStackTrace;policy=(Get-ExecutionPolicy)} | ConvertTo-Json | Out-File -LiteralPath `$errorPath -Encoding utf8
    exit 1
}
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($watchCode))
# Only this trusted local task's process policy changes; no registry/global policy modification.
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot/System32/WindowsPowerShell/v1.0/powershell.exe" -Argument "-NoProfile -NonInteractive -ExecutionPolicy RemoteSigned -WindowStyle Hidden -EncodedCommand $encoded"
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $window.task_name -Action $action -Settings $settings -Trigger (New-ScheduledTaskTrigger -AtStartup) -User SYSTEM -RunLevel Highest | Out-Null
try {
    Start-ScheduledTask -TaskName $window.task_name
    $ackDeadline = [DateTime]::UtcNow.AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 250
        $ready = @(Get-ChildItem -LiteralPath (Join-Path $rootFull 'events') -Filter '*.json' -ErrorAction SilentlyContinue | ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json } | Where-Object kind -eq 'watchdog-ready')
    } while (-not $ready.Count -and [DateTime]::UtcNow -lt $ackDeadline)
    if (-not $ready.Count -or $ready[0].data.sid -ne 'S-1-5-18') { throw 'Watchdog did not acknowledge SYSTEM ownership' }
    if ($simulated) {
        if ($Mode -eq 'SelfTestController') { Stop-Process -Id $controller.Id -Force; Event 'synthetic-controller-stopped' $window.controller }
        # Expired deadline exercises the same independent restoration path; no real service mutation.
        while (-not (Test-Path -LiteralPath (Join-Path $rootFull 'restored.json')) -and [DateTime]::UtcNow -lt $deadline.AddSeconds(30)) { Start-Sleep -Milliseconds 250 }
        if (-not (Test-Path -LiteralPath (Join-Path $rootFull 'restored.json'))) { throw 'SelfTest restoration missing' }
        $events = @(Get-ChildItem -LiteralPath (Join-Path $rootFull 'events') -Filter '*.json' | ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json })
        $trigger = @($events | Where-Object kind -eq 'watchdog-trigger')
        $restoration = @($events | Where-Object kind -eq 'restore-service-intent' | Sort-Object at)
        if (-not $trigger.Count -or $restoration[0].data.name -ne 'AnyDesk' -or $restoration[0].pid -eq $PID) { throw 'Independent restoration/order not demonstrated' }
        Save-New (Join-Path $rootFull 'selftest-passed.json') @{at=[DateTime]::UtcNow.ToString('o'); mode=$Mode; trigger=$trigger; simulated_services=$true}
        exit
    }
    $notice = 'CloudRAG: prueba tecnica autorizada. AnyDesk se desconectara. Restauracion automatica antes de ' + $deadline.ToLocalTime().ToString('yyyy-MM-dd HH:mm:ss zzz') + '. Estado: ' + $rootFull
    if ($KeepAnyDesk) { $notice = 'CloudRAG: prueba tecnica autorizada; AnyDesk permanece activo. Restauracion de NVIDIA/launchers antes de ' + $deadline.ToLocalTime().ToString('yyyy-MM-dd HH:mm:ss zzz') + '. Estado: ' + $rootFull }
    Save-New (Join-Path $rootFull 'notice.json') @{at=[DateTime]::UtcNow.ToString('o'); text=$notice; deadline_utc=$deadline.ToString('o')}
    $noticeResult = Show-Notice $notice 10
    Event 'notice-delivered' @{session=$sessionId; result=$noticeResult}
    Start-Sleep -Seconds 5
    Save-New (Join-Path $rootFull 'armed.json') @{at=[DateTime]::UtcNow.ToString('o'); deadline_utc=$deadline.ToString('o')}
    foreach ($task in $tasks) {
        Event 'disable-task-intent' $task
        if ($task.enabled) { Disable-ScheduledTask -TaskName $task.name -TaskPath $task.path | Out-Null }
    }
    foreach ($service in $services) {
        if ($KeepAnyDesk -and $service.name -eq 'AnyDesk') { continue }
        Event 'stop-service-intent' $service
        Set-Service -Name $service.name -StartupType Disabled
        Stop-Service -Name $service.name
        (Get-Service -Name $service.name).WaitForStatus('Stopped', [TimeSpan]::FromSeconds(30))
        Event 'service-stopped' $service
    }
    foreach ($process in $processes) {
        if (Same-Process $process.identity) { Event 'stop-process-intent' $process; Stop-Process -Id $process.identity.pid -Force; Event 'process-stopped' $process }
    }
    Event 'process-window-ready' @{deadline_utc=$deadline.ToString('o'); remote_cut=(-not $KeepAnyDesk)}
    $python = Join-Path $project '.venv-app/Scripts/python.exe'
    $payload = Start-Process -FilePath $python -ArgumentList @('scripts/run_managed_gate.py','--root',('"' + $rootFull + '"')) -WorkingDirectory $project -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $rootFull 'payload.stdout.log') -RedirectStandardError (Join-Path $rootFull 'payload.stderr.log')
    while (-not $payload.HasExited -and [DateTime]::UtcNow -lt $deadline) {
        [IO.File]::WriteAllText((Join-Path $rootFull 'heartbeat'), [DateTime]::UtcNow.ToString('o'))
        if (Test-Path -LiteralPath (Join-Path $rootFull 'restored.json')) { throw 'Window restored by watchdog' }
        Start-Sleep -Seconds 2
        $payload.Refresh()
    }
    Event 'payload-ended' @{has_exited=$payload.HasExited; deadline_reached=([DateTime]::UtcNow -ge $deadline)}
} catch { Event 'controller-error' $_.Exception.Message; throw } finally {
    Restore-Window
}

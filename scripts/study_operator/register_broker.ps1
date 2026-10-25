param([Parameter(Mandatory=$true)][string]$Package,[Parameter(Mandatory=$true)][string]$App)
$ErrorActionPreference='Stop'
$identity=[Security.Principal.WindowsIdentity]::GetCurrent()
$caller=New-Object Security.Principal.WindowsPrincipal($identity)
if($caller.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)){throw 'Broker registration requires Limited caller'}
$name='CloudRAG-I5-LimitedTaskBroker'
if(Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue){throw 'Broker already exists; inspect instead of replacing'}
$state=Get-Content -LiteralPath (Join-Path $Package 'STATE.json') -Raw -Encoding UTF8|ConvertFrom-Json
$remaining=[DateTimeOffset]::Parse($state.closure_reserved_utc)-[DateTimeOffset]::UtcNow
if($remaining.TotalMinutes -lt 5 -or $state.status -ne 'ACTIVE'){throw 'Broker admission already closed'}
$principal=New-ScheduledTaskPrincipal -UserId $identity.Name -LogonType Interactive -RunLevel Limited
$action=New-ScheduledTaskAction -Execute (Join-Path $App '.venv-app/Scripts/python.exe') -Argument ('-B -m scripts.study_operator.task_broker --package "'+$Package+'" --app "'+$App+'"') -WorkingDirectory $App
$trigger=New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(15) -RepetitionInterval (New-TimeSpan -Minutes 1) -RepetitionDuration $remaining
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 2) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$task=New-ScheduledTask -Principal $principal -Action $action -Trigger $trigger -Settings $settings
$task|Export-ScheduledTask|Set-Content -LiteralPath (Join-Path $Package 'limited-task-broker.xml') -Encoding UTF8
Add-Content -LiteralPath (Join-Path $Package 'SYSTEM_CHANGES.md') -Encoding UTF8 -Value ('BEFORE own broker '+$name+'; Limited registering caller and worker; one-minute recurrence ending at closure reserve; each tick native2min; retire broker and child tasks at closure. No user applications touched.')
Register-ScheduledTask -TaskName $name -InputObject $task|Out-Null
$observed=Get-ScheduledTask -TaskName $name
if($observed.Principal.RunLevel -ne 'Limited'){throw 'Broker principal differs'}
@{task=$name;registering_token_limited=$true;run_level='Limited';native_tick_minutes=2;recurrence_minutes=1;closure_reserved_utc=$state.closure_reserved_utc;at=[DateTimeOffset]::UtcNow.ToString('o')}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $Package 'limited-task-broker-registration.json') -Encoding UTF8
Start-ScheduledTask -TaskName $name
$name

param(
    [Parameter(Mandatory=$true)][string]$Package,
    [Parameter(Mandatory=$true)][string]$App,
    [Parameter(Mandatory=$true)][string]$Sdk
)
$ErrorActionPreference='Stop'
$state=Get-Content -LiteralPath (Join-Path $Package 'STATE.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$reserve=[DateTimeOffset]::Parse($state.closure_reserved_utc).ToUniversalTime()
$deadline=[DateTimeOffset]::Parse($state.deadline_utc).ToUniversalTime().AddHours(-1)
if($reserve -le [DateTimeOffset]::UtcNow -or $deadline -le $reserve){throw 'Closure dates invalid or elapsed; no safety registration'}
$python=Join-Path $App '.venv-app/Scripts/python.exe'
$principal=New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$receipts=@()
foreach($item in @(@{suffix='reserve';at=$reserve},@{suffix='deadline-backup';at=$deadline})){
    $name='CloudRAG-I5-safety-'+$item.suffix
    if(Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue){throw 'Safety task already exists; inspect instead of replacing'}
    $when=$item.at.AddSeconds(1).LocalDateTime
    $action=New-ScheduledTaskAction -Execute $python -Argument ('-B -m scripts.study_operator.cloud_safety --package "'+$Package+'" --sdk "'+$Sdk+'"') -WorkingDirectory $App
    $trigger=New-ScheduledTaskTrigger -Once -At $when
    $settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 45) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -StartWhenAvailable
    $task=New-ScheduledTask -Principal $principal -Action $action -Trigger $trigger -Settings $settings
    $task | Export-ScheduledTask | Set-Content -LiteralPath (Join-Path $Package ($name+'.xml')) -Encoding UTF8
    Add-Content -LiteralPath (Join-Path $Package 'SYSTEM_CHANGES.md') -Encoding UTF8 -Value ('BEFORE own Limited safety '+$name+'; due UTC '+$item.at.ToString('o')+'; stop owned VM IDs/confirmed own intents only; native45min. No local user application changes.')
    Register-ScheduledTask -TaskName $name -InputObject $task | Out-Null
    $observed=Get-ScheduledTask -TaskName $name
    $actual=[DateTimeOffset]::Parse($observed.Triggers[0].StartBoundary).ToUniversalTime()
    if($observed.Principal.RunLevel -ne 'Limited' -or [Math]::Abs(($actual-$item.at).TotalSeconds) -gt 2){throw 'Safety trigger verification failed'}
    $receipts+=@{task=$name;run_level='Limited';trigger_utc=$actual.ToString('o');native_limit_minutes=45;registered_utc=[DateTimeOffset]::UtcNow.ToString('o')}
}
$receipts | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Package 'independent-closure-registration.json') -Encoding UTF8
$receipts | ConvertTo-Json -Depth 5

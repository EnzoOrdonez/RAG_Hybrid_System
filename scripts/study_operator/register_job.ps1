param([Parameter(Mandatory=$true)][string]$Plan,[int]$Minutes=60)
$ErrorActionPreference='Stop'
$data=Get-Content -LiteralPath $Plan -Raw -Encoding UTF8 | ConvertFrom-Json
if($data.label -notmatch '^[a-zA-Z0-9_-]+$'){throw 'Unsafe job name'}
if($Minutes -lt 1 -or $Minutes -gt 180){throw 'Native job limit must be 1..180 minutes'}
$name='CloudRAG-I5-'+$data.label
if(Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue){throw 'Task already exists; inspect before replay'}
$python=Join-Path $data.app '.venv-app/Scripts/python.exe'
$principal=New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$action=New-ScheduledTaskAction -Execute $python -Argument ('-B -m scripts.study_operator.run_control --plan "'+$Plan+'"') -WorkingDirectory $data.app
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes $Minutes) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
$task=New-ScheduledTask -Principal $principal -Action $action -Settings $settings
$task | Export-ScheduledTask | Set-Content -LiteralPath (Join-Path $data.root ($data.label+'-task.xml')) -Encoding UTF8
Add-Content -LiteralPath (Join-Path $data.root 'SYSTEM_CHANGES.md') -Encoding UTF8 -Value ('BEFORE own Limited task '+$name+'; scope '+$Plan+'; native limit '+$Minutes+' min; preserve XML and retire at closure. No user applications affected.')
Register-ScheduledTask -TaskName $name -InputObject $task | Out-Null
Start-ScheduledTask -TaskName $name
[pscustomobject]@{task=$name;run_level='Limited';native_limit_minutes=$Minutes;at=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $data.root ($data.label+'-task-receipt.json')) -Encoding UTF8
$name

param([Parameter(Mandatory=$true)][string]$Plan)
$ErrorActionPreference='Stop'
$identity=[Security.Principal.WindowsIdentity]::GetCurrent()
if(([Security.Principal.WindowsPrincipal]::new($identity)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)){throw 'Limited token required'}
$data=Get-Content -LiteralPath $Plan -Raw -Encoding UTF8 | ConvertFrom-Json
$package=[IO.Path]::GetFullPath($data.package)
$external=[IO.Path]::GetFullPath($data.external)
if([IO.Path]::GetDirectoryName($external) -ne [IO.Path]::GetDirectoryName($package) -or [IO.Path]::GetFileName($external) -notlike 'iteration5-finalization-*'){throw 'Own external sibling required'}
$statePath=Join-Path $package 'STATE.json'
$state=Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
if($state.status -ne 'ACTIVE' -or (Test-Path -LiteralPath (Join-Path $package 'MANIFEST_SHA256.jsonl'))){throw 'Only active unsealed run may register finalization'}
$reserve=[DateTimeOffset]::Parse($state.closure_reserved_utc).ToUniversalTime()
$deadline=[DateTimeOffset]::Parse($state.deadline_utc).ToUniversalTime()
$principal=New-ScheduledTaskPrincipal -UserId $identity.Name -LogonType Interactive -RunLevel Limited
$python=Join-Path $data.app '.venv-app/Scripts/python.exe'
$receipts=@()
# Both triggers leave time for a native60-minute attempt before the global limit.
foreach($item in @(@{name='CloudRAG-I5-finalization-reserve';at=$reserve.AddMinutes(30)},@{name='CloudRAG-I5-finalization-backup';at=$deadline.AddMinutes(-70)})){
 if($item.at -le [DateTimeOffset]::UtcNow -or $item.at.AddMinutes(60) -ge $deadline){throw 'Finalization trigger elapsed or exceeds global deadline'}
 if(Get-ScheduledTask -TaskName $item.name -ErrorAction SilentlyContinue){throw 'Existing finalizer: inspect, never overwrite'}
 $entry=$data.finalizer_plans.PSObject.Properties[$item.name].Value
 if([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($entry)) -ne $external){throw 'Entry plan outside own external directory'}
 $entryData=Get-Content -LiteralPath $entry -Raw -Encoding UTF8 | ConvertFrom-Json
 if($entryData.current_task -ne $item.name -or $entryData.package -ne $data.package){throw 'Entry plan identity mismatch'}
 $action=New-ScheduledTaskAction -Execute $python -Argument ('-B -m scripts.study_operator.finalization --plan "'+$entry+'"') -WorkingDirectory $data.app
 $trigger=New-ScheduledTaskTrigger -Once -At $item.at.AddSeconds(1).LocalDateTime
 $settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 60) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -StartWhenAvailable
 $task=New-ScheduledTask -Principal $principal -Action $action -Trigger $trigger -Settings $settings
 Add-Content -LiteralPath (Join-Path $package 'SYSTEM_CHANGES.md') -Encoding UTF8 -Value ('BEFORE own Limited finalizer '+$item.name+'; due UTC '+$item.at.ToString('o')+'; external outputs and60min native limit; no user applications.')
 Register-ScheduledTask -TaskName $item.name -InputObject $task | Out-Null
 # Record ownership immediately, including partial registration if a later task fails.
 $record="import json,sys; from pathlib import Path; from filelock import FileLock; from src.ui.components.session_storage import atomic_json; p=Path(sys.argv[1]);`nwith FileLock(str(p/'state.lock'),timeout=10):`n s=json.loads((p/'STATE.json').read_bytes()); assert s['status']=='ACTIVE'; s['scheduled_tasks']=sorted(set(s.get('scheduled_tasks',[]))|{sys.argv[2]}); atomic_json(p/'STATE.json',s)"
 & $python -B -c $record $package $item.name
 if($LASTEXITCODE -ne 0){throw 'Registered task ownership checkpoint failed; inspect live task'}
 $observed=Get-ScheduledTask -TaskName $item.name
 $actual=[DateTimeOffset]::Parse($observed.Triggers[0].StartBoundary).ToUniversalTime()
 if($observed.Principal.RunLevel.ToString() -ne 'Limited' -or $observed.Settings.ExecutionTimeLimit -ne 'PT1H' -or [Math]::Abs(($actual-$item.at).TotalSeconds) -gt 2){throw 'Finalization trigger or native limit mismatch'}
 $receipts+=@{task=$item.name;run_level='Limited';trigger_utc=$actual.ToString('o');native_limit_minutes=60;entry_plan=$entry;entry_sha256=(Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLower();registered_utc=[DateTimeOffset]::UtcNow.ToString('o')}
}
$receipts|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $package 'independent-finalization-registration.json') -Encoding UTF8
$receipts|ConvertTo-Json -Depth 5

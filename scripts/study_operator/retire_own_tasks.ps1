param([Parameter(Mandatory=$true)][string]$Plan,[switch]$DryRun)
$ErrorActionPreference='Stop'
$identity=[Security.Principal.WindowsIdentity]::GetCurrent()
if(([Security.Principal.WindowsPrincipal]::new($identity)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)){throw 'Limited token required'}
$data=Get-Content -LiteralPath $Plan -Raw -Encoding UTF8 | ConvertFrom-Json
$state=Get-Content -LiteralPath (Join-Path $data.package 'STATE.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if($state.status -ne 'CLOSING' -and -not ($DryRun -and $state.status -eq 'ACTIVE')){throw 'Admission must close before own task retirement'}
if(Test-Path -LiteralPath (Join-Path $data.package 'MANIFEST_SHA256.jsonl')){throw 'Sealed package is read-only'}
$python=[IO.Path]::GetFullPath((Join-Path $data.app '.venv-app/Scripts/python.exe'))
$sdkPython=[IO.Path]::GetFullPath((Join-Path ([IO.Path]::GetDirectoryName($data.sdk)) '../platform/bundledpython/python.exe'))
$app=[IO.Path]::GetFullPath($data.app).TrimEnd('\','/')
$scope=[IO.Path]::GetFullPath($data.package).Replace('\','/').ToLowerInvariant()
$known=@($state.scheduled_tasks)
$tasks=@(Get-ScheduledTask | Where-Object {$_.TaskName -like 'CloudRAG-I5-*'})
foreach($task in $tasks){
 if($task.TaskName -notin $known -or $task.Principal.RunLevel.ToString() -ne 'Limited' -or $task.Actions.Count -ne 1){throw 'Unknown or elevated task; no retirement'}
 $action=$task.Actions[0]
 if([IO.Path]::GetFullPath($action.Execute) -ne $python -or [IO.Path]::GetFullPath($action.WorkingDirectory).TrimEnd('\','/') -ne $app){throw 'Task action is not the recorded worktree Python'}
 $arguments=$action.Arguments.Replace('\','/').ToLowerInvariant()
 $own=$arguments.Contains($scope)
 if($task.TaskName -in @($data.finalizer_tasks)){
  $entry=$data.finalizer_plans.PSObject.Properties[$task.TaskName].Value
  $own=$entry -and $arguments.Contains('scripts.study_operator.finalization') -and $arguments.Contains([IO.Path]::GetFullPath($entry).Replace('\','/').ToLowerInvariant())
 }
 if(-not $own){throw 'Task arguments do not prove package scope'}
}
if($DryRun){
 [pscustomobject]@{status='OWN_TASK_RETIREMENT_DRY_RUN_VALIDATED';tasks=$tasks.Count;no_tasks_stopped=$true;no_tasks_unregistered=$true} | ConvertTo-Json
 exit 0
}
# Validate the complete census before the first stop/unregister effect.
Add-Content -LiteralPath (Join-Path $data.package 'SYSTEM_CHANGES.md') -Encoding UTF8 -Value ('BEFORE finalization: retire only '+$tasks.Count+' registered I5 Limited tasks, bound to own worktree Python and package; refuse seal if Python writers remain. No I4 tasks or user applications touched.')
foreach($task in $tasks){
 if($task.TaskName -ne $data.current_task -and $task.State.ToString() -eq 'Running'){Stop-ScheduledTask -TaskName $task.TaskName}
 Unregister-ScheduledTask -TaskName $task.TaskName -Confirm:$false
}
$until=[DateTime]::UtcNow.AddSeconds(30)
do {
 $processes=@(Get-CimInstance Win32_Process | Where-Object {$_.ExecutablePath -and [IO.Path]::GetFullPath($_.ExecutablePath) -in @($python,$sdkPython) -and $_.ProcessId -ne $data.coordinator_pid})
 if($processes.Count -eq 0){break}
 Start-Sleep -Milliseconds 500
} while([DateTime]::UtcNow -lt $until)
if($processes.Count){throw 'Worktree Python still alive; preserve evidence and do not seal'}
if(@(Get-ScheduledTask | Where-Object {$_.TaskName -like 'CloudRAG-I5-*'}).Count){throw 'Own task definitions remain; no seal'}
[pscustomobject]@{status='OWN_TASKS_RETIRED_AND_WRITERS_QUIESCENT';tasks=@($tasks.TaskName);remaining_python=0;at=[DateTimeOffset]::UtcNow.ToString('o');user_applications_untouched=$true} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $data.retirement_receipt -Encoding UTF8

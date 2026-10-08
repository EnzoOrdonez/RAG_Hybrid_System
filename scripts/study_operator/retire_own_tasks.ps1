param([Parameter(Mandatory=$true)][string]$Plan,[switch]$DryRun,[switch]$SealedCleanup)
$ErrorActionPreference='Stop'
$identity=[Security.Principal.WindowsIdentity]::GetCurrent()
if(([Security.Principal.WindowsPrincipal]::new($identity)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)){throw 'Limited token required'}
$data=Get-Content -LiteralPath $Plan -Raw -Encoding UTF8 | ConvertFrom-Json
$state=Get-Content -LiteralPath (Join-Path $data.package 'STATE.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$sealed=Test-Path -LiteralPath (Join-Path $data.package 'MANIFEST_SHA256.jsonl')
if($SealedCleanup){
 if(-not $sealed -or $state.status -ne 'CLOSED_AWAITING_EXTERNAL_SEAL'){throw 'Verified sealed package required for external cleanup'}
 $external=[IO.Path]::GetFullPath($data.external)
 if([IO.Path]::GetDirectoryName($external) -ne [IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($data.package)) -or [IO.Path]::GetFileName($external) -notlike 'iteration5-finalization-*' -or [IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($data.retirement_receipt)) -ne $external){throw 'Cleanup receipt must remain external'}
}else{
 if($state.status -ne 'CLOSING' -and -not ($DryRun -and $state.status -eq 'ACTIVE')){throw 'Admission must close before own task retirement'}
 if($sealed){throw 'Sealed package is read-only'}
}
$python=[IO.Path]::GetFullPath((Join-Path $data.app '.venv-app/Scripts/python.exe'))
$sdkPython=[IO.Path]::GetFullPath((Join-Path ([IO.Path]::GetDirectoryName($data.sdk)) '../platform/bundledpython/python.exe'))
$binding=& $python -B -m scripts.study_operator.writer_quiescence --binding
if($LASTEXITCODE -ne 0){throw 'Native Python binding unavailable; no retirement'}
$native=($binding|ConvertFrom-Json).native_executable
$app=[IO.Path]::GetFullPath($data.app).TrimEnd('\','/')
$scope=[IO.Path]::GetFullPath($data.package).Replace('\','/').ToLowerInvariant()
$known=@($state.scheduled_tasks)
$tasks=@(Get-ScheduledTask | Where-Object {$_.TaskName -like 'CloudRAG-I5-*'})
$protected=@($data.finalizer_tasks)+@('CloudRAG-I5-safety-reserve','CloudRAG-I5-safety-deadline-backup')
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
 if($SealedCleanup -and ($task.TaskName -notin $protected -or ($task.State.ToString() -eq 'Running' -and $task.TaskName -ne $data.current_task))){throw 'Unexpected task or active backup; preserve independent retries'}
}
if($SealedCleanup){
 foreach($task in $tasks){Unregister-ScheduledTask -TaskName $task.TaskName -Confirm:$false}
 if(@(Get-ScheduledTask | Where-Object {$_.TaskName -like 'CloudRAG-I5-*'}).Count){throw 'Own safety definitions still remain'}
 [pscustomobject]@{status='OWN_SAFETY_TASKS_RETIRED_AFTER_VERIFIED_SEAL';tasks=@($tasks.TaskName);at=[DateTimeOffset]::UtcNow.ToString('o');sealed_package_not_modified=$true;user_applications_untouched=$true} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $data.retirement_receipt -Encoding UTF8
 exit 0
}
if($DryRun){
 [pscustomobject]@{status='OWN_TASK_RETIREMENT_DRY_RUN_VALIDATED';tasks=$tasks.Count;no_tasks_stopped=$true;no_tasks_unregistered=$true} | ConvertTo-Json
 exit 0
}
# Validate the complete census before the first stop/unregister effect.
Add-Content -LiteralPath (Join-Path $data.package 'SYSTEM_CHANGES.md') -Encoding UTF8 -Value ('BEFORE finalization: retire only '+$tasks.Count+' registered I5 Limited tasks, bound to own worktree Python and package; refuse seal if Python writers remain. No I4 tasks or user applications touched.')
foreach($task in $tasks){
 if($task.TaskName -in $protected){continue}
 if($task.TaskName -ne $data.current_task -and $task.State.ToString() -eq 'Running'){Stop-ScheduledTask -TaskName $task.TaskName}
 Unregister-ScheduledTask -TaskName $task.TaskName -Confirm:$false
}
$until=[DateTime]::UtcNow.AddSeconds(30)
do {
 $rows=@(Get-CimInstance Win32_Process | Where-Object {$_.ExecutablePath -and [IO.Path]::GetFullPath($_.ExecutablePath) -in @($python,$native,$sdkPython)} | Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine)
 $json=ConvertTo-Json -InputObject $rows -Depth 4 -Compress
 $selection=$json | & $python -B -m scripts.study_operator.writer_quiescence --native $native --launcher $python --sdk-python $sdkPython --package $data.package --owner 'C:/CloudRAG/operator-iteration5' --coordinator-pid $data.coordinator_pid --coordinator-plan $data.entry_plan
 if($LASTEXITCODE -ne 0){throw 'Native writer census failed; no seal'}
 $processes=@(($selection|ConvertFrom-Json).writer_pids)
 if($processes.Count -eq 0){break}
 Start-Sleep -Milliseconds 500
} while([DateTime]::UtcNow -lt $until)
if($processes.Count){throw 'Worktree Python still alive; preserve evidence and do not seal'}
# Preserve all independent STOP/report/seal retries until external verification
# succeeds. Their removal and receipt then happen outside the sealed package.
$remaining=@(Get-ScheduledTask | Where-Object {$_.TaskName -like 'CloudRAG-I5-*'})
if(@($remaining|Where-Object {$_.TaskName -notin $protected -or ($_.State.ToString() -eq 'Running' -and $_.TaskName -ne $data.current_task)}).Count){throw 'Unexpected remaining task or active safety backup; no seal'}
[pscustomobject]@{status='OWN_WRITERS_QUIESCENT_BACKUPS_PRESERVED';retired_tasks=@($tasks|Where-Object {$_.TaskName -notin $protected}|ForEach-Object {$_.TaskName});remaining_safety_tasks=@($remaining.TaskName);remaining_python=0;at=[DateTimeOffset]::UtcNow.ToString('o');user_applications_untouched=$true} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $data.retirement_receipt -Encoding UTF8

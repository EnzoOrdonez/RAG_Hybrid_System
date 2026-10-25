param([Parameter(Mandatory=$true)][string]$Plan)
$ErrorActionPreference='Stop'
$identity=[Security.Principal.WindowsIdentity]::GetCurrent()
if(([Security.Principal.WindowsPrincipal]::new($identity)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)){throw 'Limited token required'}
$data=Get-Content -LiteralPath $Plan -Raw -Encoding UTF8 | ConvertFrom-Json
foreach($name in $data.names){
 if($name -notmatch '^CloudRAG-I4-[A-Za-z0-9_.-]+$'){throw 'Unknown task scope'}
 $task=Get-ScheduledTask -TaskName $name
 $info=Get-ScheduledTaskInfo -TaskName $name
 if($task.State.ToString() -eq 'Running' -or ($info.NextRunTime -and $info.NextRunTime -gt (Get-Date))){throw 'Task became running or pending; preserve'}
 $task | Export-ScheduledTask | Set-Content -LiteralPath (Join-Path $data.xml_root ($name+'.xml')) -Encoding UTF8
 Unregister-ScheduledTask -TaskName $name -Confirm:$false
}
if(@(Get-ScheduledTask | Where-Object {$_.TaskName -like 'CloudRAG-I4-*'}).Count){throw 'Other I4 task definitions remain'}
[pscustomobject]@{status='OWNED_COMPLETED_I4_TASKS_RETIRED';count=$data.names.Count} | ConvertTo-Json

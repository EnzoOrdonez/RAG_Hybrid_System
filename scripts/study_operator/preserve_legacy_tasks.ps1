param([Parameter(Mandatory=$true)][string]$Plan)
$ErrorActionPreference='Stop'
$data=Get-Content -LiteralPath $Plan -Raw -Encoding UTF8 | ConvertFrom-Json
foreach($name in $data.names){if($name -notmatch '^CloudRAG-I4-[A-Za-z0-9_.-]+$'){throw 'Unknown task scope'}}
# One native enumeration avoids a separate slow CIM transaction per definition.
$service=New-Object -ComObject 'Schedule.Service'
$service.Connect()
$tasks=@($service.GetFolder('\').GetTasks(0) | Where-Object {$_.Name -like 'CloudRAG-I4-*'})
if($tasks.Count -ne $data.names.Count){throw 'Task membership changed; preserve'}
foreach($task in $tasks){
 if($task.Name -notin $data.names){throw 'Unknown owner'}
 if([int]$task.State -eq 4 -or $task.NextRunTime -gt (Get-Date)){throw 'Running or future writer'}
 $task.Xml | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $data.xml_root ($task.Name+'.xml'))
}
[pscustomobject]@{status='COMPLETED_DEFINITIONS_PRESERVED_NOT_RETIRED';count=$tasks.Count} | ConvertTo-Json

param([Parameter(ValueFromRemainingArguments=$true)][string[]]$OperatorArguments)
$ErrorActionPreference='Stop'
$identity=[Security.Principal.WindowsIdentity]::GetCurrent()
$principal=New-Object Security.Principal.WindowsPrincipal($identity)
if ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Abre PowerShell sin administrador para usar este operador.' }
$settings=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'installation.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$bundle=Join-Path $PSScriptRoot 'bundle'
$manifestPath=Join-Path $PSScriptRoot 'bundle_manifest.json'
$pointerPath=Join-Path $PSScriptRoot 'release.json'
if(Test-Path -LiteralPath $pointerPath){
    $pointer=Get-Content -LiteralPath $pointerPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if($pointer.schema_version -ne 1 -or $pointer.source_commit -cnotmatch '^[0-9a-f]{40}$' -or $pointer.bundle_path -cne ('releases/'+$pointer.source_commit+'/bundle')){throw 'Puntero de versión inválido. Conserva la instalación y revisa el recibo de actualización.'}
    $bundle=Join-Path $PSScriptRoot $pointer.bundle_path
    $manifestPath=Join-Path (Split-Path $bundle -Parent) 'bundle_manifest.json'
    if((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $pointer.manifest_sha256){throw 'Inventario de versión alterado. No ejecutes start; verifica la actualización.'}
}
$manifest=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if((Test-Path -LiteralPath $pointerPath) -and ($manifest.source_commit -cne $pointer.source_commit -or (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $manifest.wrapper_sha256)){throw 'Lanzador o versión alterados. Conserva la instalación y verifica sus hashes.'}
$files=@($manifest.files.PSObject.Properties)
if(@(Get-ChildItem -LiteralPath $bundle -Recurse -File).Count -ne $files.Count){throw 'Archivos inesperados en el operador. No ejecutes start; verifica el inventario.'}
foreach($file in $files){
    if($file.Name -cnotmatch '^(scripts/study_operator/|src/|config/)' -or $file.Name.Contains('..') -or $file.Name.Contains(':') -or $file.Name.Contains('\')){throw 'Ruta de versión inválida. Conserva la instalación.'}
    $target=Join-Path $bundle $file.Name
    $item=Get-Item -LiteralPath $target
    while($item){
        if($item.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Enlace inesperado en el operador. No ejecutes start.'}
        $item=if($item -is [IO.FileInfo]){$item.Directory}else{$item.Parent}
    }
    if((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -cne $file.Value){throw 'Código del operador alterado. No ejecutes start; verifica sus hashes.'}
}
$previousPath=$env:PYTHONPATH
$previousBytecode=$env:PYTHONDONTWRITEBYTECODE
$previousUtf8=$env:PYTHONUTF8
$previousLocation=Get-Location
try {
    $env:PYTHONPATH=$bundle
    $env:PYTHONDONTWRITEBYTECODE='1'
    $env:PYTHONUTF8='1'
    Set-Location -LiteralPath $PSScriptRoot
    & $settings.python -B -m scripts.study_operator.cli --root $PSScriptRoot @OperatorArguments
    $operatorExit=$LASTEXITCODE
} finally {
    Set-Location -LiteralPath $previousLocation.Path
    $env:PYTHONPATH=$previousPath
    $env:PYTHONDONTWRITEBYTECODE=$previousBytecode
    $env:PYTHONUTF8=$previousUtf8
}
exit $operatorExit

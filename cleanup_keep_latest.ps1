param(
    [switch]$Apply
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = [System.IO.Path]::GetFullPath($Root)

$KeepRelativeFiles = @(
    "top_angle_estimator.py",
    "TOP_ANGLE_USAGE.md",
    "preprocess_interface.py",
    "top_best.pt",
    "side_best.pt",
    "cleanup_keep_latest.ps1",
    "outputs\top_angle_resnet50_retrained_300_best.pth",
    "outputs\ultralytics_config\Ultralytics\settings.json",
    "outputs\ultralytics_config\Ultralytics\persistent_cache.json",
    "outputs\opencv_refined_all.csv"
)

$KeepFiles = $KeepRelativeFiles | ForEach-Object {
    [System.IO.Path]::GetFullPath((Join-Path $Root $_))
}

$KeepDataSamples = [System.IO.Path]::GetFullPath((Join-Path $Root "data_samples"))

function Test-IsKept {
    param([string]$Path)

    $FullPath = [System.IO.Path]::GetFullPath($Path)

    if ($KeepFiles -contains $FullPath) {
        return $true
    }

    if (
        $FullPath -eq $KeepDataSamples -or
        $FullPath.StartsWith($KeepDataSamples + [System.IO.Path]::DirectorySeparatorChar)
    ) {
        return $true
    }

    return $false
}

function Test-IsKeepAncestor {
    param([string]$Path)

    $FullPath = [System.IO.Path]::GetFullPath($Path)

    foreach ($KeepFile in $KeepFiles) {
        if ($KeepFile.StartsWith($FullPath + [System.IO.Path]::DirectorySeparatorChar)) {
            return $true
        }
    }

    if ($KeepDataSamples.StartsWith($FullPath + [System.IO.Path]::DirectorySeparatorChar)) {
        return $true
    }

    return $false
}

function Remove-OrPreview {
    param([System.IO.FileSystemInfo]$Item)

    if ($Apply) {
        Remove-Item -LiteralPath $Item.FullName -Recurse -Force
        Write-Host "Deleted: $($Item.FullName)"
    } else {
        Write-Host "Would delete: $($Item.FullName)"
    }
}

function Remove-NonKeptTree {
    param([System.IO.FileSystemInfo]$Item)

    if (Test-IsKept $Item.FullName) {
        return
    }

    if ($Item.PSIsContainer -and (Test-IsKeepAncestor $Item.FullName)) {
        foreach ($Child in Get-ChildItem -LiteralPath $Item.FullName -Force) {
            Remove-NonKeptTree $Child
        }
        return
    }

    Remove-OrPreview $Item
}

Write-Host "Project root: $Root"
Write-Host "Mode: $(if ($Apply) { 'APPLY - files will be deleted' } else { 'PREVIEW - no files will be deleted' })"
Write-Host ""

foreach ($Item in Get-ChildItem -LiteralPath $Root -Force) {
    Remove-NonKeptTree $Item
}

if ($Apply) {
    foreach ($Dir in Get-ChildItem -LiteralPath $Root -Directory -Recurse -Force | Sort-Object FullName -Descending) {
        if (Test-IsKept $Dir.FullName) {
            continue
        }

        $ChildCount = @(Get-ChildItem -LiteralPath $Dir.FullName -Force -ErrorAction SilentlyContinue).Count
        if ($ChildCount -eq 0) {
            Remove-Item -LiteralPath $Dir.FullName -Force
            Write-Host "Deleted empty dir: $($Dir.FullName)"
        }
    }
}

Write-Host ""
if ($Apply) {
    Write-Host "Cleanup complete."
} else {
    Write-Host "Preview complete. Run with -Apply to actually delete these files."
}

param(
    [string]$Root = "D:\desktop\github_push\sram_rc"
)

$ErrorActionPreference = "Stop"

function Get-ExperimentProcesses {
    Get-CimInstance Win32_Process |
        Where-Object {
            (
                $_.Name -eq "python.exe" -and
                $_.CommandLine -match "main.py --task_level (edge|node)" -and
                $_.CommandLine -match "maxnorm"
            ) -or (
                $_.Name -match "bash.exe" -and
                $_.CommandLine -match "for model in gat gcn sage pna sgformer polynormer" -and
                $_.CommandLine -match "maxnorm"
            )
        }
}

function Parse-LogFile {
    param(
        [string]$Path
    )

    if (!(Test-Path -LiteralPath $Path)) {
        return $null
    }

    $lines = Get-Content -LiteralPath $Path -Encoding UTF8
    if ($lines.Count -eq 0) {
        return $null
    }

    $trainMatches = @()
    $valMatches = @()
    $epochLineCounts = @{}
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "Epoch\s+(\d+)/\d+.*accuracy=([0-9.]+)\s+f1=([0-9.]+)") {
            $entry = [pscustomobject]@{
                Line = $i
                Epoch = [int]$matches[1]
                Acc = $matches[2]
                F1 = $matches[3]
            }

            $epochKey = [string]$entry.Epoch
            if (!$epochLineCounts.ContainsKey($epochKey)) {
                $epochLineCounts[$epochKey] = 0
            }
            $epochLineCounts[$epochKey]++

            if ($epochLineCounts[$epochKey] -eq 1) {
                $trainMatches += $entry
            }
            elseif ($epochLineCounts[$epochKey] -eq 2) {
                $valMatches += $entry
            }
        }
    }

    if ($trainMatches.Count -eq 0 -or $valMatches.Count -eq 0) {
        return $null
    }

    $bestVal = $valMatches | Sort-Object { [double]$_.F1 } -Descending | Select-Object -First 1
    $lastTrain = $trainMatches | Select-Object -Last 1
    $test = @{
        sandwich = $null
        ultra8t = $null
        array = $null
    }

    for ($i = $bestVal.Line + 1; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^Epoch\s+") {
            break
        }
        if ($lines[$i] -match "sandwich.*accuracy=([0-9.]+)\s+f1=([0-9.]+)") {
            $test.sandwich = @{ Acc = $matches[1]; F1 = $matches[2] }
        }
        elseif ($lines[$i] -match "ultra8t.*accuracy=([0-9.]+)\s+f1=([0-9.]+)") {
            $test.ultra8t = @{ Acc = $matches[1]; F1 = $matches[2] }
        }
        elseif ($lines[$i] -match "array_128_32_8t.*accuracy=([0-9.]+)\s+f1=([0-9.]+)") {
            $test.array = @{ Acc = $matches[1]; F1 = $matches[2] }
        }
    }

    if ($null -eq $test.sandwich -or $null -eq $test.ultra8t -or $null -eq $test.array) {
        return $null
    }

    [pscustomobject]@{
        TrainAcc = $lastTrain.Acc
        TrainF1 = $lastTrain.F1
        SandwichAcc = $test.sandwich.Acc
        SandwichF1 = $test.sandwich.F1
        Ultra8tAcc = $test.ultra8t.Acc
        Ultra8tF1 = $test.ultra8t.F1
        ArrayAcc = $test.array.Acc
        ArrayF1 = $test.array.F1
        BestValF1 = $bestVal.F1
        BestEpoch = $bestVal.Epoch
    }
}

function Find-LatestModelLog {
    param(
        [string]$LogsDir,
        [string]$Task,
        [string]$Model
    )

    $candidates = Get-ChildItem -LiteralPath $LogsDir -File -Filter "classification_$Task*.txt" |
        Sort-Object LastWriteTime -Descending

    foreach ($file in $candidates) {
        if ($file.Length -eq 0) {
            continue
        }
        $head = Get-Content -LiteralPath $file.FullName -Encoding UTF8 -TotalCount 8
        if (($head -join "`n") -match "(?i)\b$([regex]::Escape($Model))\b") {
            return $file.FullName
        }
    }

    return $null
}

function Format-Row {
    param(
        [string]$Model,
        $Result
    )

    "| $Model | $($Result.TrainAcc) | $($Result.TrainF1) | $($Result.SandwichAcc) | $($Result.SandwichF1) | $($Result.Ultra8tAcc) | $($Result.Ultra8tF1) | $($Result.ArrayAcc) | $($Result.ArrayF1) | $($Result.BestValF1) | $($Result.BestEpoch) |"
}

function Update-ResultFile {
    param(
        [string]$Path,
        [hashtable]$Rows
    )

    $lines = Get-Content -LiteralPath $Path -Encoding UTF8
    $updated = foreach ($line in $lines) {
        $changed = $false
        foreach ($model in $Rows.Keys) {
            if ($line -match "^\|\s*$([regex]::Escape($model))\s*\|") {
                Format-Row -Model $model -Result $Rows[$model]
                $changed = $true
                break
            }
        }
        if (!$changed) {
            if ($line -match "^\*更新时间:") {
                "*更新时间: $(Get-Date -Format 'yyyy-MM-dd') | 环境: shumo*"
            }
            else {
                $line
            }
        }
    }

    Set-Content -LiteralPath $Path -Value $updated -Encoding UTF8
}

while (@(Get-ExperimentProcesses).Count -gt 0) {
    Start-Sleep -Seconds 300
}

$logsDir = Join-Path $Root "rcg\logs\ssram"
$edgeFile = Join-Path $Root "edge_classification_results_maxnorm.md"
$nodeFile = Join-Path $Root "node_classification_results_maxnorm.md"

$edgeRows = @{}
$nodeRows = @{}

foreach ($model in @("SGFormer", "Polynormer")) {
    $logPath = Find-LatestModelLog -LogsDir $logsDir -Task "edge" -Model $model.ToLower()
    $result = if ($logPath) { Parse-LogFile -Path $logPath } else { $null }
    if ($result) {
        $edgeRows[$model] = $result
    }
}

$nodeLogPath = Find-LatestModelLog -LogsDir $logsDir -Task "node" -Model "polynormer"
$nodeResult = if ($nodeLogPath) { Parse-LogFile -Path $nodeLogPath } else { $null }
if ($nodeResult) {
    $nodeRows["Polynormer"] = $nodeResult
}

if ($edgeRows.Count -gt 0) {
    Update-ResultFile -Path $edgeFile -Rows $edgeRows
}

if ($nodeRows.Count -gt 0) {
    Update-ResultFile -Path $nodeFile -Rows $nodeRows
}

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"$stamp updated edge=$($edgeRows.Count) node=$($nodeRows.Count)" |
    Set-Content -LiteralPath (Join-Path $Root "watch_update_maxnorm_results.log") -Encoding UTF8

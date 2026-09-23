param(
    [int]$Workers = 4,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$runFolder = Join-Path $repoRoot 'results\b_validation'
New-Item -ItemType Directory -Force -Path $runFolder | Out-Null
$logPath = Join-Path $runFolder 'driver.log'
$statusPath = Join-Path $runFolder 'driver_status.json'

function Save-Status([string]$Stage, [int]$ExitCode) {
    $status = @{
        stage = $Stage
        exit_code = $ExitCode
        updated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    }
    $temporary = "$statusPath.tmp"
    $status | ConvertTo-Json | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

Save-Status 'forecasting' 0
& $Python -u run_b_validation.py --workers $Workers 2>&1 | Tee-Object -FilePath $logPath -Append
if ($LASTEXITCODE -ne 0) { Save-Status 'forecast_failed' $LASTEXITCODE; exit $LASTEXITCODE }

Save-Status 'retrying_identical_failures' 0
& $Python -u run_b_validation.py --workers $Workers --retry-failures 2>&1 | Tee-Object -FilePath $logPath -Append
if ($LASTEXITCODE -ne 0) { Save-Status 'retry_failed' $LASTEXITCODE; exit $LASTEXITCODE }

Save-Status 'evaluating' 0
& $Python -u evaluate_b_validation.py --freeze 2>&1 | Tee-Object -FilePath $logPath -Append
if ($LASTEXITCODE -ne 0) { Save-Status 'evaluation_or_freeze_failed' $LASTEXITCODE; exit $LASTEXITCODE }

Save-Status 'frozen' 0

# ─────────────────────────────────────────────────────────────────────────────
# run_batch.ps1  —  여러 영상에 대해 reid_performance.ipynb 를 자동 실행합니다.
#
# 사용법:
#   .\run_batch.ps1 -Videos "C:\data\cam1.avi","C:\data\cam2.avi"
#   .\run_batch.ps1 -Videos (Get-Item "C:\data\*.avi").FullName
#
# 옵션 파라미터:
#   -Videos      분석할 영상 파일 경로 목록 (필수)
#   -OutputDir   결과 pkl 저장 폴더       (기본값: results)
#   -MaxFrames   분석할 최대 프레임 수     (기본값: 4000)
#   -Parallel    동시 실행 수              (기본값: 1, 순차 실행)
#
# 예시:
#   .\run_batch.ps1 -Videos "C:\data\cam1.avi","C:\data\cam2.avi" -MaxFrames 2000
# ─────────────────────────────────────────────────────────────────────────────

param(
    [Parameter(Mandatory=$true)]
    [string[]]$Videos,

    [string]$OutputDir = "results",
    [int]$MaxFrames    = 4000,
    [int]$Parallel     = 1
)

$ErrorActionPreference = "Stop"

# 노트북 경로 (이 스크립트와 같은 폴더)
$NbIn     = Join-Path $PSScriptRoot "reid_performance.ipynb"
$NbOutDir = Join-Path $PSScriptRoot "outputs"

# 결과 폴더 생성
New-Item -ItemType Directory -Force -Path $NbOutDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# papermill 설치 확인
if (-not (Get-Command papermill -ErrorAction SilentlyContinue)) {
    Write-Host "[오류] papermill 이 설치되어 있지 않습니다." -ForegroundColor Red
    Write-Host "  pip install papermill"
    exit 1
}

Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Re-ID 배치 분석 시작"
Write-Host "  영상 수    : $($Videos.Count)"
Write-Host "  결과 폴더  : $OutputDir"
Write-Host "  최대 프레임: $MaxFrames"
Write-Host "  동시 실행  : $Parallel"
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Cyan

$OkCount   = 0
$FailCount = 0
$SkipCount = 0

function Invoke-OneVideo {
    param([string]$Video)

    $Name   = [System.IO.Path]::GetFileNameWithoutExtension($Video)
    $NbOut  = Join-Path $NbOutDir "$Name.ipynb"

    Write-Host ""
    Write-Host "▶ 처리 중: $Video" -ForegroundColor Yellow

    if (-not (Test-Path $Video)) {
        Write-Host "  ✗ 파일 없음: $Video — 건너뜁니다." -ForegroundColor DarkYellow
        return "skip"
    }

    try {
        papermill $NbIn $NbOut `
            -p VIDEO_PATH  $Video      `
            -p OUTPUT_DIR  $OutputDir  `
            -p MAX_FRAMES  $MaxFrames  `
            --log-output 2>&1 | Where-Object { $_ -match "(완료|오류|Error|WARNING|Executing)" }

        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ 완료 → $NbOut" -ForegroundColor Green
            Write-Host "  ✓ pkl  → $OutputDir\$Name.pkl" -ForegroundColor Green
            return "ok"
        } else {
            Write-Host "  ✗ 실패: $Video (exitcode=$LASTEXITCODE)" -ForegroundColor Red
            return "fail"
        }
    } catch {
        Write-Host "  ✗ 예외 발생: $_" -ForegroundColor Red
        return "fail"
    }
}

# 순차 실행 (Parallel=1) 또는 병렬 실행
if ($Parallel -gt 1) {
    $Results = $Videos | ForEach-Object -Parallel {
        $Video   = $_
        $Name    = [System.IO.Path]::GetFileNameWithoutExtension($Video)
        $NbOut   = Join-Path $using:NbOutDir "$Name.ipynb"

        if (-not (Test-Path $Video)) {
            Write-Host "  ✗ 파일 없음: $Video" -ForegroundColor DarkYellow
            return "skip"
        }
        try {
            papermill $using:NbIn $NbOut `
                -p VIDEO_PATH $Video `
                -p OUTPUT_DIR $using:OutputDir `
                -p MAX_FRAMES $using:MaxFrames `
                --log-output | Out-Null
            return "ok"
        } catch {
            return "fail"
        }
    } -ThrottleLimit $Parallel

    $OkCount   = ($Results | Where-Object { $_ -eq "ok"   }).Count
    $FailCount = ($Results | Where-Object { $_ -eq "fail" }).Count
    $SkipCount = ($Results | Where-Object { $_ -eq "skip" }).Count
} else {
    foreach ($Video in $Videos) {
        $Result = Invoke-OneVideo -Video $Video
        switch ($Result) {
            "ok"   { $OkCount++   }
            "fail" { $FailCount++ }
            "skip" { $SkipCount++ }
        }
    }
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  배치 실행 완료"
Write-Host "  성공: $OkCount  /  실패: $FailCount  /  건너뜀: $SkipCount"
Write-Host "  pkl 결과    : $OutputDir\"
Write-Host "  출력 노트북 : $NbOutDir\"
Write-Host ""
Write-Host "  다음 단계: reid_aggregate.ipynb 실행"
Write-Host "  jupyter notebook notebooks/reid_aggregate.ipynb"
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Cyan

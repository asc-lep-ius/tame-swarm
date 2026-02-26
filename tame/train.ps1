# TAME Training Script
# Runs training workflow in Docker container with GPU support
#
# Usage:
#   .\train.ps1                    # Quick test (100 steps)
#   .\train.ps1 train              # Full training (5000 steps)
#   .\train.ps1 train 10000        # Custom steps
#   .\train.ps1 export             # Export checkpoint for inference
#   .\train.ps1 full               # Train + Export

param(
    [Parameter(Position=0)]
    [ValidateSet("test", "train", "export", "full", "check")]
    [string]$Mode = "test",
    
    [Parameter(Position=1)]
    [int]$Steps = 5000,
    
    [switch]$UseLora,
    [switch]$Help
)

if ($Help) {
    Write-Host "TAME Training Script" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\train.ps1 [mode] [steps] [-UseLora] [-Help]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Modes:"
    Write-Host "  test    - Quick test (100 steps, verify setup works)"
    Write-Host "  train   - Full training (default 5000 steps)"
    Write-Host "  export  - Export latest checkpoint for inference"
    Write-Host "  full    - Train + Export in one step"
    Write-Host "  check   - Verify dependencies"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\train.ps1                  # Quick test"
    Write-Host "  .\train.ps1 train            # Train 5000 steps"
    Write-Host "  .\train.ps1 train 10000      # Train 10000 steps"
    Write-Host "  .\train.ps1 train -UseLora   # Train with LoRA adapters"
    Write-Host "  .\train.ps1 full             # Train + Export"
    exit 0
}

Write-Host "TAME Training - Mode: $Mode" -ForegroundColor Cyan
Write-Host ""

# Build command args
$cmdArgs = @("--mode", $Mode)

if ($Mode -eq "train" -or $Mode -eq "full") {
    $cmdArgs += @("--steps", $Steps)
    Write-Host "Training for $Steps steps..." -ForegroundColor Green
}

if ($UseLora) {
    $cmdArgs += "--use_lora"
    Write-Host "Using LoRA for memory-efficient training" -ForegroundColor Green
}

Write-Host ""

# Run container
docker-compose -f docker-compose.train.yml run --rm train @cmdArgs

$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host ""
    Write-Host "Training complete!" -ForegroundColor Green
    
    if ($Mode -eq "full" -or $Mode -eq "export") {
        Write-Host ""
        Write-Host "Exported model ready at: ./tame_inference" -ForegroundColor Cyan
        Write-Host "Start inference server with: .\dev.ps1" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "Training failed with exit code $exitCode" -ForegroundColor Red
}

exit $exitCode

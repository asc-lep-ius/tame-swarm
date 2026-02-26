@echo off
REM TAME Training Script
REM Runs training workflow in Docker container with GPU support
REM
REM Usage:
REM   train.bat                    - Quick test (100 steps)
REM   train.bat train              - Full training (5000 steps)
REM   train.bat train 10000        - Custom steps
REM   train.bat export             - Export checkpoint for inference
REM   train.bat full               - Train + Export

setlocal enabledelayedexpansion

set MODE=%1
set STEPS=%2

if "%MODE%"=="" set MODE=test
if "%STEPS%"=="" set STEPS=5000

echo TAME Training - Mode: %MODE%
echo.

if "%MODE%"=="test" (
    echo Running quick test ^(100 steps^)...
    docker-compose -f docker-compose.train.yml run --rm train --mode test
) else if "%MODE%"=="train" (
    echo Training for %STEPS% steps...
    docker-compose -f docker-compose.train.yml run --rm train --mode train --steps %STEPS%
) else if "%MODE%"=="export" (
    echo Exporting checkpoint for inference...
    docker-compose -f docker-compose.train.yml run --rm train --mode export
) else if "%MODE%"=="full" (
    echo Running full pipeline ^(train + export^)...
    docker-compose -f docker-compose.train.yml run --rm train --mode full --steps %STEPS%
) else if "%MODE%"=="check" (
    echo Checking dependencies...
    docker-compose -f docker-compose.train.yml run --rm train --mode check
) else (
    echo Unknown mode: %MODE%
    echo.
    echo Usage: train.bat [mode] [steps]
    echo.
    echo Modes:
    echo   test    - Quick test ^(100 steps^)
    echo   train   - Full training ^(default 5000 steps^)
    echo   export  - Export checkpoint for inference
    echo   full    - Train + Export
    echo   check   - Verify dependencies
    echo.
    echo Examples:
    echo   train.bat                  - Quick test
    echo   train.bat train            - Train 5000 steps
    echo   train.bat train 10000      - Train 10000 steps
    echo   train.bat full             - Train + Export
    exit /b 1
)

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Training complete!
    if "%MODE%"=="full" (
        echo.
        echo Exported model ready at: .\tame_inference
        echo Start inference server with: dev.bat
    )
    if "%MODE%"=="export" (
        echo.
        echo Exported model ready at: .\tame_inference
        echo Start inference server with: dev.bat
    )
) else (
    echo.
    echo Training failed with exit code %ERRORLEVEL%
)

pause

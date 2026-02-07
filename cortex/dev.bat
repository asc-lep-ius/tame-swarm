@echo off
REM TAME Cortex Development Server
REM Starts the container with live code reload

echo Starting TAME Cortex in development mode...
echo Code changes will auto-reload (no rebuild needed)
echo.

docker-compose -f docker-compose.dev.yml up --build

pause

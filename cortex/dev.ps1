# TAME Cortex Development Server
# Starts the container with live code reload

Write-Host "Starting TAME Cortex in development mode..." -ForegroundColor Cyan
Write-Host "Code changes will auto-reload (no rebuild needed)" -ForegroundColor Green
Write-Host ""

# Build and run with compose
docker-compose -f docker-compose.dev.yml up --build

Write-Host ""
Write-Host "Container stopped." -ForegroundColor Yellow

# Helper script to run Python scripts in WSL with GPU support
# Usage: .\run_wsl.ps1 "path/to/script.py"

param(
    [string]$ScriptPath = "tflite_room_training.py"
)

Write-Host "Running in WSL with GPU support..." -ForegroundColor Green
Write-Host "Script: $ScriptPath" -ForegroundColor Cyan
Write-Host ""

wsl bash -c "cd /mnt/c/Users/Klein/Desktop/Labcourse && source .venv_wsl/bin/activate && python3 '$ScriptPath'"
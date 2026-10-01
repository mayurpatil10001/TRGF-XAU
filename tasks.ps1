# TEGF-XAU Task Runner (PowerShell)
# Usage: .\tasks.ps1 <target>

param([string]$Target = "help")

$Root = Split-Path $MyInvocation.MyCommand.Path
Set-Location $Root

function Run-Python { python -m $args }

switch ($Target) {
    "help" {
        Write-Host "Available targets:"
        Write-Host "  setup      - Install Python and Node dependencies"
        Write-Host "  data       - Load, clean, and QC the raw data"
        Write-Host "  features   - Build features and labels"
        Write-Host "  train      - Run walk-forward training"
        Write-Host "  backtest   - Run the cost-aware backtester"
        Write-Host "  gate       - Run the profit gate evaluation"
        Write-Host "  report     - Generate the final report"
        Write-Host "  test       - Run all pytest tests"
        Write-Host "  bot        - Start the live MT5 bot"
        Write-Host "  api        - Start the FastAPI backend"
        Write-Host "  ui         - Start the React frontend dev server"
        Write-Host "  all        - Run the full pipeline (data->report)"
    }
    "setup" {
        pip install -r requirements.txt
        Set-Location frontend
        npm install
        Set-Location $Root
    }
    "data" {
        Run-Python src.data.loader --config config/config.yaml
    }
    "features" {
        Run-Python src.features.feature_pipeline --config config/config.yaml
        Run-Python src.labels.triple_barrier --config config/config.yaml
    }
    "train" {
        Run-Python src.training.walk_forward --config config/config.yaml
    }
    "backtest" {
        Run-Python src.backtest.engine --config config/config.yaml
    }
    "gate" {
        Run-Python src.evaluation.profit_gate --config config/config.yaml
    }
    "report" {
        Run-Python src.reporting.report_builder --config config/config.yaml
    }
    "test" {
        pytest tests/ -v --tb=short
    }
    "bot" {
        Run-Python src.live.run_bot --config config/config.yaml
    }
    "api" {
        uvicorn src.api.main:app --port 8000 --reload
    }
    "ui" {
        Set-Location frontend
        npm run dev
    }
    "all" {
        Run-Python src.run_all --config config/config.yaml
    }
    default {
        Write-Host "Unknown target: $Target. Run '.\tasks.ps1 help' for usage."
    }
}

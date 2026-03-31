# ─────────────────────────────────────────────────────────────────────────────
# ScoutForge — Windows Control Script (PowerShell)
# Usage: .\run.ps1 [command]
# ─────────────────────────────────────────────────────────────────────────────
# Requirements:
#   - Docker Desktop for Windows (https://docker.com/products/docker-desktop)
#   - Ollama for Windows      (https://ollama.com)
#   - PowerShell 5.1 or later (built into Windows 10/11)
# ─────────────────────────────────────────────────────────────────────────────

param([string]$Command = "help")

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$AgentName    = "ScoutForge"
$EnvFile      = ".env"
$DashboardPort = 8888

# ── Colours ───────────────────────────────────────────────────────────────────
function Write-Info    { param($msg) Write-Host "  $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "  $msg" -ForegroundColor Green }
function Write-Warn    { param($msg) Write-Host "  $msg" -ForegroundColor Yellow }
function Write-Err     { param($msg) Write-Host "  $msg" -ForegroundColor Red; exit 1 }

function Show-Banner {
    Write-Host ""
    Write-Host "  ╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "  ║   🔭  $AgentName — Agentic Research Agent      ║" -ForegroundColor Cyan
    Write-Host "  ╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

# ── Helpers ───────────────────────────────────────────────────────────────────
function Load-Env {
    if (-not (Test-Path $EnvFile)) {
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" $EnvFile
            Write-Warn ".env created from .env.example — review it, then re-run."
        } else {
            Write-Warn ".env not found. Creating a default one..."
            @"
OLLAMA_URL=http://host.docker.internal:11434
REPORTS_DIR=./reports
RESEARCH_PORT=8888
RUN_ON_START=false
"@ | Set-Content $EnvFile
        }
    }
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $k = $matches[1].Trim(); $v = $matches[2].Trim()
            if ($k -eq "RESEARCH_PORT") { $script:DashboardPort = [int]$v }
            [System.Environment]::SetEnvironmentVariable($k, $v, "Process")
        }
    }
}

function Check-Docker {
    try { docker info 2>&1 | Out-Null }
    catch { Write-Err "Docker is not running. Start Docker Desktop first." }
    if ($LASTEXITCODE -ne 0) { Write-Err "Docker is not running. Start Docker Desktop first." }
}

function Generate-Secret {
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    return ([System.BitConverter]::ToString($bytes) -replace '-', '').ToLower()
}

# ── Commands ──────────────────────────────────────────────────────────────────
function Cmd-Setup {
    Show-Banner
    Check-Docker
    Load-Env

    # Generate SearXNG secret key if still placeholder
    $settingsFile = "docker\searxng\settings.yml"
    if (Test-Path $settingsFile) {
        $content = Get-Content $settingsFile -Raw
        if ($content -match "change-me-run-openssl-rand-hex-32") {
            Write-Warn "Generating SearXNG secret key..."
            $key = Generate-Secret
            $content = $content -replace "change-me-run-openssl-rand-hex-32-and-paste-here", $key
            Set-Content $settingsFile $content
            Write-Success "SearXNG secret key generated."
        }
    }

    Write-Info "Building and starting all services..."
    docker compose up -d --build
    if ($LASTEXITCODE -ne 0) { Write-Err "docker compose failed." }

    Write-Host ""
    Write-Success "Setup complete!"
    Write-Host "   Dashboard  -> http://localhost:$DashboardPort" -ForegroundColor Cyan
    Write-Host "   Run again  -> .\run.ps1 open" -ForegroundColor Yellow
    Write-Host ""
}

function Cmd-Start {
    Show-Banner
    Check-Docker
    Load-Env
    Write-Info "Starting $AgentName..."
    docker compose up -d --build
    if ($LASTEXITCODE -ne 0) { Write-Err "docker compose failed." }
    Write-Host ""
    Write-Success "Running!"
    Write-Host "   Dashboard -> http://localhost:$DashboardPort" -ForegroundColor Cyan
    Write-Host ""
}

function Cmd-Stop {
    Check-Docker
    Write-Info "Stopping $AgentName..."
    docker compose down
    Write-Success "Stopped."
}

function Cmd-Restart {
    Check-Docker
    Load-Env
    Write-Info "Restarting $AgentName..."
    docker compose restart
    Write-Success "Restarted -> http://localhost:$DashboardPort"
}

function Cmd-Rebuild {
    Check-Docker
    Load-Env
    Write-Info "Rebuilding $AgentName (picks up code/config changes)..."
    docker compose up -d --build --force-recreate
    Write-Success "Rebuilt -> http://localhost:$DashboardPort"
}

function Cmd-Logs {
    Check-Docker
    Write-Info "Streaming logs (Ctrl+C to stop)..."
    docker compose logs -f --tail=100
}

function Cmd-Status {
    Check-Docker
    Write-Host ""
    docker compose ps
    Write-Host ""
    try {
        $health = Invoke-RestMethod "http://localhost:$DashboardPort/health" -ErrorAction Stop
        Write-Success "Agent is healthy."
    } catch {
        Write-Warn "Agent not reachable at http://localhost:$DashboardPort"
    }
    Write-Host ""
}

function Cmd-Open {
    Load-Env
    $url = "http://localhost:$DashboardPort"
    Write-Info "Opening $url ..."
    Start-Process $url
}

function Cmd-Research {
    Load-Env
    Write-Info "Triggering on-demand research run..."
    try {
        Invoke-RestMethod -Method POST "http://localhost:$DashboardPort/api/run" -ErrorAction Stop | Out-Null
        Write-Success "Research run started. Takes 4–10 minutes."
        Write-Host "   Watch logs -> .\run.ps1 logs" -ForegroundColor Yellow
        Write-Host "   Dashboard  -> http://localhost:$DashboardPort" -ForegroundColor Cyan
    } catch {
        Write-Err "Could not reach agent. Is it running? Try: .\run.ps1 start"
    }
}

function Cmd-Help {
    Show-Banner
    Write-Host "  Usage: .\run.ps1 [command]" -ForegroundColor White
    Write-Host ""
    Write-Host "  First time:" -ForegroundColor Cyan
    Write-Host "    setup       Generate secret key, build and start all services"
    Write-Host ""
    Write-Host "  Core:" -ForegroundColor Cyan
    Write-Host "    start       Build and start all services"
    Write-Host "    stop        Stop all services"
    Write-Host "    restart     Restart without rebuilding"
    Write-Host "    rebuild     Rebuild after code changes"
    Write-Host ""
    Write-Host "  Info:" -ForegroundColor Cyan
    Write-Host "    status      Show container status"
    Write-Host "    logs        Stream live logs"
    Write-Host "    open        Open dashboard in browser"
    Write-Host "    research    Trigger a research run now"
    Write-Host ""
    Write-Host "  Examples:" -ForegroundColor Cyan
    Write-Host "    .\run.ps1 setup       # Run once on first install"
    Write-Host "    .\run.ps1 start       # Start everything"
    Write-Host "    .\run.ps1 open        # Open dashboard in browser"
    Write-Host "    .\run.ps1 logs        # Watch live logs"
    Write-Host ""
}

# ── Router ────────────────────────────────────────────────────────────────────
switch ($Command.ToLower()) {
    "setup"    { Cmd-Setup }
    "start"    { Cmd-Start }
    "stop"     { Cmd-Stop }
    "restart"  { Cmd-Restart }
    "rebuild"  { Cmd-Rebuild }
    "logs"     { Cmd-Logs }
    "status"   { Cmd-Status }
    "open"     { Cmd-Open }
    "research" { Cmd-Research }
    "help"     { Cmd-Help }
    default    { Write-Err "Unknown command '$Command'. Run: .\run.ps1 help" }
}

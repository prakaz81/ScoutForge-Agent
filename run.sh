#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# InfoExplorer Agent — Control Script
# Usage: ./run.sh [command]
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

AGENT_NAME="InfoExplorer Agent"
DASHBOARD_PORT="${RESEARCH_PORT:-8888}"
ENV_FILE=".env"
COMPOSE="docker compose"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'

info()    { echo -e "${CYAN}▶  $*${NC}"; }
success() { echo -e "${GREEN}✔  $*${NC}"; }
warn()    { echo -e "${YELLOW}⚠  $*${NC}"; }
error()   { echo -e "${RED}✖  $*${NC}"; exit 1; }

banner() {
  echo ""
  echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
  echo -e "${CYAN}║   🤖  ${AGENT_NAME}          ║${NC}"
  echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
  echo ""
}

# ── Preflight checks ──────────────────────────────────────────────────────────
check_env() {
  if [ ! -f "$ENV_FILE" ]; then
    warn ".env not found — creating from .env.example"
    cp .env.example .env
    warn "Please edit .env and set your SEARXNG_URL port, then re-run."
    exit 1
  fi
  # Load env for port reading
  set -a; source "$ENV_FILE"; set +a
  DASHBOARD_PORT="${RESEARCH_PORT:-8888}"
}

check_docker() {
  docker info &>/dev/null || error "Docker is not running. Start Docker Desktop first."
}

# ── Commands ──────────────────────────────────────────────────────────────────

cmd_start() {
  banner
  check_docker
  check_env
  info "Building and starting $AGENT_NAME..."
  $COMPOSE up -d --build
  echo ""
  success "Agent is running!"
  echo -e "   Dashboard → ${CYAN}http://localhost:${DASHBOARD_PORT}${NC}"
  echo -e "   Logs      → ${YELLOW}./run.sh logs${NC}"
  echo -e "   Run now   → ${YELLOW}./run.sh research${NC}"
  echo ""
}

cmd_stop() {
  check_docker
  info "Stopping $AGENT_NAME..."
  $COMPOSE down
  success "Agent stopped."
}

cmd_restart() {
  check_docker
  check_env
  info "Restarting $AGENT_NAME..."
  $COMPOSE restart
  success "Agent restarted → http://localhost:${DASHBOARD_PORT}"
}

cmd_rebuild() {
  check_docker
  check_env
  info "Rebuilding and restarting $AGENT_NAME (picks up code/config changes)..."
  $COMPOSE up -d --build --force-recreate
  success "Agent rebuilt → http://localhost:${DASHBOARD_PORT}"
}

cmd_logs() {
  check_docker
  info "Streaming logs (Ctrl+C to stop)..."
  $COMPOSE logs -f --tail=100
}

cmd_status() {
  check_docker
  echo ""
  $COMPOSE ps
  echo ""
  # Hit the health endpoint if the container is up
  if curl -sf "http://localhost:${DASHBOARD_PORT:-8888}/health" &>/dev/null; then
    STATUS=$(curl -sf "http://localhost:${DASHBOARD_PORT:-8888}/api/status" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'  Last run : {d.get(\"status\",\"unknown\")}')
print(f'  Timestamp: {d.get(\"timestamp\",\"—\")}')
print(f'  Report   : {d.get(\"report\",\"none yet\")}')
" 2>/dev/null || echo "  (could not read status)")
    echo -e "${STATUS}"
  fi
  echo ""
}

cmd_research() {
  info "Triggering on-demand research run..."
  RESP=$(curl -sf -X POST "http://localhost:${DASHBOARD_PORT:-8888}/api/run" 2>/dev/null || echo "")
  if [ -z "$RESP" ]; then
    error "Could not reach agent. Is it running? Try: ./run.sh start"
  fi
  success "Research run started. This takes 3–5 minutes."
  echo -e "   Watch progress → ${YELLOW}./run.sh logs${NC}"
  echo -e "   Dashboard      → ${CYAN}http://localhost:${DASHBOARD_PORT:-8888}${NC}"
}

cmd_reports() {
  REPORT_DIR="${REPORTS_DIR:-./reports}"
  echo ""
  if [ -d "$REPORT_DIR" ] && [ "$(ls -A "$REPORT_DIR"/*.md 2>/dev/null)" ]; then
    info "Saved reports in ${REPORT_DIR}:"
    echo ""
    ls -lt "$REPORT_DIR"/*.md | awk '{print "  " $NF}' | head -20
    echo ""
    LATEST=$(ls -t "$REPORT_DIR"/*.md 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
      echo -e "   Latest → ${CYAN}${LATEST}${NC}"
      echo -e "   Open   → ${YELLOW}open \"${LATEST}\"${NC}"
    fi
  else
    warn "No reports yet. Run: ./run.sh research"
  fi
  echo ""
}

cmd_open() {
  open "http://localhost:${DASHBOARD_PORT:-8888}" 2>/dev/null || \
    info "Dashboard: http://localhost:${DASHBOARD_PORT:-8888}"
}

cmd_setup() {
  banner
  check_docker
  check_env

  # Generate SearXNG secret key if still placeholder
  SEARXNG_SETTINGS="docker/searxng/settings.yml"
  if [ -f "$SEARXNG_SETTINGS" ] && grep -q "change-me-run-openssl-rand-hex-32" "$SEARXNG_SETTINGS" 2>/dev/null; then
    warn "Generating SearXNG secret key..."
    KEY=$(openssl rand -hex 32)
    sed -i '' "s|change-me-run-openssl-rand-hex-32-and-paste-here|${KEY}|g" "$SEARXNG_SETTINGS"
    success "SearXNG secret key generated."
  fi

  info "Building and starting all services..."
  $COMPOSE up -d --build

  echo ""
  success "Setup complete!"
  echo -e "   Research Agent → ${CYAN}http://localhost:${DASHBOARD_PORT:-8888}${NC}"
  echo ""
  echo -e "   Run research   → ${YELLOW}./run.sh research${NC}"
  echo ""
}

cmd_open_latest() {
  REPORT_DIR="${REPORTS_DIR:-./reports}"
  LATEST=$(ls -t "$REPORT_DIR"/*.md 2>/dev/null | head -1)
  if [ -z "$LATEST" ]; then
    warn "No reports yet. Run: ./run.sh research"
  else
    info "Opening latest report: $LATEST"
    open "$LATEST" 2>/dev/null || cat "$LATEST"
  fi
}

cmd_help() {
  banner
  echo "Usage: ./run.sh [command]"
  echo ""
  echo -e "${CYAN}First time:${NC}"
  echo "  setup          Generate SearXNG secret key, build and start all services"
  echo ""
  echo -e "${CYAN}Core:${NC}"
  echo "  start          Build and start all services"
  echo "  stop           Stop all services"
  echo "  restart        Restart without rebuilding"
  echo "  rebuild        Rebuild after code/config changes"
  echo ""
  echo -e "${CYAN}Research:${NC}"
  echo "  research       Trigger an on-demand research run now"
  echo "  reports        List all saved research reports"
  echo "  latest         Open the latest report"
  echo ""
  echo -e "${CYAN}Info & UI:${NC}"
  echo "  status         Show all containers + last run status"
  echo "  logs           Stream all service logs"
  echo "  open           Open research agent dashboard in browser"
  echo ""
  echo -e "${CYAN}Examples:${NC}"
  echo "  ./run.sh setup          # Run ONCE on first install"
  echo "  ./run.sh start          # Start everything"
  echo "  ./run.sh open           # Open dashboard"
  echo "  ./run.sh research       # Run research right now"
  echo "  ./run.sh latest         # Read latest brief"
  echo ""
}

# ── Router ────────────────────────────────────────────────────────────────────
COMMAND="${1:-help}"

case "$COMMAND" in
  setup)         cmd_setup        ;;
  start)         cmd_start        ;;
  stop)          cmd_stop         ;;
  restart)       cmd_restart      ;;
  rebuild)       cmd_rebuild      ;;
  logs)          cmd_logs         ;;
  status)        cmd_status       ;;
  research)      cmd_research     ;;
  reports)       cmd_reports      ;;
  latest)        cmd_open_latest  ;;
  open)          cmd_open         ;;
  help|--help|-h) cmd_help        ;;
  *) error "Unknown command: '$COMMAND'. Run ./run.sh help" ;;
esac

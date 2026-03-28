# Adding a New Research Agent

The platform is designed to host multiple independent agents, each covering a different research domain. A template is included in `agents/_template/` to make this straightforward.

---

## When to Add a New Agent

Add a new agent when you want to:
- Cover a completely separate research area (e.g. threat intelligence, compliance monitoring, market trends)
- Run on a different schedule from the main research agent
- Generate reports with a different structure or format
- Use a different LLM model for a specific domain

If you just want to add more research topics to the existing agent, add them to `agents/explorer/config.yaml` instead — see [configuration.md](configuration.md).

---

## Step-by-Step Guide

### 1. Copy the template

```bash
cp -r agents/_template agents/<your-agent-name>
```

For example:
```bash
cp -r agents/_template agents/compliance
cp -r agents/_template agents/threat-intel
```

### 2. Edit `config.yaml`

Customize the schedule, research topics, and Ollama settings for your agent:

```yaml
schedule:
  frequency: "daily"
  hour: 10              # Different time from the main agent to avoid Ollama contention
  minute: 0
  timezone: "Asia/Kolkata"
  day_of_week: "mon"
  day: 1

research:
  max_results_per_query: 5
  time_range: "week"
  max_age_months: 3
  fetch_article_content: true
  max_article_chars: 2000
  dedup_against_last_n_reports: 2

  topics:
    - area: "Your Research Domain"
      queries:
        - "specific search query 1"
        - "specific search query 2"
        # Add 6–10 queries per domain
        # More specific queries = better results

report:
  output_dir: "/reports"
  agent_name: "Your Agent Name"
  executive_summary_points: 6
  key_insights_points: 6
  watch_list_points: 4
  format: "markdown"

ollama:
  model: "llama3.1:8b"
  num_predict: 3000
  temperature: 0.3
```

### 3. Customize `agent.py` (optional)

The template `agent.py` is functional as-is and will generate reports based on your `config.yaml` topics. Customize it if you need:

- A different report structure (edit `synthesize_advisory_report()` prompt)
- Additional API endpoints
- Custom data sources beyond SearXNG
- Different deduplication logic

### 4. Add a service block in `docker-compose.yml`

Add your new agent following the existing pattern. The commented-out examples at the bottom of `docker-compose.yml` are ready to fill in:

```yaml
  compliance-agent:
    build:
      context: ./agents/compliance
      dockerfile: Dockerfile
    container_name: agent-compliance
    restart: unless-stopped
    ports:
      - "${COMPLIANCE_PORT:-8889}:8888"
    volumes:
      - ${REPORTS_DIR:-./reports}/compliance:/reports
      - ./agents/compliance/config.yaml:/app/config.yaml
    environment:
      - SEARXNG_URL=http://research-searxng:8080
      - OLLAMA_URL=${OLLAMA_URL:-http://host.docker.internal:11434}
      - CONFIG_PATH=/app/config.yaml
      - PORT=8888
      - RUN_ON_START=${RUN_ON_START:-false}
    extra_hosts:
      - "host.docker.internal:host-gateway"
    networks:
      - agentic-platform
    depends_on:
      searxng:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8888/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s
```

**Key points:**
- Use a unique `container_name` (e.g. `agent-compliance`)
- Use a unique port (e.g. `8889`, `8890`)
- Consider separating reports into subdirectories (e.g. `/reports/compliance`)
- Attach to `agentic-platform` network so it can reach `research-searxng`

### 5. Add the port to `.env`

```bash
# .env
COMPLIANCE_PORT=8889
```

### 6. Update `run.sh` (optional)

Add commands to control your new agent, following the existing pattern:

```bash
cmd_compliance_logs() {
  check_docker
  info "Streaming compliance agent logs..."
  $COMPOSE logs -f --tail=100 compliance-agent
}
```

And add to the router:
```bash
compliance-logs) cmd_compliance_logs ;;
```

### 7. Build and start

```bash
./run.sh rebuild
```

Your new agent dashboard will be at `http://localhost:8889`.

---

## Agent Isolation

Each agent is fully isolated:

- **Separate container** — crashes or restarts don't affect other agents
- **Separate config** — different schedule, topics, model settings
- **Separate reports directory** — reports don't mix unless you intentionally point them at the same path
- **Shared SearXNG** — all agents reuse the same bundled search engine (no duplication)
- **Shared Ollama** — all agents use the same local Ollama instance; schedule agents at different times to avoid simultaneous LLM calls

---

## Scheduling Multiple Agents

If you have multiple agents, space their schedules to avoid both calling Ollama at the same time (Ollama handles concurrent requests but it's slower):

```yaml
# agents/explorer/config.yaml
schedule:
  hour: 9          # Research agent runs at 9:00 AM

# agents/compliance/config.yaml
schedule:
  hour: 10         # Compliance agent runs at 10:00 AM

# agents/threat-intel/config.yaml
schedule:
  hour: 11         # Threat intel agent runs at 11:00 AM
```

---

## Template Contents

```
agents/_template/
├── agent.py          # Minimal Flask app with research pipeline
├── config.yaml       # Starter configuration (edit this)
├── requirements.txt  # Same dependencies as research agent
└── Dockerfile        # Same Python 3.12-slim container
```

The template `agent.py` includes all the core functions from the research agent: `search()`, `fetch_content()`, `call_ollama()`, `gather_findings()`, `deduplicate()`, `synthesize_report()`, and the Flask routes. It's ready to run as-is once you fill in `config.yaml`.

# ScoutForge

> Multi-topic automated web research & intelligence synthesis — powered by local AI, no cloud dependencies.

A self-hosted, schedule-driven research agent that monitors any number of independent topics you define. For each topic it searches the web across your configured query areas, deduplicates against previous reports, and synthesises structured intelligence briefs using a local LLM — automatically, on any schedule you choose.

Built on [SearXNG](https://searxng.github.io/searxng/) (self-hosted search) and [Ollama](https://ollama.com/) (local LLM), served via a Flask web dashboard. Everything runs in Docker. No API keys, no external AI services, no data leaving your network.

---

## What It Does

Every day (or on a schedule you choose), the agent:

1. **Searches the web** across all configured research areas using targeted queries
2. **Extracts article content** from every result
3. **Deduplicates** against previous reports — only new, unique findings are included
4. **Synthesizes** a structured intelligence brief using a local LLM
5. **Saves** the brief as a Markdown file and displays it in a web dashboard

You can also trigger research on-demand, ask questions about past reports, and run deep-dives on any topic — all from the browser.

---

## Who It's For

- **AI product owners and researchers** who need to stay current on model releases, framework updates, and security incidents
- **Security practitioners** tracking AI attack vectors, CVEs, compliance requirements, and governance developments
- **Competitive intelligence teams** monitoring multiple domains simultaneously
- **Anyone** who wants a curated, scheduled briefing on any topic delivered locally

---

## Features

- **Multiple independent topics** — create and manage as many research topics as you need from the dashboard
- **Full topic management UI** — create, configure, and delete topics without touching config files
- **Research Queries editor** — add, edit, and remove research areas and their search queries from the browser
- **Intelligent deduplication** — Ollama compares new findings against previous reports so every brief contains only genuinely new information
- **Local-first** — Ollama (LLM) and SearXNG (search) run locally; no cloud AI or third-party search APIs needed
- **Scheduled runs** — Daily, weekly, or monthly; configurable per topic from the dashboard without restarting
- **Live topic research** — Trigger ad-hoc research on any topic and get a new report instantly
- **Per-report Q&A** — Ask any question scoped to a single report (RAG over that report only)
- **Ask All Reports** — RAG across all saved reports for cross-brief questions
- **Adjustable report depth** — 1-pager summary through full 5-page intelligence report
- **Real-time progress tracking** — Dashboard shows exactly which step is running
- **Prompt injection guardrails** — Two-stage defence (static patterns + LLM semantic check) on every article before it enters the pipeline
- **Skills descriptions** — Plain-English description per topic shown on the dashboard; empty-skill indicator when not yet configured

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Docker: agentic-platform                     │
│                                                                     │
│  ┌─────────────────────────────┐    ┌──────────────────────────┐   │
│  │   ScoutForge Agent          │    │   SearXNG                │   │
│  │   (agent-infoexplorer:8888) │───▶│   (research-searxng:8080)│   │
│  │                             │    │   Google · Bing · DDG    │   │
│  │   Flask Dashboard           │    │   Google News · Bing News│   │
│  │   APScheduler               │    └──────────────────────────┘   │
│  │   Dedup Engine              │                                    │
│  │   Report Writer             │                                    │
│  └──────────┬──────────────────┘                                   │
│             │                                                       │
└─────────────┼───────────────────────────────────────────────────────┘
              │ http://host.docker.internal:11434
              ▼
┌─────────────────────────┐
│   Ollama (Mac host)     │
│   llama3.1:8b           │
│   Dedup · Synthesis     │
│   Q&A · Topic research  │
└─────────────────────────┘

Browser → http://localhost:8888
```

**How the components connect:**
- The **ScoutForge Agent** queries SearXNG over the internal Docker network — SearXNG is never exposed to the host
- SearXNG searches Google, Bing, DuckDuckGo, Google News, and Bing News and returns results in JSON
- Ollama runs natively on the Mac host and is reached via Docker's `host.docker.internal` bridge
- Reports are written to a shared volume and served directly by Flask

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Docker Desktop** | macOS — [download here](https://www.docker.com/products/docker-desktop/) |
| **Ollama** | Installed on the Mac host — [download here](https://ollama.com/) |
| **llama3.1:8b** | The default model — pull with `ollama pull llama3.1:8b` |
| **RAM** | 8 GB minimum recommended (16 GB for comfortable headroom) |
| **Disk** | ~1 GB for containers + model; reports are small Markdown files |

> **Other models:** Any model available in Ollama works. Larger models (e.g. llama3.3:70b on sufficient hardware) produce noticeably better reports. Smaller models (3b, 1b) work but synthesis quality drops.

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/prakaz81/ScoutForge.git
cd ScoutForge
```

### 2. Pull the Ollama model

```bash
ollama pull llama3.1:8b
```

### 3. First-time setup

```bash
./run.sh setup
```

This generates a SearXNG secret key and builds + starts all containers.

### 4. Open the dashboard

```bash
./run.sh open
# → http://localhost:8888
```

### 5. Run your first research brief

Click **▶ Run Full Research Now** on any topic, or:

```bash
./run.sh research
```

---

## Dashboard

Open `http://localhost:8888` in a browser.

### Navigation

The top nav shows one tab per topic. Click **⊕ Topic Mgmt** (always visible on the right) to create, manage, or delete topics.

### Topic Management

- **Create** — Enter a name and click Create. ScoutForge scaffolds the config and skill files immediately; the new tab appears in the nav without a restart.
- **Delete** — Removes the topic, its config, and all its reports.
- **Configure** — After creating a topic, open it and use ⚙️ Settings to fill in its research queries and skill description.

### Settings Modal (per topic)

| Tab | What It Controls |
|---|---|
| **🤖 Model** | Ollama model used for synthesis and Q&A (global) |
| **📋 Skills** | Plain-English description of what the topic monitors (shown on dashboard) |
| **🔍 Research Queries** | Research areas and their search queries — add/edit/remove without restarting |
| **⚙ Topic Settings** | Time range filter, max article age, dedup window |
| **🛡️ Guardrails** | Log of articles blocked by the prompt injection defence |

### Empty Skill Indicator

Topics without a skill description show an amber **●** dot in the nav tab and an orange warning banner on their page. Fill in the Skill to clear it.

### Dashboard Sections

| Section | What It Does |
|---|---|
| **Last Run** | Status badge, timestamp, report name, stats (gathered / unique / deduped) |
| **Research Run** | Trigger full research now; real-time step-by-step progress while running |
| **Ask All Reports** | Type any question — searches last 5 reports and synthesizes an answer |
| **Schedule** | Change frequency (Daily / Weekly / Monthly), time, and day — takes effect immediately |
| **Live Topic Research** | One-off web research on any topic; choose depth (1–5 pages); result saved as a report |
| **Intelligence Reports** | All saved reports with type badges, view button, per-report chat, delete |

### Per-Report Chat

Click the 💬 icon next to any report to open a chat window scoped to that report only.

---

## Report Files

Reports are saved as Markdown files under `./reports/{topic-id}/` with this naming pattern:

```
research_brief_{topic}_{YYYYMMDD_HHMMSS}.md    ← Scheduled full research run
topic_{topic}_{subject}_{YYYYMMDD_HHMMSS}.md   ← On-demand live topic research
product_brief_{name}_{YYYYMMDD_HHMMSS}.md      ← On-demand product research
```

---

## Configuration

### Global engine config — `agents/explorer/config.yaml`

Controls Ollama settings, article fetch limits, and report format. Shared across all topics.

```yaml
ollama:
  model: "llama3.1:8b"
  num_predict: 4000
  temperature: 0.3

research:
  max_results_per_query: 5
  fetch_article_content: true
  max_article_chars: 2000

report:
  output_dir: "/reports"
  executive_summary_points: 6
  key_insights_points: 6
  watch_list_points: 4
```

### Per-topic config — `explorations/{id}/config.yaml`

Controls the schedule, research areas, and queries for each topic. Editable from the dashboard via **⚙️ Settings → Research Queries** and **⚙️ Settings → Topic Settings**.

```yaml
id: my-topic
title: "My Topic"

schedule:
  frequency: "daily"       # daily | weekly | monthly
  hour: 8
  minute: 0
  timezone: "Asia/Kolkata"

research:
  time_range: ""            # "" | "day" | "week" | "month"
  max_age_months: 3
  dedup_against_last_n_reports: 2
  topics:
    - area: "Area Name"
      queries:
        - search query one
        - search query two
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama API endpoint |
| `SEARXNG_URL` | `http://research-searxng:8080` | SearXNG endpoint (internal network) |
| `RESEARCH_PORT` | `8888` | Host port for the dashboard |
| `REPORTS_DIR` | `./reports` | Where reports are saved |
| `RUN_ON_START` | `false` | Set `true` to run research immediately on container start |

---

## Control Script

```bash
./run.sh setup        # First-time: generate SearXNG key, build, start
./run.sh start        # Start all services
./run.sh stop         # Stop all services
./run.sh restart      # Restart without rebuilding
./run.sh rebuild      # Rebuild after code changes
./run.sh logs         # Stream container logs
./run.sh open         # Open dashboard in browser
./run.sh help         # Show all commands
```

---

## Project Structure

```
ScoutForge/
├── agents/
│   └── explorer/
│       ├── agent.py            # Flask app, research engine, all API routes
│       ├── config.yaml         # Global engine config (Ollama, fetch, report format)
│       ├── requirements.txt    # Python dependencies
│       └── Dockerfile
├── explorations/               # Per-topic configs (one directory per topic)
│   ├── ai-world/
│   │   ├── config.yaml         # Schedule, research areas & queries for this topic
│   │   └── skills.md           # Plain-English description shown on dashboard
│   └── default/
│       ├── config.yaml
│       └── skills.md
├── docker/
│   └── searxng/
│       └── settings.yml        # SearXNG config (engines, format, secret key)
├── reports/                    # Generated intelligence briefs (gitignored)
├── docker-compose.yml
├── run.sh
└── README.md
```

---

## Technical Stack

| Component | Technology | Purpose |
|---|---|---|
| Research Engine | Python 3.12 + Flask | Core pipeline, REST API, web dashboard |
| Scheduling | APScheduler (CronTrigger) | Per-topic automated runs |
| Search | SearXNG (self-hosted) | Privacy-respecting multi-engine web search |
| Article Extraction | Trafilatura | Converts web pages to clean text |
| LLM Inference | Ollama (local) | Deduplication, synthesis, Q&A |
| Default Model | llama3.1:8b | Runs on 8 GB RAM |
| Container Orchestration | Docker Compose | Multi-service stack with health checks |
| Report Format | Markdown | Portable, readable, version-controllable |

---

## Troubleshooting

### Dashboard shows "never run" after a container restart

Expected — run status is in-memory. ScoutForge restores it automatically from the most recent report on startup.

### Research run times out

The default Ollama timeout is 300 seconds per call. If synthesis is failing, the model may be too large for available RAM. Try a smaller model (`llama3.2:3b`) or increase Docker's memory allocation.

### SearXNG returns no results

Check that SearXNG started cleanly: `docker ps` and `./run.sh logs`. Results from Google, Bing, and DuckDuckGo are aggregated — if one is temporarily unavailable, others compensate.

### Ollama not reachable

```bash
ollama list
curl http://localhost:11434/api/tags
```

On Linux hosts, replace `host.docker.internal` with your Docker bridge IP (usually `172.17.0.1`).

---

## License

MIT — see `LICENSE`.

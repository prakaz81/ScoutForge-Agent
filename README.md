# InfoExplorer Agent

> Automated web research & intelligence synthesis — powered by local AI, no cloud dependencies.

A self-hosted, schedule-driven research agent that monitors any domain you configure, searches the web across multiple query angles, deduplicates against previous reports, and synthesises structured intelligence briefs using a local LLM — automatically, on any schedule you choose.

Built on [SearXNG](https://searxng.github.io/searxng/) (self-hosted search) and [Ollama](https://ollama.com/) (local LLM), served via a Flask web dashboard. Everything runs in Docker. No API keys, no external AI services, no data leaving your network.

---

## What It Does

Every day (or on a schedule you choose), the agent:

1. **Searches the web** across 9 research domains using ~90 targeted queries
2. **Extracts article content** from every result
3. **Deduplicates** against previous reports — only new, unique findings are included
4. **Synthesizes** a structured intelligence brief using a local LLM
5. **Saves** the brief as a Markdown file and displays it in a web dashboard

You can also trigger research on-demand, ask questions about past reports, and run deep-dives on any topic or vendor — all from the browser.

---

## Who It's For

- **AI product owners and researchers** who need to stay current on model releases, framework updates, and security incidents
- **Security practitioners** tracking AI attack vectors, CVEs, compliance requirements, and governance developments
- **Competitive intelligence teams** monitoring the AI security product landscape
- **Anyone** who wants a daily, curated briefing on the AI world delivered locally

---

## Features

- **9 research domains** covering AI models, agentic AI, ecosystems, frameworks, security incidents, startups, compliance, governance, and competitive landscape
- **Intelligent deduplication** — Ollama compares new findings against previous reports so every brief contains only genuinely new information
- **Local-first** — Ollama (LLM) and SearXNG (search) run locally; no cloud AI or third-party search APIs needed
- **Scheduled runs** — Daily, weekly, or monthly; configurable from the dashboard without restarting
- **Live topic research** — Trigger ad-hoc research on any topic and get a new report instantly
- **Product intelligence** — Deep-dive any vendor, product, or startup across 8 targeted query angles
- **Per-report Q&A** — Ask any question scoped to a single report (RAG over that report only)
- **Ask All Reports** — RAG across all saved reports for cross-brief questions
- **Adjustable report depth** — 1-pager summary through full 5-page intelligence report
- **Real-time progress tracking** — Dashboard shows exactly which domain is being searched
- **Web dashboard** — White-theme responsive UI with run controls, report viewer, schedule manager, and ask interface

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Docker: agentic-platform                     │
│                                                                     │
│  ┌─────────────────────────────┐    ┌──────────────────────────┐   │
│  │   Research Agent            │    │   SearXNG                │   │
│  │   (agent-infoexplorer:8888)     │───▶│   (research-searxng:8080)│   │
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
- The **Research Agent** queries SearXNG over the internal Docker network — SearXNG is never exposed to the host
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

> **Other models:** Any model available in Ollama works. Larger models (e.g. llama3.1:70b on sufficient hardware) produce noticeably better reports. Smaller models (3b, 1b) work but synthesis quality drops.

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-org/InfoExplorerAgent.git
cd InfoExplorerAgent
```

### 2. Pull the Ollama model

```bash
ollama pull llama3.1:8b
```

### 3. Copy and review the environment file

```bash
cp .env.example .env
```

Edit `.env` if needed (port, reports directory, timezone). Defaults work out of the box.

### 4. First-time setup

```bash
./run.sh setup
```

This generates a SearXNG secret key and builds + starts all containers.

### 5. Open the dashboard

```bash
./run.sh open
# → http://localhost:8888
```

### 6. Run your first research brief

Click **▶ Run Full Research Now** in the dashboard, or:

```bash
./run.sh research
```

The run takes 4–6 minutes. Progress is shown live in the dashboard.

---

## Dashboard

Open `http://localhost:8888` in a browser.

```
┌─────────────────────────────────────────────────────────────────────┐
│ 🔭 InfoExplorer Agent           🤖 llama3.1:8b │
├──────────────────────────────────┬──────────────────────────────────┤
│ LAST RUN                         │                                  │
│ ✅ SUCCESS  2026-03-27 09:00     │  Intelligence Reports (12)       │
│  450 gathered · 87 unique · 363  │  newest first                    │
│  dupes removed                   │                                  │
├──────────────────────────────────│  📄 Daily  research_brief_...md  │
│ RESEARCH RUN                     │  📄 Topic  topic_mcp_...md       │
│ [▶ Run Full Research Now]        │  📄 Prod   product_brief_...md   │
│                                  │                                  │
├──────────────────────────────────│  📅 Daily at 09:00 (Asia/Kolkata)│
│ ASK ALL REPORTS (RAG)            │  🤖 llama3.1:8b                  │
│ [What CVEs were disclosed?] [Ask]│  📂 9 domains · 🔄 Dedup last 2  │
│ ┌──────────────────────────────┐ │                                  │
│ │ Answer appears here...       │ └──────────────────────────────────┘
│ └──────────────────────────────┘
├──────────────────────────────────┤
│ SCHEDULE                         │
│ Frequency: [Daily ▾]  Time: [09]:[00]  [💾 Save]                    │
├──────────────────────────────────┤
│ LIVE TOPIC RESEARCH              │
│ [MCP security risks...]  [1-pager▾]  [🔍 Research This Topic]       │
└─────────────────────────────────────────────────────────────────────┘
```

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

Click the 💬 icon next to any report to open a chat window scoped to that report only. Ask specific questions — the LLM answers using only that report's content.

### Skills

Click **📋 Skills** in the header to view or edit the OpenClaw skills file in-browser.

---

## Research Domains

Each run executes ~10 targeted queries per domain:

| # | Domain | What It Tracks |
|---|---|---|
| 1 | **AI Models — Buzz, Releases & Advances** | GPT, Claude, Gemini, Llama, Mistral, Qwen, Grok releases; benchmarks; open source drops; safety incidents |
| 2 | **Agentic AI — Buzz, News & Multi-Agent Systems** | Autonomous agents; multi-agent research; orchestration; open source projects gaining traction |
| 3 | **Agent Ecosystems & Interoperability** | MCP (Model Context Protocol); A2A; agent marketplaces; interoperability standards |
| 4 | **AI Frameworks — Buzz, Releases & Updates** | LangGraph, LangChain, AutoGen, CrewAI, Azure AI Foundry, AWS Bedrock, GCP Vertex AI, N8N, Dify, Flowise, Semantic Kernel, OpenAI Agents SDK |
| 5 | **Agentic Security Products, Startups & Buzz** | Runtime security products; red team tools; AI monitoring; identity and access for agents; funding rounds |
| 6 | **AI Security Incidents, Attacks & Vulnerabilities** | Real-world attacks; CVEs; jailbreaks; prompt injection; agent compromises; data poisoning |
| 7 | **AI Compliance & Regulation** | EU AI Act; NIST AI RMF; OWASP LLM Top 10; MITRE ATLAS; ISO 42001; GDPR; government policy |
| 8 | **AI Governance & Trust** | Governance frameworks; responsible AI; explainability; human oversight; audit standards |
| 9 | **AI Security Products & Competitive Landscape** | Trust control layers; AI governance platforms; compliance monitoring; posture management |

---

## Report Format

Reports are saved as Markdown files in `./reports/` with this naming pattern:

```
research_brief_YYYYMMDD_HHMMSS.md    ← Full domain research run
topic_<slug>_YYYYMMDD_HHMMSS.md     ← On-demand topic research
product_brief_<name>_YYYYMMDD.md    ← On-demand product research
```

Each report has a header with run metadata and this structure:

```markdown
# AI World & Cybersecurity Research Brief
**Date**: Monday, March 27, 2026 — 09:00:12
**Model**: llama3.1:8b | **Topics**: 9 domains | **Search range**: last week
**Articles gathered**: 450 | **Unique new**: 87 | **Duplicates removed**: 363

---

## What's New This Week
- [6 bullet points covering top cross-domain developments]

## Intelligence by Domain
### AI Models — Buzz, Releases & Advances
### Agentic AI — What's New & Buzzing
### Agent Ecosystems & Interoperability
### AI Frameworks & Platforms — What's New
### AI Security Incidents, Attacks & Vulnerabilities ⚠️
### AI Security Products & Startups
### AI Compliance & Regulation
### AI Governance & Trust

## Key Insights & Takeaways
[6 numbered, actionable items]

## Watch List — Signals to Monitor
[4 early signals to track]
```

---

## Configuration

Configuration lives in `agents/explorer/config.yaml`. See [docs/configuration.md](docs/configuration.md) for the full reference.

Key settings:

```yaml
schedule:
  frequency: "daily"          # daily | weekly | monthly
  hour: 9
  minute: 0
  timezone: "Asia/Kolkata"

research:
  max_results_per_query: 5    # Results from SearXNG per query
  time_range: ""              # "" | "day" | "week" | "month"
  max_age_months: 3           # LLM is told to focus on this window
  fetch_article_content: true # Extract full article text (recommended)
  max_article_chars: 2000     # Truncation per article
  dedup_against_last_n_reports: 2

ollama:
  model: "llama3.1:8b"
  num_predict: 3000
  temperature: 0.3
```

The schedule can also be changed live from the dashboard — no restart needed.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama API endpoint |
| `SEARXNG_URL` | `http://research-searxng:8080` | SearXNG endpoint (internal network) |
| `RESEARCH_PORT` | `8888` | Host port for the dashboard |
| `REPORTS_DIR` | `./reports` | Where reports are saved |
| `RUN_ON_START` | `false` | Set `true` to run research immediately on container start |
| `TIMEZONE` | `Asia/Kolkata` | Timezone (for logging; schedule is in config.yaml) |

---

## Control Script

```bash
./run.sh setup        # First-time: generate SearXNG key, build, start
./run.sh start        # Start all services
./run.sh stop         # Stop all services
./run.sh restart      # Restart without rebuilding
./run.sh rebuild      # Rebuild after code/config changes
./run.sh research     # Trigger an on-demand research run
./run.sh reports      # List all saved reports
./run.sh latest       # Open latest report
./run.sh status       # Show container status + last run info
./run.sh logs         # Stream container logs
./run.sh open         # Open dashboard in browser
./run.sh help         # Show all commands
```

---

## Skill Commands

The `skills/InfoExplorerAgentSkills.js` file provides 28 conversational commands for use with OpenClaw or any compatible chat interface that supports custom skills.

See [docs/commands.md](docs/commands.md) for the full command reference.

Quick overview:

| Category | Commands |
|---|---|
| Help | `/help` |
| Daily Intel | `/daily-brief` |
| Research Runs | `/research-run`, `/research-status` |
| Domain Briefs | `/ai-models`, `/ai-agents`, `/ai-ecosystems`, `/ai-frameworks`, `/ai-incidents`, `/ai-security-products`, `/ai-compliance`, `/ai-governance`, `/ai-competitive` |
| Research & Analysis | `/research-ask`, `/research-topic`, `/research-custom [depth]`, `/compete [depth]`, `/research-product [depth]` |
| Report Management | `/research-latest`, `/research-list`, `/research-search`, `/research-report`, `/report-ask` |

---

## Adding a New Research Agent

The project is built to support multiple independent agents on the same Docker network. A template is provided in `agents/_template/`.

See [docs/adding-agents.md](docs/adding-agents.md) for the step-by-step guide.

Short version:
1. Copy `agents/_template/` to `agents/<your-agent-name>/`
2. Edit `config.yaml` with your research topics and queries
3. Add a new service block in `docker-compose.yml` following the existing pattern
4. Run `./run.sh rebuild`

---

## Troubleshooting

### Dashboard shows "never run" after a container restart

This is expected — `_last_run_status` is in-memory. The agent automatically restores status from the most recent report file on startup. If it shows "never run" and reports exist, check that the `REPORTS_DIR` volume is correctly mounted.

### Research run times out

The default Ollama timeout is 300 seconds per call. If synthesis is failing, the model may be too large for available RAM, causing swapping. Try a smaller model (`llama3.2:3b`) or increase Docker's memory allocation.

### SearXNG returns no results

Check that SearXNG started cleanly:
```bash
./run.sh logs
docker ps
```
SearXNG may be blocked by a search engine temporarily. Results from Google, Bing, and DuckDuckGo are aggregated — if one is unavailable, others compensate.

### `url fetch failed` messages in logs

Normal. Some URLs block scrapers. Trafilatura falls back to the SearXNG snippet for those articles. Set `fetch_article_content: false` in `config.yaml` to disable full-text fetching entirely (faster runs, lower quality synthesis).

### Ollama not reachable

Verify Ollama is running on the host:
```bash
ollama list
curl http://localhost:11434/api/tags
```
On Linux hosts, replace `host.docker.internal` with your Docker bridge IP (usually `172.17.0.1`) in `.env`.

### Schedule changes not persisting after container restart

The `config.yaml` volume is mounted writable. Schedule changes from the dashboard are written back to `agents/explorer/config.yaml` on the host. If they aren't persisting, check file permissions:
```bash
ls -la agents/explorer/config.yaml
```

---

## Project Structure

```
InfoExplorerAgent/
├── agents/
│   ├── research/                   # Active research agent
│   │   ├── agent.py                # Flask app, research engine, all API routes
│   │   ├── config.yaml             # Schedule, research domains, Ollama settings
│   │   ├── requirements.txt        # Python dependencies
│   │   └── Dockerfile              # Python 3.12-slim container
│   └── _template/                  # Starter template for adding new agents
│       ├── agent.py
│       ├── config.yaml
│       ├── requirements.txt
│       └── Dockerfile
├── docker/
│   └── searxng/
│       └── settings.yml            # SearXNG config (engines, format, secret key)
├── skills/
│   ├── InfoExplorerAgentSkills.js          # 28 conversational commands
│   └── InfoExplorerAgentSkills.default.js  # Default backup (for in-browser reset)
├── reports/                        # Generated intelligence briefs (gitignored)
├── docs/
│   ├── configuration.md            # Full config.yaml reference
│   ├── commands.md                 # All 28 skill commands
│   ├── architecture.md             # Technical deep dive
│   └── adding-agents.md            # Guide for adding new agents
├── docker-compose.yml              # Service orchestration
├── .env                            # Runtime environment (gitignored)
├── .env.example                    # Template — copy to .env
├── run.sh                          # Control script
└── README.md
```

---

## Technical Stack

| Component | Technology | Purpose |
|---|---|---|
| Research Engine | Python 3.12 + Flask | Core pipeline, REST API, web dashboard |
| Scheduling | APScheduler (CronTrigger) | Daily / weekly / monthly automated runs |
| Search | SearXNG (self-hosted) | Privacy-respecting multi-engine web search |
| Article Extraction | Trafilatura | Converts web pages to clean text |
| LLM Inference | Ollama (local) | Deduplication, synthesis, Q&A |
| Default Model | llama3.1:8b | Runs on 8 GB RAM |
| Container Orchestration | Docker Compose | Multi-service stack with health checks |
| Skill Interface | JavaScript (OpenClaw) | 28 conversational commands |
| Report Format | Markdown | Portable, readable, version-controllable |

---

## Contributing

Contributions welcome. Areas that would benefit most:

- **New research domains** — Add queries to `agents/explorer/config.yaml` under a new `topics` entry
- **Richer report formats** — The `synthesize_advisory_report()` prompt in `agent.py` drives report structure
- **New agent types** — Follow the pattern in `agents/_template/` and `docs/adding-agents.md`
- **Search engine improvements** — `docker/searxng/settings.yml` controls which engines are active
- **Model compatibility** — Testing with different Ollama models and noting quality differences

Please open an issue before large changes to align on direction.

---

## License

MIT — see `LICENSE`.

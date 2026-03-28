# Architecture Deep Dive

---

## Overview

The system is a multi-container Docker application with two active services: the **Research Agent** and **SearXNG**. Ollama runs natively on the host and is accessed via Docker's host-gateway bridge.

```
Host Machine (macOS / Linux)
├── Ollama (native, port 11434)
│     llama3.1:8b — LLM inference
│
└── Docker: agentic-platform (bridge network)
      ├── agent-infoexplorer (port 8888 → host)
      │     Flask app
      │     APScheduler
      │     Research pipeline
      │     Report storage
      │
      └── research-searxng (port 8080, internal only)
            Multi-engine web search
            Google · Bing · DuckDuckGo
            Google News · Bing News
```

---

## Research Pipeline

Each run follows four sequential steps:

### Step 1 — Gather Findings

```
config.yaml topics (9 domains × ~10 queries = ~90 queries)
    │
    ▼
SearXNG /search?format=json (per query, up to 5 results each)
    │
    ▼
Trafilatura.fetch_url() — extract article text (timeout: 8s per URL)
    │
    ▼
Truncate to max_article_chars (default: 2000)
    │
    ▼
all_findings: [ { area, findings: [ {title, url, content, date} ] } ]
```

**Status tracking:** `_last_run_status["step_detail"]` is updated after each domain — the dashboard displays "Domain 3/9: Agentic AI — What's New & Buzzing" etc.

### Step 2 — Load Previous Reports

```
REPORTS_DIR.glob("*.md")
    │ sorted by mtime, take last N (default: 2)
    ▼
Read text of each report (first 4000 chars per report)
    │
    ▼
previous_content: concatenated text block
    │
is_first_report = len(previous_content) == 0
```

### Step 3 — Deduplicate

```
If is_first_report:
    skip → all findings are new

Else:
    Ollama call: extract_covered_topics(previous_content)
        → Prompt: "List all specific topics, events, products already covered"
        → Returns: line-delimited list of covered items

    Ollama call: filter_new_findings(findings_block, covered_topics)
        → Prompt: "Keep findings that are genuinely new; remove already-covered items"
        → Returns: filtered findings block (or "NO_NEW_FINDINGS")

    Count: original articles vs. kept articles → new_count, skipped_count
```

Deduplication uses two LLM calls. This is the most expensive step in terms of tokens but produces significantly better reports by avoiding repetition across daily runs.

### Step 4 — Synthesize & Save

```
Ollama call: synthesize_advisory_report(filtered_findings, ...)
    → ~1000 token prompt + filtered findings block
    → Returns: structured Markdown (9 domain sections + summary + watch list)

Prepend header with metadata (date, model, stats)

Write: REPORTS_DIR / research_brief_YYYYMMDD_HHMMSS.md

Update: _last_run_status (in-memory + shown on dashboard)
```

---

## Flask Application

### Route Map

| Route | Method | Handler | Description |
|---|---|---|---|
| `/` | GET | `dashboard()` | Renders the full dashboard HTML |
| `/api/status` | GET | `api_status()` | Current `_last_run_status` as JSON |
| `/api/run` | POST | `api_run()` | Starts `run_research()` in a background thread |
| `/api/research/product` | POST | `api_product_research()` | Runs `research_product()`, waits up to 300s |
| `/api/research/topic` | POST | `api_topic_research()` | Runs `research_topic()`, waits up to 360s |
| `/api/reports` | GET | `api_reports()` | List all `.md` files in `REPORTS_DIR` |
| `/reports/<filename>` | GET | `view_report()` | Serve report as `text/plain` |
| `/api/ask` | POST | `api_ask_all()` | RAG over last 5 reports |
| `/api/ask/report/<filename>` | POST | `api_ask_report()` | RAG scoped to one report |
| `/api/skill` | GET | `api_skill_get()` | Read skills file content |
| `/api/skill` | POST | `api_skill_save()` | Write skills file content |
| `/api/skill/reset` | POST | `api_skill_reset()` | Copy default backup to active file |
| `/api/schedule` | GET | `api_schedule_get()` | Current schedule config + next run time |
| `/api/schedule` | POST | `api_schedule_post()` | Update schedule; reschedules live APScheduler job |
| `/health` | GET | `health()` | Liveness check |

### State Management

The agent maintains state in two places:

**In-memory (`_last_run_status`):**
```python
{
  "status": "success" | "running" | "error" | "never_run",
  "timestamp": "ISO 8601 string",
  "report": "research_brief_YYYYMMDD_HHMMSS.md",
  "total_articles": 450,
  "new_items": 87,
  "duplicates_removed": 363,
  # During a run:
  "step": "1/4",
  "step_label": "Searching SearXNG across all domains",
  "step_detail": "Domain 3/9: Agentic AI — What's New & Buzzing",
}
```

**On disk (reports/):**
Each report file header contains the same stats in Markdown format. On startup, `restore_status_from_disk()` reads the most recent report and parses the header to restore `_last_run_status` — so the dashboard doesn't show "never run" after a container restart.

### Concurrency Model

- All research runs execute in a background `Thread(daemon=True)` so the Flask server stays responsive
- `_last_run_status` is a plain dict modified from the background thread and read by Flask request handlers — no locks
- On-demand product and topic research use `Thread.join(timeout=N)` to block the request until complete

---

## Scheduler

APScheduler's `BackgroundScheduler` manages the automated run schedule.

```python
_scheduler = None  # module-level, set by start_scheduler()

def start_scheduler():
    global _scheduler
    _scheduler = BackgroundScheduler(timezone=tz)
    _scheduler.add_job(
        run_research,
        _build_cron_trigger(s),    # CronTrigger from config.yaml
        id="daily_research",
        replace_existing=True,
    )
    _scheduler.start()
```

Live rescheduling (from `POST /api/schedule`):
```python
_scheduler.reschedule_job("daily_research", trigger=_build_cron_trigger(new_s))
```

The `_scheduler` module-level variable is what makes live rescheduling work — routes can reach it without passing it around.

---

## On-Demand Research (Product & Topic)

These follow the same pattern as the scheduled run but with targeted queries instead of the full 9-domain sweep.

**Product research** (`research_product(name)`):
- 8 fixed queries built around the product name (features, funding, customers, weaknesses, competitors, roadmap, security)
- Always fetches full article text
- Saves as `product_brief_<name>_YYYYMMDD.md`

**Topic research** (`research_topic(topic, user_context)`):
- First, calls Ollama to generate 6 smart queries for the topic
- Executes those queries against SearXNG
- Synthesizes findings into a structured brief
- Saves as `topic_<slug>_YYYYMMDD.md`

Both functions return a dict with `status`, `report`, `sources`, and `content` — the skill can then re-synthesize the content at the requested depth level.

---

## Q&A (RAG)

### Ask All Reports

```
POST /api/ask  { "question": "..." }
    │
    ▼
Load last 5 reports from REPORTS_DIR (sorted by mtime)
Truncate each to 2500 chars
Concatenate into context block
    │
    ▼
Ollama call:
  system: "Answer using ONLY information from the reports provided"
  user: "REPORTS (5):\n{context}\n\nQUESTION: {question}"
    │
    ▼
Return: { "answer": "...", "reports_searched": ["report1.md", ...] }
```

### Ask Single Report

```
POST /api/ask/report/<filename>  { "question": "..." }
    │
    ▼
Read full report file (up to 8000 chars)
    │
    ▼
Ollama call:
  system: "Answer using ONLY the content of this single report"
  user: "REPORT: {filename}\n\n{content}\n\nQUESTION: {question}"
    │
    ▼
Return: { "answer": "...", "report": "filename.md" }
```

The per-report ask uses 8000 chars (4× the all-reports version) because it's scoped to one file, allowing more complete answers.

---

## Dashboard Rendering

The entire dashboard HTML is a single Python string (`DASHBOARD_HTML`) using Jinja2 template syntax. Flask's `render_template_string()` fills in the variables at request time.

Template variables passed by `dashboard()`:

| Variable | Source | Used For |
|---|---|---|
| `status` | `_last_run_status` | Status badge, stats row |
| `reports` | `REPORTS_DIR.glob("*.md")` | Report list |
| `schedule_desc` | `_describe_schedule(s)` | Config bar, schedule card subtitle |
| `next_run` | `_next_run()` | Schedule card subtitle |
| `schedule_freq/hour/minute/dow/day` | `CFG["schedule"]` | Pre-populate schedule form inputs |
| `model` | `OLLAMA_MODEL` | Header badge, config bar |
| `topic_count` | `len(TOPICS)` | Config bar |
| `dedup_n` | `DEDUP_N` | Config bar |
| `timezone` | `CFG["schedule"]["timezone"]` | Config bar |

### Progress Polling

The browser polls `GET /api/status` every 5 seconds while a run is in progress. The response includes `step`, `step_label`, and `step_detail` which are rendered in the progress indicator.

```javascript
setInterval(checkStatus, 5000);

// On completion (status === "success"):
clearInterval(refreshTimer);
setTimeout(() => location.reload(), 2000);  // Reload to show new report
```

---

## Container Startup Sequence

1. SearXNG starts and waits for health check to pass (`wget /healthz`)
2. Research Agent starts (depends on SearXNG being healthy)
3. Agent loads `config.yaml`
4. `restore_status_from_disk()` — reads latest report file, restores `_last_run_status`
5. `start_scheduler()` — creates APScheduler with CronTrigger from config
6. If `RUN_ON_START=true`, spawns background thread to run `run_research()` immediately
7. Flask starts on `0.0.0.0:8888`

---

## Report Storage

Reports are written to `/reports/` inside the container, which maps to `./reports/` on the host (configurable via `REPORTS_DIR` in `.env`).

The volume is declared writable in `docker-compose.yml`:
```yaml
volumes:
  - ${REPORTS_DIR:-./reports}:/reports
```

Report files accumulate indefinitely. Delete unwanted reports via the dashboard (🗑 button) or directly from the host filesystem.

---

## Adding New Agents

The Docker Compose file includes commented-out blocks for additional agents (`compliance-agent`, `threat-intel-agent`). Each agent:
- Gets its own service block in `docker-compose.yml`
- Runs on the `agentic-platform` network (same as infoexplorer-agent and SearXNG)
- Has its own port, config, and reports directory
- Can share the same SearXNG and Ollama instances

See [adding-agents.md](adding-agents.md) for the complete guide.

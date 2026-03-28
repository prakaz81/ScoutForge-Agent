# Configuration Reference

All configuration lives in `agents/explorer/config.yaml`. The file is mounted writable into the container, so changes take effect on the next run — or immediately for schedule changes made via the dashboard.

---

## Top-Level Structure

```yaml
schedule:   # When to run automatically
research:   # What to search for and how
report:     # How reports are written
ollama:     # LLM settings
```

---

## `schedule`

Controls when the automated research run fires.

```yaml
schedule:
  frequency: "daily"        # daily | weekly | monthly
  hour: 9                   # 0–23
  minute: 0                 # 0–59
  timezone: "Asia/Kolkata"  # Any pytz timezone string
  day_of_week: "mon"        # Used when frequency = weekly
                            # mon | tue | wed | thu | fri | sat | sun
  day: 1                    # Used when frequency = monthly (1–28)
```

### Changing the schedule without restarting

The dashboard's **Schedule** card lets you change frequency, time, and day live. It calls `POST /api/schedule`, which:
1. Validates the input
2. Updates APScheduler's in-memory cron trigger immediately
3. Writes the new values back to `config.yaml` for persistence across restarts

### Timezone reference

Use any [pytz-compatible timezone string](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones), e.g.:
- `Asia/Kolkata` — India Standard Time (IST, UTC+5:30)
- `America/New_York` — Eastern Time
- `Europe/London` — GMT/BST
- `UTC` — Coordinated Universal Time

---

## `research`

Controls how the search pipeline behaves.

```yaml
research:
  max_results_per_query: 5      # How many search results to fetch per query
  time_range: ""                # Filter results by age: "day" | "week" | "month" | ""
  max_age_months: 3             # Prompt instruction: focus on news within N months
  fetch_article_content: true   # Whether to extract full article text
  max_article_chars: 2000       # Truncation limit per article (characters)
  dedup_against_last_n_reports: 2  # Compare new findings against last N reports
```

### `max_results_per_query`

Each topic has ~10 queries. With 9 topics, that's ~90 queries per run. At `max_results_per_query: 5`, a full run gathers up to 450 articles. Increasing to `10` doubles coverage but roughly doubles run time.

### `time_range`

Passed directly to SearXNG. Restricts results to articles published within the given window. Empty string means no filter — SearXNG returns broadly relevant results.

| Value | Meaning |
|---|---|
| `""` | No filter (recommended for weekly/monthly runs) |
| `"day"` | Last 24 hours |
| `"week"` | Last 7 days |
| `"month"` | Last 30 days |

### `max_age_months`

Even with an empty `time_range`, the LLM synthesis prompt instructs the model to focus only on developments within this many months. Acts as a soft filter — older content is ignored during synthesis.

### `fetch_article_content`

When `true`, the agent fetches each article URL and extracts clean text using Trafilatura. This produces significantly better synthesis quality because the LLM sees actual content rather than SearXNG snippets.

Set to `false` for faster runs (2–3 minutes instead of 4–6) at the cost of synthesis quality.

### `dedup_against_last_n_reports`

Before synthesizing, the agent asks Ollama to compare new findings against the last N reports and remove duplicate topics. A value of `2` checks the last two reports. Set to `0` to disable deduplication entirely (first run always behaves as if `0`).

---

## `research.topics`

Defines the research domains and queries. Each topic has:

```yaml
topics:
  - area: "Human-readable domain name"
    queries:
      - "specific search query 1"
      - "specific search query 2"
      # ...up to ~10 queries
```

### The 9 default domains

| Domain | Focus |
|---|---|
| AI Models — Buzz, Releases & Advances | Model releases, benchmarks, controversies, safety |
| Agentic AI — Buzz, News & Multi-Agent Systems | Autonomous agents, multi-agent systems, orchestration |
| Agent Ecosystems & Interoperability | MCP, A2A, protocols, marketplaces, standards |
| AI Frameworks — Buzz, Releases & Updates | LangGraph, AutoGen, Azure, AWS, GCP, N8N, Flowise |
| Agentic Security Products, Startups & Buzz | Security product launches, runtime protection, funding |
| AI Security Incidents, Attacks & Vulnerabilities | CVEs, attacks, jailbreaks, prompt injection |
| AI Compliance & Regulation | EU AI Act, NIST, OWASP, MITRE ATLAS, ISO 42001 |
| AI Governance & Trust | Governance frameworks, responsible AI, transparency |
| AI Security Products & Competitive Landscape | Trust layers, governance platforms, posture management |

### Adding a new domain

Append a new `- area:` block to the `topics` list:

```yaml
    - area: "Quantum AI & Post-Quantum Cryptography"
      queries:
        - "quantum AI research breakthrough 2025 2026"
        - "post-quantum cryptography AI model protection"
        - "quantum machine learning new paper announcement"
        - "quantum computing threat AI encryption news"
        - "NIST post-quantum standard AI implementation"
```

The agent automatically picks it up on the next run — no code change needed.

### Removing a domain

Delete the `- area:` block. The removed domain will no longer appear in reports.

---

## `report`

Controls report formatting and output location.

```yaml
report:
  output_dir: "/reports"               # Path inside the container
  agent_name: "InfoExplorer Agent"
  executive_summary_points: 6         # Bullets in "What's New This Week"
  key_insights_points: 6              # Numbered items in "Key Insights"
  watch_list_points: 4                # Items in "Watch List"
  format: "markdown"                  # Currently only markdown is supported
```

### `output_dir`

This is the path **inside the container**. It maps to the host path via the Docker volume:

```yaml
# docker-compose.yml
volumes:
  - ${REPORTS_DIR:-./reports}:/reports
```

Change `REPORTS_DIR` in `.env` to save reports somewhere else on the host.

### Adjusting report length

Increasing `executive_summary_points`, `key_insights_points`, or `watch_list_points` instructs the LLM to include more items in those sections. Keep in mind that `ollama.num_predict` limits total token output — if you increase bullet counts significantly, also increase `num_predict`.

---

## `ollama`

LLM settings for the locally-running Ollama instance.

```yaml
ollama:
  model: "llama3.1:8b"    # Any model available in your Ollama installation
  num_predict: 3000        # Maximum tokens to generate per call
  temperature: 0.3         # 0.0 = deterministic, 1.0 = creative
```

### Choosing a model

| Model | RAM Required | Quality | Speed |
|---|---|---|---|
| `llama3.2:1b` | ~1 GB | Basic | Very fast |
| `llama3.2:3b` | ~3 GB | Acceptable | Fast |
| `llama3.1:8b` | ~8 GB | Good (default) | Moderate |
| `llama3.3:70b` | ~40 GB | Excellent | Slow |
| `mistral:7b` | ~6 GB | Good | Moderate |
| `qwen2.5:14b` | ~14 GB | Very good | Moderate |

The model must be pre-pulled on the host before starting the container:
```bash
ollama pull llama3.1:8b
```

### `num_predict`

Controls the maximum length of each LLM generation. The synthesis prompt generates the longest output. At `3000` tokens with `llama3.1:8b`, the main report covers all 9 domains with reasonable depth. Increase to `5000–8000` for more detailed reports (paired with larger `executive_summary_points`).

### `temperature`

`0.3` is deliberately low to keep reports factual and reproducible. Increasing it to `0.5–0.7` produces more varied phrasing but may introduce more hallucination.

---

## Environment Variables

Set in `.env` (copy from `.env.example`). These override nothing in `config.yaml` — they configure the container environment.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama API base URL |
| `SEARXNG_URL` | `http://research-searxng:8080` | Internal SearXNG URL (Docker network) |
| `RESEARCH_PORT` | `8888` | Host port the dashboard is served on |
| `REPORTS_DIR` | `./reports` | Host directory for saved reports |
| `RUN_ON_START` | `false` | `true` triggers a research run immediately when the container starts |
| `TIMEZONE` | `Asia/Kolkata` | Used in log timestamps |

### Linux hosts

On Linux, Docker's `host.docker.internal` may not resolve. Replace with your Docker bridge IP:

```bash
# Find your bridge IP
docker network inspect bridge | grep Gateway
```

Then set in `.env`:
```
OLLAMA_URL=http://172.17.0.1:11434
```

---

## SearXNG Settings

`docker/searxng/settings.yml` configures the bundled SearXNG instance.

The `secret_key` is generated automatically by `./run.sh setup`. You can also set it manually:
```bash
openssl rand -hex 32
```

The `formats: [html, json]` entry is **required** — the agent calls SearXNG's `/search?format=json` endpoint.

`limiter: false` disables rate limiting because SearXNG's default limiter throttles clients making many rapid requests, which the agent does during a research run.

To add or remove search engines, edit the `engines:` section. See the [SearXNG documentation](https://docs.searxng.org/admin/settings/settings_engines.html) for the full engine list.

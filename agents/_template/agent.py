"""
InfoExplorer Agent — Template
Copy this file to agents/<your-agent-name>/ and customise config.yaml.
See docs/adding-agents.md for the full guide.
"""

import os
import re
import logging
import requests
import yaml
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Thread

import trafilatura
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from flask import Flask, jsonify, request, render_template_string

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "/app/config.yaml"))
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://host.docker.internal:8080")
OLLAMA_URL  = os.getenv("OLLAMA_URL",  "http://host.docker.internal:11434")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

OLLAMA_MODEL      = CFG["ollama"]["model"]
OLLAMA_PREDICT    = CFG["ollama"]["num_predict"]
OLLAMA_TEMP       = CFG["ollama"]["temperature"]
REPORTS_DIR       = Path(CFG["report"]["output_dir"])
TOPICS            = CFG["research"]["topics"]
MAX_RESULTS       = CFG["research"]["max_results_per_query"]
TIME_RANGE        = CFG["research"]["time_range"]
FETCH_CONTENT     = CFG["research"]["fetch_article_content"]
MAX_ARTICLE_CHARS = CFG["research"]["max_article_chars"]
DEDUP_N           = CFG["research"]["dedup_against_last_n_reports"]
AGENT_NAME        = CFG["report"]["agent_name"]
EXEC_BULLETS      = CFG["report"]["executive_summary_points"]
INSIGHT_BULLETS   = CFG["report"]["key_insights_points"]
WATCH_BULLETS     = CFG["report"]["watch_list_points"]

app = Flask(__name__)
_last_run_status = {"status": "never_run", "timestamp": None, "report": None}

# ── SearXNG Search ───────────────────────────────────────────────────────────

def search(query: str) -> list[dict]:
    """Query SearXNG and return results list."""
    try:
        resp = requests.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json",
                "categories": "general,news",
                "time_range": TIME_RANGE,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])[:MAX_RESULTS]
    except Exception as e:
        log.warning(f"SearXNG search failed for '{query}': {e}")
        return []


def fetch_content(url: str) -> str | None:
    """Extract readable text from a URL using trafilatura."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
            )
            if text:
                return text[:MAX_ARTICLE_CHARS]
    except Exception as e:
        log.debug(f"Content fetch failed for {url}: {e}")
    return None


# ── Ollama Synthesis ─────────────────────────────────────────────────────────

def call_ollama(prompt: str, system: str = "") -> str:
    """Send prompt to Ollama and return generated text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/v1/chat/completions",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "temperature": OLLAMA_TEMP,
                "max_tokens": OLLAMA_PREDICT,
            },
            headers={"Authorization": "Bearer ollama-local"},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.error(f"Ollama call failed: {e}")
        return f"[ERROR: Ollama synthesis failed — {e}]"


# ── Research Run ─────────────────────────────────────────────────────────────

def gather_findings() -> list[dict]:
    """Search all topics and collect raw findings."""
    all_findings = []

    for topic in TOPICS:
        area  = topic["area"]
        queries = topic["queries"]
        log.info(f"  Researching: {area}")

        area_findings = []
        for query in queries:
            results = search(query)
            for r in results:
                url     = r.get("url", "")
                snippet = r.get("content", "")
                content = fetch_content(url) if FETCH_CONTENT and url else None

                area_findings.append({
                    "title":   r.get("title", "Untitled"),
                    "url":     url,
                    "snippet": snippet,
                    "content": content or snippet,
                    "date":    r.get("publishedDate", ""),
                    "query":   query,
                })

        all_findings.append({"area": area, "findings": area_findings})
        log.info(f"    → {sum(len(t['findings']) for t in all_findings if t['area']==area)} articles found")

    return all_findings


def build_findings_block(all_findings: list[dict]) -> str:
    """Format findings into a text block for the LLM prompt."""
    block = ""
    for topic in all_findings:
        block += f"\n\n### AREA: {topic['area']}\n"
        for f in topic["findings"]:
            block += f"\n**{f['title']}**\n"
            if f["date"]:
                block += f"Published: {f['date']}\n"
            block += f"URL: {f['url']}\n"
            block += f"Content: {f['content'][:800]}\n"
            block += "---\n"
    return block


def synthesize(all_findings: list[dict], run_time: datetime) -> str:
    """Ask Ollama to produce a structured briefing report from raw findings."""
    findings_block = build_findings_block(all_findings)
    total_articles = sum(len(t["findings"]) for t in all_findings)

    prompt = f"""You are a senior AI Security researcher and analyst specializing in Agentic AI systems.

Today is {run_time.strftime('%Y-%m-%d')}. You have gathered {total_articles} research articles across {len(all_findings)} topic areas.

Below are the raw research findings for today:
{findings_block}

---

Based ONLY on the findings above, produce a professional intelligence briefing report with EXACTLY this structure:

## Executive Summary
(Exactly {EXEC_BULLETS} bullet points. Each bullet = one specific, concrete insight from today's findings. No generalities. Lead with the most impactful finding.)

## Key Findings by Area

### Agentic AI Attack Techniques & Exploits
(Summarize specific new attacks, CVEs, exploits, or research published today. Be precise — name tools, techniques, papers.)

### Agentic AI Security Architecture & Defense
(Summarize new defensive approaches, architectures, best practices, or tools published today.)

### AI Compliance & Governance
(Summarize regulatory updates, compliance guidance, policy announcements relevant to agentic AI.)

### Agentic AI Threat Intelligence
(Summarize threat reports, incidents, attacker TTPs related to AI agents.)

### Frameworks, SDKs & Platforms
(Summarize new releases, updates, security patches, or announcements for agentic AI frameworks.)

### Vendor, Startup & Industry News
(Summarize notable product launches, funding rounds, acquisitions, or enterprise announcements.)

### AI Buzz, Trends & Emerging Themes
(Summarize what the community is discussing, debating, or hyping today in the agentic AI space.)

## Watch List
(Exactly {WATCH_BULLETS} numbered items. Each item: bold title + 2-sentence explanation of WHY it matters and WHAT specifically to watch.)

---
*If a section has no relevant findings today, write: "No significant findings for this period."*
*Be specific. Name vendors, frameworks, CVEs, researchers, papers where possible.*
"""

    log.info("Synthesizing report with Ollama...")
    return call_ollama(prompt)


def save_report(content: str, run_time: datetime, total_articles: int) -> Path:
    """Save the final report as a timestamped markdown file."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"agentic_ai_brief_{run_time.strftime('%Y%m%d_%H%M%S')}.md"
    filepath = REPORTS_DIR / filename

    header = f"""# Agentic AI Security Intelligence Brief
**Date**: {run_time.strftime('%A, %B %d, %Y — %H:%M:%S')}
**Model**: {OLLAMA_MODEL}
**Topics**: {len(TOPICS)} research areas
**Articles Analyzed**: {total_articles}
**Search Range**: Last {TIME_RANGE}

---

"""
    with open(filepath, "w") as fh:
        fh.write(header + content)

    log.info(f"Report saved → {filepath}")
    return filepath


def run_research() -> dict:
    """Full research pipeline: search → synthesize → save."""
    global _last_run_status
    run_time = datetime.now()

    log.info("=" * 60)
    log.info(f"Research run started at {run_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    _last_run_status = {"status": "running", "timestamp": run_time.isoformat(), "report": None}

    try:
        findings      = gather_findings()
        total_articles = sum(len(t["findings"]) for t in findings)
        log.info(f"Total articles gathered: {total_articles}")

        report_body   = synthesize(findings, run_time)
        filepath      = save_report(report_body, run_time, total_articles)

        _last_run_status = {
            "status":    "success",
            "timestamp": run_time.isoformat(),
            "report":    filepath.name,
        }
        log.info("Research run complete.")
        return _last_run_status

    except Exception as e:
        log.error(f"Research run failed: {e}", exc_info=True)
        _last_run_status = {
            "status":    "error",
            "timestamp": run_time.isoformat(),
            "report":    None,
            "error":     str(e),
        }
        return _last_run_status


# ── Flask API + UI ────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>Agentic AI Research Agent</title>
  <meta charset="utf-8">
  <style>
    body { font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #0f1117; color: #e2e8f0; }
    h1 { color: #7dd3fc; border-bottom: 1px solid #334155; padding-bottom: 12px; }
    h2 { color: #94a3b8; font-size: 1rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 32px; }
    .card { background: #1e293b; border-radius: 10px; padding: 20px; margin: 12px 0; }
    .status { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
    .status.success { background: #064e3b; color: #6ee7b7; }
    .status.running { background: #1e3a5f; color: #7dd3fc; }
    .status.error   { background: #450a0a; color: #fca5a5; }
    .status.never_run { background: #374151; color: #9ca3af; }
    button { background: #3b82f6; color: white; border: none; padding: 10px 24px; border-radius: 8px;
             font-size: 1rem; cursor: pointer; margin-right: 8px; }
    button:hover { background: #2563eb; }
    button.danger { background: none; border: 1px solid #ef4444; color: #ef4444; }
    a { color: #7dd3fc; text-decoration: none; }
    a:hover { text-decoration: underline; }
    ul { list-style: none; padding: 0; }
    li { padding: 8px 12px; border-bottom: 1px solid #1e293b; display: flex; justify-content: space-between; }
    li:last-child { border-bottom: none; }
    .report-name { font-family: monospace; font-size: 0.9rem; }
    #run-result { margin-top: 12px; color: #6ee7b7; display: none; }
    pre { background: #0f172a; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 0.8rem; color: #94a3b8; }
  </style>
</head>
<body>
  <h1>Agentic AI Security Research Agent</h1>

  <div class="card">
    <h2>Status</h2>
    <span class="status {{ status.status }}">{{ status.status | upper }}</span>
    {% if status.timestamp %}<span style="margin-left:12px; color:#64748b; font-size:0.9rem;">Last run: {{ status.timestamp }}</span>{% endif %}
    {% if status.report %}<span style="margin-left:12px; color:#6ee7b7; font-size:0.9rem;">→ {{ status.report }}</span>{% endif %}
  </div>

  <div class="card">
    <h2>Controls</h2>
    <button onclick="triggerRun()">Run Research Now</button>
    <div id="run-result">Research started in background. Check status in a moment.</div>
  </div>

  <div class="card">
    <h2>Reports ({{ reports|length }} total)</h2>
    <ul>
      {% for r in reports %}
      <li>
        <span class="report-name">{{ r }}</span>
        <a href="/reports/{{ r }}" target="_blank">View</a>
      </li>
      {% else %}
      <li><span style="color:#64748b;">No reports yet. Run research to generate the first one.</span></li>
      {% endfor %}
    </ul>
  </div>

  <div class="card">
    <h2>Schedule</h2>
    <pre>Daily at {{ schedule_time }} | Search range: {{ time_range }} | Model: {{ model }} | Topics: {{ topic_count }}</pre>
  </div>

  <script>
    async function triggerRun() {
      document.getElementById('run-result').style.display = 'block';
      await fetch('/api/run', { method: 'POST' });
    }
    setTimeout(() => location.reload(), 30000);  // auto-refresh every 30s
  </script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    reports = sorted(
        [p.name for p in REPORTS_DIR.glob("*.md")],
        reverse=True,
    ) if REPORTS_DIR.exists() else []

    schedule_cfg = CFG.get("schedule", {})
    schedule_time = f"{schedule_cfg.get('hour', 7):02d}:{schedule_cfg.get('minute', 0):02d}"

    return render_template_string(
        DASHBOARD_HTML,
        status=_last_run_status,
        reports=reports,
        schedule_time=schedule_time,
        time_range=TIME_RANGE,
        model=OLLAMA_MODEL,
        topic_count=len(TOPICS),
    )


@app.route("/api/status")
def api_status():
    return jsonify(_last_run_status)


@app.route("/api/run", methods=["POST"])
def api_run():
    """Trigger a research run in the background."""
    Thread(target=run_research, daemon=True).start()
    return jsonify({"status": "started", "message": "Research run started in background."})


@app.route("/api/reports")
def api_reports():
    reports = sorted(
        [p.name for p in REPORTS_DIR.glob("*.md")],
        reverse=True,
    ) if REPORTS_DIR.exists() else []
    return jsonify({"count": len(reports), "reports": reports})


@app.route("/reports/<filename>")
def view_report(filename: str):
    """Serve a report file as plain text."""
    filepath = REPORTS_DIR / filename
    if not filepath.exists() or not filepath.suffix == ".md":
        return "Report not found", 404
    return filepath.read_text(), 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": OLLAMA_MODEL, "searxng": SEARXNG_URL})


# ── Scheduler ─────────────────────────────────────────────────────────────────

def start_scheduler():
    schedule_cfg = CFG.get("schedule", {})
    hour   = schedule_cfg.get("hour", 7)
    minute = schedule_cfg.get("minute", 0)
    tz     = schedule_cfg.get("timezone", "UTC")

    scheduler = BackgroundScheduler(timezone=tz)
    scheduler.add_job(
        run_research,
        CronTrigger(hour=hour, minute=minute, timezone=tz),
        id="daily_research",
        name="Daily Agentic AI Research",
        replace_existing=True,
    )
    scheduler.start()
    log.info(f"Scheduler started — daily run at {hour:02d}:{minute:02d} ({tz})")
    return scheduler


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scheduler = start_scheduler()

    run_on_start = os.getenv("RUN_ON_START", "false").lower() == "true"
    if run_on_start:
        log.info("RUN_ON_START=true — running research immediately on startup.")
        Thread(target=run_research, daemon=True).start()

    port = int(os.getenv("PORT", 8888))
    log.info(f"Dashboard available at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

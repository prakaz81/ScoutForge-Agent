# Skill Commands Reference

`InfoExplorerAgentSkills.js` provides 28 commands for interacting with the research agent through a conversational interface (OpenClaw or any compatible chat platform that supports custom skills).

---

## Setup

The skill file is located at `skills/InfoExplorerAgentSkills.js`. It reads three environment variables:

| Variable | Default | Description |
|---|---|---|
| `RESEARCH_AGENT_URL` | `http://agent-infoexplorer:8888` | URL of the research agent container |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `llama3.2` | Model to use for skill-side synthesis |

You can also view and edit the skill file directly from the dashboard by clicking **📋 Skills**.

---

## Depth Levels

Several commands accept an optional `[depth]` parameter (1–5):

| Depth | Length | Use Case |
|---|---|---|
| `1` | ~300 words (default) | Quick executive summary |
| `2` | ~600 words | Brief with supporting context |
| `3` | ~900 words | Detailed analysis with examples |
| `4` | ~1200 words | Comprehensive, all significant findings |
| `5` | ~1500 words | Full intelligence report, every detail |

---

## Command Reference

### HELP

#### `/help`
Lists all available commands with brief descriptions.

```
/help
```

---

### DAILY INTEL

#### `/daily-brief`
Your morning intelligence briefing. Synthesizes the 2 most recent reports into a structured brief across 7 sections:
- 🚨 Incidents & Threats
- 🔥 What's Hot Right Now
- 🏁 Competitor & Market Moves
- 📋 Compliance & Regulatory Pulse
- 🛠 Framework & Platform Updates
- 💡 Strategic Implications (3 actionable items)
- 📡 Signals to Watch

**Best used:** First thing each morning to get up to speed quickly.

```
/daily-brief
```

---

### RESEARCH RUNS

#### `/research-run`
Triggers a full research run across all 9 domains. The run happens in the background — use `/research-status` to check progress, or watch the dashboard.

**Takes:** 4–6 minutes.

```
/research-run
```

#### `/research-status`
Shows the current status of the last or active research run.

**Returns:** status, last run timestamp, report filename, articles gathered, unique new findings, duplicates removed, and any error.

```
/research-status
```

---

### DOMAIN BRIEFS

Domain briefs extract and summarize information about one specific area from the 3 most recent reports. They do not trigger a new web search — they synthesize from already-saved reports.

**Best used:** When you want a focused view on one area without running a full brief.

#### `/ai-models`
AI model releases, benchmarks, buzz, controversies. Covers GPT, Claude, Gemini, Llama, Mistral, Qwen, Grok, SLMs, reasoning models, multimodal models, and open source drops.

#### `/ai-agents`
Agentic AI news — autonomous agents, multi-agent systems, agent frameworks, orchestration approaches, memory, planning, and open source agent projects gaining traction.

#### `/ai-ecosystems`
Agent ecosystem developments — MCP (Model Context Protocol), A2A (Agent-to-Agent), interoperability protocols, agent marketplaces, plugin registries, and emerging standards.

#### `/ai-frameworks`
Framework and platform updates — Azure AI Foundry, Google Vertex AI, AWS Bedrock, N8N, Dify, Flowise, LangGraph, LangChain, AutoGen, CrewAI, Semantic Kernel, OpenAI Agents SDK, Anthropic, Salesforce Agentforce.

#### `/ai-incidents`
⚠️ AI security incidents, attacks, and vulnerabilities. Each finding is flagged with severity. Covers CVEs, jailbreaks, prompt injection, agent compromises, data breaches, and exploits.

#### `/ai-security-products`
AI security product landscape — new product launches, funding rounds, new entrants in the agent security and trust space.

#### `/ai-compliance`
AI compliance and regulation — EU AI Act, NIST AI RMF, OWASP LLM Top 10, MITRE ATLAS, ISO 42001, GDPR AI enforcement, government policy, and compliance deadlines.

#### `/ai-governance`
AI governance and trust — governance frameworks, responsible AI standards, explainability, transparency, human oversight, audit standards, and red team guidance.

#### `/ai-competitive`
Who is building in the AI security, trust, and governance space — maps company positioning, funding, and momentum. Focus on products serving agentic AI security, compliance, and governance.

---

### RESEARCH & ANALYSIS

These commands trigger active processing — either synthesizing from saved reports or running a new live web search.

#### `/research-ask <question>`
Ask any question across the 5 most recent reports (RAG over saved reports). The LLM answers using only information found in those reports and cites which report contains the answer.

```
/research-ask What prompt injection techniques were disclosed this month?
/research-ask Which AI governance frameworks were released recently?
/research-ask What funding rounds happened in agentic AI security?
```

#### `/research-topic <topic>`
Deep-dive any topic across the 10 most recent reports. Produces a structured breakdown covering what's known, key developments, strategic relevance, and what to watch next.

```
/research-topic MCP security risks
/research-topic EU AI Act enforcement timeline
/research-topic multi-agent orchestration patterns
```

#### `/research-custom <topic> [depth]`
**Live web research** — searches the web for the given topic right now and generates a new report. Results are saved to the reports directory.

Takes 2–4 minutes. Depth controls report length (1–5, default 1).

```
/research-custom A2A protocol enterprise adoption
/research-custom AI agent red team techniques 3
/research-custom post-quantum cryptography AI models 5
```

#### `/compete <vendor> [depth]`
Competitive analysis of any AI security vendor or product. Searches the web for current information and produces a structured brief with:
- One-sentence positioning
- Strengths and weaknesses
- Market position (funding, customers, traction)
- Key differentiators
- Signals to watch
- Threat level (Low / Medium / High)

Depth controls report length (1–5, default 1).

```
/compete Microsoft Purview AI
/compete Salesforce Agentforce 3
/compete any AI governance platform
```

#### `/research-product <name> [depth]`
Full product intelligence brief for any AI product, platform, or company. Produces:
- What it is (one-sentence positioning)
- Key capabilities (specific features)
- Target market and known customers
- Business momentum (funding, investors, team)
- Strengths and weaknesses
- Competitive landscape
- Verdict

Depth controls report length (1–5, default 1).

```
/research-product N8N
/research-product AWS Bedrock Agents 3
/research-product Google Vertex AI Agent Builder 5
```

---

### REPORT MANAGEMENT

#### `/research-latest`
Summarizes the most recent report — executive summary bullets, one key highlight per domain, and top 3 action items.

```
/research-latest
```

#### `/research-list`
Lists all saved reports (newest first, up to 20). Reports are numbered — use the number with `/research-report` and `/report-ask`.

```
/research-list
```

Returns output like:
```
1. research_brief_20260327_090012.md
2. topic_mcp_security_20260326_143022.md
3. product_brief_microsoft_purview_20260325.md
...
```

#### `/research-search <keyword>`
Filters the report list by keyword in the filename. Useful for finding topic or product reports.

```
/research-search topic
/research-search product
/research-search mcp
/research-search 20260327
```

#### `/research-report <number>`
Reads and summarizes a specific report by number (from `/research-list` or `/research-search`).

```
/research-report 1
/research-report 3
```

#### `/report-ask <number> <question>`
Ask a specific question scoped to one report. The LLM answers using **only** that report's content. If the answer isn't in the report, it says so clearly.

This is RAG at the single-report level — precise and fast compared to searching all reports.

```
/report-ask 1 What CVEs were disclosed this period?
/report-ask 2 Which AI frameworks shipped new releases?
/report-ask 3 What were the top 3 agentic AI incidents?
```

---

## Usage Tips

**Morning workflow:**
```
/daily-brief                         → Overview across all domains
/ai-incidents                        → Focus on anything security-critical
/ai-competitive                      → Market moves
```

**Deep research on a topic:**
```
/research-custom <topic>             → Live search if topic is new
/research-ask <question>             → If you think it's in recent reports
/research-topic <topic>              → Cross-report synthesis
```

**Competitive intelligence:**
```
/compete <vendor>                    → Quick 1-pager
/compete <vendor> 3                  → Detailed analysis
/research-product <product> 5        → Full product brief
```

**Working with a specific report:**
```
/research-list                       → Find the report number
/research-report 2                   → Read the summary
/report-ask 2 your question here     → Ask specific questions
```

---
name: infoexplorer-agent
display_name: InfoExplorer Agent
version: 1.0.0
---

A self-hosted, schedule-driven research agent that monitors any domain you configure — searching the web, deduplicating against previous reports, and synthesising structured intelligence briefs using a local LLM. No cloud AI, no external APIs, no data leaving your network.

## Research Domains

Configure your research domains and queries in `config.yaml`. The default configuration monitors the AI and cybersecurity world:

- **AI Models** — releases, benchmarks, buzz across GPT, Claude, Gemini, Llama and the open source ecosystem
- **Agentic AI** — autonomous agents, multi-agent systems, orchestration, open source projects
- **Agent Ecosystems** — MCP, A2A protocols, agent marketplaces, interoperability standards
- **AI Frameworks** — LangGraph, AutoGen, Azure AI Foundry, AWS Bedrock, N8N, CrewAI and more
- **AI Security Incidents** — CVEs, jailbreaks, prompt injection, agent compromises
- **AI Security Products** — startup launches, funding rounds, product announcements
- **AI Compliance & Regulation** — EU AI Act, NIST, OWASP, MITRE ATLAS, ISO 42001
- **AI Governance & Trust** — frameworks, responsible AI, transparency, human oversight
- **AI Security Landscape** — trust control layers, governance platforms, posture management

## Dashboard Capabilities

- **Run Full Research** — trigger a research run across all configured domains on demand
- **Ask All Reports** — ask any question across all saved reports (RAG over last 5 reports)
- **Per-Report Chat** — open any report and ask questions scoped to that report only
- **Live Topic Research** — ad-hoc web research on any topic; result saved as a new report
- **Schedule Management** — set daily, weekly, or monthly automated runs from the dashboard
- **Settings** — configure the Ollama model and edit this skills definition

## Customising Research Topics

To change what InfoExplorer Agent monitors, edit the `topics` section in `agents/explorer/config.yaml`:

```yaml
topics:
  - area: "Your Research Domain"
    queries:
      - "specific search query 1"
      - "specific search query 2"
```

Add as many domains and queries as needed. Changes take effect on the next research run — no rebuild required.

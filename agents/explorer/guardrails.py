"""
Prompt injection guardrails for ScoutForge.

This module is the single authoritative source for all injection-detection
logic.  Import it wherever user-controlled text or fetched content is about
to be forwarded to the LLM.

Public API
----------
check_user_input(text)           -> (blocked: bool, reason: str)
    Screens any string that comes from the user (chat questions, topic names,
    research goals, ad-hoc context, etc.) before it enters an LLM call.

check_article_static(article)    -> (blocked: bool, reason: str)
    Stage-1 (zero-latency) static check for fetched web articles.
    Used as the first gate in the two-stage article pipeline.

normalize_for_guardrail(text)    -> str
    Exposed for logging / testing.  Applies NFKC, homoglyph mapping, and
    whitespace collapsing so that bypass techniques are neutralised before
    patterns are applied.
"""

import re
import unicodedata
import logging

log = logging.getLogger(__name__)

# ── Static injection patterns ─────────────────────────────────────────────────
#
# Applied to BOTH web articles (Stage 1) and user inputs.  Any match causes
# the content to be blocked immediately — no LLM call is made.

_DIRECT_INJECTION_PATTERNS: list[str] = [
    # Role / system override
    r"ignore\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?|system)",
    r"disregard\s+(all\s+)?(previous|prior|above|your)(\s+(previous|prior))?\s+(instructions?|prompts?|rules?)",
    r"forget\s+(everything|all)\s+(you\s+)?(know|were\s+told|have\s+been\s+told)",
    r"you\s+are\s+now\s+(a\s+)?(new|different|unrestricted|evil|dan|jailbreak)",
    r"(act|behave|respond)\s+as\s+(if\s+you\s+(are|were)\s+)?(a\s+)?(different|unrestricted|evil|jailbreak)",
    r"new\s+(system\s+)?prompt\s*[:\-]",
    r"override\s+(system|safety|all)\s+(prompt|instructions?|rules?|constraints?)",
    r"\[system\s*\]",
    r"<\s*system\s*>",
    r"<\s*/?instruction\s*>",
    # Role-play jailbreak attempts
    r"\bDAN\b.{0,30}(mode|prompt|jailbreak)",
    r"developer\s+mode\s+(enabled|activated|on)",
    r"jailbreak\s+(mode|prompt|enabled)",
    r"do\s+anything\s+now",
    # Output manipulation
    r"print\s+(the\s+)?(following|this)\s+(text|message|content|exactly)",
    r"output\s+(only|exactly|the\s+following)",
    r"respond\s+(only\s+)?with\s+[\"']",
    r"your\s+(new\s+)?instructions?\s+(are|is)\s*:",
    r"(from\s+now\s+on|henceforth).{0,40}(you\s+(must|will|should)|always)",
    # Data exfiltration / SSRF via prompt
    r"fetch\s+(the\s+)?(url|page|content)\s+(at|from)\s+https?://",
    r"make\s+a\s+(get|post)\s+request\s+to",
    r"send\s+(the\s+)?(output|response|result)\s+to\s+https?://",
    r"http[s]?://[^\s]{5,}\s*(for\s+instructions?|to\s+get\s+instructions?)",
    # Prompt boundary confusion
    r"---+\s*(end\s+of\s+article|article\s+ends?\s+here)",
    r"={3,}\s*(new\s+instructions?|system\s+prompt)",
    r"```\s*system",
    r"\[\s*(INST|SYS|SYSTEM)\s*\]",
    r"<<\s*(SYS|SYSTEM|INST)\s*>>",
    # Prompt extraction / leaking
    r"(repeat|echo|show|display|reveal|output|print)\s+(all\s+|everything\s+)?(above|previous|prior|before\s+this)",
    r"(reveal|leak|expose|disclose)\s+(your\s+)?(system\s+)?(prompt|instructions?|context|rules?)",
    r"what\s+(are|were|is)\s+(your\s+)?(current\s+)?(instructions?|system\s+prompt|rules?|constraints?)",
    r"(summarize|repeat|show)\s+(everything|all)\s+(above|before|prior|previous)",
    # Special token / ChatML injection (LLM format boundary attacks)
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|endoftext\|>",
    r"<\|system\|>",
    # Additional roleplay / simulation jailbreaks
    r"(roleplay|role\s*-?\s*play)\s+(as|a\s+|an?\s+)",
    r"simulate\s+(being|you\s+are)\s+(a\s+)?(different|unrestricted|uncensored)",
    r"pretend\s+(you\s+)?(have\s+no\s+(rules?|restrictions?|limits?)|are\s+(uncensored|unrestricted|not\s+bound))",
    r"you\s+have\s+no\s+(rules?|restrictions?|limits?|guidelines?|constraints?|ethical\s+bounds?)",
    # Safety bypass / escape attempts
    r"(bypass|circumvent|escape|break\s+out\s+of)\s+(your\s+)?(restrictions?|safety\s+(guidelines?|rules?)?|constraints?|sandbox|guardrails?)",
    # Encoding / obfuscation hints
    r"base64\s*(decode|encoded|the\s+following)",
    r"(decode|deobfuscate)\s+(the\s+following|this)",
]

_COMPILED_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE | re.DOTALL) for p in _DIRECT_INJECTION_PATTERNS
]

# ── Extra patterns that apply only to interactive user inputs ─────────────────
#
# These are broader / more sensitive than the article patterns because user
# inputs are the highest-trust boundary — we can afford slightly more caution.

_USER_INPUT_EXTRA_PATTERNS: list[re.Pattern] = [
    # Note: standalone \bDAN\b omitted — handled more precisely by _COMPILED_PATTERNS
    # (requires DAN + mode|prompt|jailbreak context, or "you are now ... dan")
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"ignore.{0,30}(instructions?|rules?|prompt)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    # Narrowed from broad "(act|pretend|behave) as (if)?" to require a suspicious target
    re.compile(
        r"(act|pretend|behave)\s+as\s+(if\s+)?(you\s+(are|were)|a?\s*(different|unrestricted|uncensored|evil|another\s+ai))",
        re.IGNORECASE,
    ),
    re.compile(r"(system|admin)\s+(prompt|override|mode)", re.IGNORECASE),
    # Roleplay / simulation
    re.compile(r"\broleplay\b", re.IGNORECASE),
    re.compile(r"pretend\s+you\s+(have\s+no|are\s+(not|now|uncensored|unrestricted))", re.IGNORECASE),
    re.compile(r"simulate\s+(being|having\s+no)", re.IGNORECASE),
    # No-restrictions / bypass phrasing
    re.compile(r"you\s+have\s+no\s+(rules?|restrictions?|limits?|constraints?)", re.IGNORECASE),
    re.compile(
        r"(bypass|circumvent)\s+(your\s+)?(restrictions?|safety|guidelines?|guardrails?|rules?)",
        re.IGNORECASE,
    ),
    # Prompt extraction
    re.compile(
        r"(reveal|show|output|repeat|print)\s+(your\s+)?(system\s+)?(prompt|instructions?)",
        re.IGNORECASE,
    ),
    re.compile(r"what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions?|rules?)", re.IGNORECASE),
    re.compile(r"(repeat|echo|show)\s+(everything|all)\s+(above|before|prior|previous)", re.IGNORECASE),
]

# ── Homoglyph normalisation table ─────────────────────────────────────────────
#
# Maps Cyrillic and Greek characters that are visually indistinguishable from
# ASCII letters to their ASCII equivalents, defeating substitution attacks such
# as replacing Latin 'i' with Cyrillic 'і' (U+0456) to bypass keyword filters.

_HOMOGLYPH_MAP: dict[int, str] = {
    # Cyrillic lookalikes
    0x0430: "a",  # а → a
    0x0435: "e",  # е → e
    0x0456: "i",  # і → i (Ukrainian/Byelorussian)
    0x04CF: "i",  # ӏ → i (Chechen)
    0x043E: "o",  # о → o
    0x0440: "p",  # р → p
    0x0441: "c",  # с → c
    0x0445: "x",  # х → x
    0x0455: "s",  # ѕ → s
    0x0501: "d",  # ԁ → d
    0x0443: "y",  # у → y
    # Greek lookalikes
    0x03B1: "a",  # α → a
    0x03BF: "o",  # ο → o
    0x03BD: "v",  # ν → v
    0x03B5: "e",  # ε → e
    0x03C1: "p",  # ρ → p
    0x03BA: "k",  # κ → k
    # Other common confusables
    0x2139: "i",  # ℹ → i
    0x1D0F: "o",  # ᴏ → o (Latin letter small capital)
    0x1D00: "a",  # ᴀ → a
}
_HOMOGLYPH_TABLE = str.maketrans(_HOMOGLYPH_MAP)

# ── Maximum accepted length for user-supplied text ────────────────────────────
MAX_USER_INPUT_LENGTH = 2000


# ── Public helpers ────────────────────────────────────────────────────────────

def normalize_for_guardrail(text: str) -> str:
    """Return a normalised copy of *text* suitable for pattern matching.

    Applies three transformations in order:
      1. NFKC unicode normalisation — collapses compatibility characters such as
         full-width letters (ｉｇｎｏｒｅ → ignore) and mathematical bold glyphs.
      2. Homoglyph substitution — replaces Cyrillic/Greek lookalikes with their
         ASCII counterparts (Cyrillic і → Latin i, etc.).
      3. Unicode whitespace collapsing — folds zero-width, non-breaking, and
         ideographic spaces into a single ASCII space so that whitespace-splitting
         attacks (e.g. "ig nore") cannot fragment keywords.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(_HOMOGLYPH_TABLE)
    normalized = re.sub(
        r"[\s\u00a0\u200b-\u200f\u202f\u205f\u3000]+", " ", normalized
    ).strip()
    return normalized


def check_user_input(text: str, label: str = "user input") -> tuple[bool, str]:
    """Scan *text* for prompt injection patterns.

    Intended for any string that originates from a human (chat questions, topic
    names, research goals, ad-hoc search context, etc.) before it is embedded
    in an LLM prompt.

    Returns
    -------
    (blocked, reason)
        *blocked* is True when the text should be rejected.
        *reason* is a short human-readable explanation (empty string if safe).
    """
    if len(text) > MAX_USER_INPUT_LENGTH:
        reason = f"input too long ({len(text)} chars, max {MAX_USER_INPUT_LENGTH})"
        log.warning(f"[GUARDRAIL] Blocked {label}: {reason}")
        return True, reason

    normalized = normalize_for_guardrail(text)
    for pat in _COMPILED_PATTERNS + _USER_INPUT_EXTRA_PATTERNS:
        m = pat.search(normalized)
        if m:
            snippet = normalized[max(0, m.start() - 15): m.end() + 15].replace("\n", " ")
            reason = f"injection pattern matched: «{snippet.strip()}»"
            log.warning(f"[GUARDRAIL] Blocked {label}: {reason} — original: {text[:120]!r}")
            return True, reason

    return False, ""


def check_article_static(article: dict) -> tuple[bool, str]:
    """Stage-1 static guardrail for a fetched web article (dict with 'title'/'content').

    Uses only _COMPILED_PATTERNS (not the extra user-input patterns) since article
    content is not interactive and the broader user-input patterns would produce
    false positives on legitimate journalism.

    Returns (blocked, reason).  If blocked, the article should be discarded
    before any LLM synthesis; Stage-2 (semantic LLM check) is the caller's
    responsibility.
    """
    combined = f"{article.get('title', '')} {article.get('content', '')}"
    for pat in _COMPILED_PATTERNS:
        m = pat.search(combined)
        if m:
            snippet = combined[max(0, m.start() - 20): m.end() + 20].replace("\n", " ")
            reason = f"direct injection pattern matched: «{snippet.strip()}»"
            return True, reason
    return False, ""

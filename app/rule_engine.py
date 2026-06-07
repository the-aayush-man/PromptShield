from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    rule_id: str
    category: str
    label: str
    pattern: re.Pattern[str]
    weight: int
    explanation: str
    mitigation: str


RULES: tuple[Rule, ...] = (
    Rule(
        "PI-001",
        "Prompt Injection",
        "Instruction override",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b.{0,60}"
            r"\b(previous|prior|above|earlier|all)\b.{0,40}"
            r"\b(instructions?|rules?|context|messages?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        82,
        "The prompt attempts to override earlier instructions or rules.",
        "Keep system instructions isolated and reject instruction-override text.",
    ),
    Rule(
        "PI-002",
        "Prompt Injection",
        "Policy bypass request",
        re.compile(
            r"\b(bypass|circumvent|disable|remove|break)\b.{0,50}"
            r"\b(safety|guardrails?|filters?|restrictions?|polic(?:y|ies))\b",
            re.IGNORECASE | re.DOTALL,
        ),
        86,
        "The prompt asks the model to bypass safety controls.",
        "Refuse the bypass request and continue applying the safety policy.",
    ),
    Rule(
        "PI-003",
        "Prompt Injection",
        "Command execution language",
        re.compile(
            r"\b(execute|run|eval|shell|terminal|subprocess|os\.system|"
            r"rm\s+-rf|powershell|cmd\.exe|bash)\b",
            re.IGNORECASE,
        ),
        68,
        "The prompt contains command execution or shell-oriented language.",
        "Do not execute user-supplied commands without sandboxing and review.",
    ),
    Rule(
        "JB-001",
        "Jailbreak",
        "DAN-style jailbreak",
        re.compile(
            r"\b(DAN|do anything now|developer mode|evilbot|personagpt)\b",
            re.IGNORECASE,
        ),
        92,
        "The prompt resembles a known jailbreak persona pattern.",
        "Treat persona override requests as untrusted user content.",
    ),
    Rule(
        "JB-002",
        "Jailbreak",
        "Unrestricted model request",
        re.compile(
            r"\b(unrestricted|uncensored|no limits?|without limits?|"
            r"no censorship|break character|stay in character)\b",
            re.IGNORECASE,
        ),
        78,
        "The prompt asks the assistant to operate without normal restrictions.",
        "Enforce the normal model behavior and refuse unsafe role instructions.",
    ),
    Rule(
        "JB-003",
        "Jailbreak",
        "Safety refusal manipulation",
        re.compile(
            r"\b(if you refuse|do not refuse|must answer|you cannot refuse|"
            r"never say you can't|no warnings?)\b",
            re.IGNORECASE,
        ),
        72,
        "The prompt tries to control refusal behavior.",
        "Ignore refusal-manipulation clauses and apply the policy normally.",
    ),
    Rule(
        "DE-001",
        "Data Extraction",
        "System prompt extraction",
        re.compile(
            r"\b(reveal|show|print|display|repeat|leak|expose)\b.{0,60}"
            r"\b(system prompt|hidden prompt|developer message|"
            r"internal instructions?|initial instructions?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        95,
        "The prompt asks for hidden or internal instructions.",
        "Never disclose system prompts, hidden policies, or developer messages.",
    ),
    Rule(
        "DE-002",
        "Data Extraction",
        "Secret or credential extraction",
        re.compile(
            r"\b(passwords?|api keys?|tokens?|secrets?|credentials?|"
            r"private keys?|database|confidential)\b",
            re.IGNORECASE,
        ),
        76,
        "The prompt references secrets, credentials, or confidential data.",
        "Block data exfiltration and return a privacy-preserving response.",
    ),
    Rule(
        "RM-001",
        "Roleplay Manipulation",
        "Malicious roleplay",
        re.compile(
            r"\b(pretend|roleplay|act as|simulate|you are now)\b.{0,70}"
            r"\b(unrestricted|hacker|evil|criminal|no rules|DAN|"
            r"developer mode)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        88,
        "The prompt disguises unsafe behavior as roleplay.",
        "Allow harmless roleplay only when it does not override safety rules.",
    ),
    Rule(
        "RM-002",
        "Roleplay Manipulation",
        "Alternate identity override",
        re.compile(
            r"\b(from now on|for this conversation|new role|new persona)\b"
            r".{0,90}\b(you are|act as|pretend)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        64,
        "The prompt tries to replace the assistant identity or persona.",
        "Keep assistant identity and instruction hierarchy unchanged.",
    ),
)


def scan_prompt(prompt: str) -> list[dict[str, object]]:
    indicators: list[dict[str, object]] = []
    for rule in RULES:
        match = rule.pattern.search(prompt)
        if not match:
            continue

        evidence = " ".join(match.group(0).split())
        indicators.append(
            {
                "rule_id": rule.rule_id,
                "category": rule.category,
                "label": rule.label,
                "weight": rule.weight,
                "evidence": evidence[:180],
                "explanation": rule.explanation,
                "mitigation": rule.mitigation,
            }
        )
    return indicators

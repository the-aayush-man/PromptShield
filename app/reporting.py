from __future__ import annotations

import textwrap
from datetime import datetime
from typing import Any


def _escape_pdf_text(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\r", " ")
    )


def _build_lines(record: dict[str, Any]) -> list[str]:
    result = record
    prompt = str(record.get("prompt", ""))
    lines = [
        "PromptShield Security Analysis Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Analysis ID: {record.get('id', '')}",
        "",
        f"Classification: {result.get('classification', '')}",
        f"Confidence: {result.get('confidence_percent', '')}%",
        f"Risk Score: {result.get('risk_score', '')}",
        f"Risk Level: {result.get('risk_level', '')}",
        f"Severity: {result.get('severity', '')}",
        "",
        "Triggered Indicators:",
    ]
    indicators = result.get("triggered_indicators", [])
    if indicators:
        for item in indicators:
            lines.append(
                "- "
                f"{item.get('rule_id', '')} | {item.get('category', '')} | "
                f"{item.get('label', '')}"
            )
            evidence = str(item.get("evidence", ""))
            if evidence:
                lines.extend(textwrap.wrap(f"  Evidence: {evidence}", 92))
    else:
        lines.append("- None")

    lines.extend(["", "Explanation:"])
    for explanation in result.get("explanations", []):
        lines.extend(textwrap.wrap(f"- {explanation}", 94))

    lines.extend(["", "Mitigation Recommendations:"])
    for mitigation in result.get("mitigations", []):
        lines.extend(textwrap.wrap(f"- {mitigation}", 94))

    lines.extend(["", "Original Prompt:"])
    lines.extend(textwrap.wrap(prompt, 94))
    return lines[:86]


def create_pdf(record: dict[str, Any]) -> bytes:
    lines = _build_lines(record)
    content_lines = ["BT", "/F1 10 Tf", "50 770 Td", "14 TL"]
    for line in lines:
        content_lines.append(f"({_escape_pdf_text(line)}) Tj")
        content_lines.append("T*")
    content_lines.append("ET")
    content = "\n".join(content_lines).encode("latin-1", errors="replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> "
            b"/Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
        (
            f"<< /Length {len(content)} >>\nstream\n".encode("ascii")
            + content
            + b"\nendstream"
        ),
    ]

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")

    xref_start = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)

#!/usr/bin/env python3
"""Sanitize JSONL session files by replacing sensitive data.

Usage:
    python scripts/sanitize_sessions.py sample_sessions/
    python scripts/sanitize_sessions.py sample_sessions/*.jsonl

Creates sanitized copies with .sanitized.jsonl suffix, leaving originals intact.
Also prints a report of what was found and replaced.
"""

import re
import sys
from pathlib import Path

# --- Patterns to find and their replacements ---

REPLACEMENTS = [
    # API keys and secrets (specific known values)
    ("1DW6P8FyiToyT1lasBLAM81pZ1PjxYk4E9Lh4Qgg", "REDACTED_CLIENT_SECRET_1"),
    ("eYTeM7yExqnIo2oSKW0VBnk3yuJdwae8", "REDACTED_CLIENT_SECRET_2"),
    ("AIzaSyA8YvsPqHg4zcFdRTWyrkBKlovLTN8wIaY", "REDACTED_GOOGLE_API_KEY"),
    ("22f497d24c47417eacde22c9637d991e", "REDACTED_CLIENT_ID_1"),
    ("a17f4a01-45ff-447d-baf0-4aedb2e4bae8", "REDACTED_CLIENT_ID_2"),
    ("aed6bfe16f3b4b37ba8dd3a83afec8ab", "REDACTED_ESI_CLIENT_ID"),
    ("1pT8SjbujcvO-ZHkaJnMSo8vBEFiVRLRl", "REDACTED_GDRIVE_FOLDER_ID"),

    # User-Agent with name
    ("Ray Zolo", "Redacted User"),

    # App names — longer/more-specific forms first to avoid partial matches.
    ("EveGuruAppBase", "SampleAppBase"),
    ("EveGuruSettings", "SampleAppSettings"),
    ("EveGuruEditions", "SampleAppEditions"),
    ("EveGuruData", "SampleAppData"),
    ("EveGuruApp", "SampleApp"),
    ("EveGuru", "SampleApp"),
    ("EVEGuru", "SampleApp"),
    ("EVEGURU", "SAMPLEAPP"),
    ("EVE_Guru", "Sample_App"),
    ("EVE-Guru", "Sample-App"),
    ("EVE Guru", "Sample App"),
    ("EVE GURU", "SAMPLE APP"),
    ("eve-guru", "sample-app"),
    ("eveguru", "sampleapp"),
    ("toads_tbc", "sample_project"),
    ("toads-tbc", "sample-project"),
]

REGEX_REPLACEMENTS: list[tuple[re.Pattern, str]] = []


def _build_regex_replacements():
    """Build regex replacements. Uses str.replace in callbacks to avoid
    regex substitution escaping issues with backslashes."""
    global REGEX_REPLACEMENTS
    REGEX_REPLACEMENTS = [
        # JSON-escaped Windows paths: C:\\Users\\leigh\\ (as stored in JSONL)
        (re.compile(r"C:\\\\Users\\\\leigh(?=\\\\)"), "C:\\\\Users\\\\user"),
        # Raw Windows paths: C:\Users\leigh\ or C:/Users/leigh/
        (re.compile(r"C:[\\\/]Users[\\\/]leigh(?=[\\\/])"), "C:/Users/user"),
        # Unix-style: /c/Users/leigh/
        (re.compile(r"/c/Users/leigh(?=/)"), "/c/Users/user"),
        # Home dir references: ~leigh or /home/leigh
        (re.compile(r"(?:/home/|~)leigh\b"), "/home/user"),
        # Bare "leigh" as a word (ownership, usernames, etc.)
        (re.compile(r"\bleigh\b"), "user"),
        # GitHub username
        (re.compile(r"lgriffin/TER"), "user/TER"),
        # Production URLs
        (re.compile(r"app\.eveguru\.online"), "app.example.com"),
    ]


_build_regex_replacements()


def sanitize_file(path: Path) -> dict:
    """Sanitize a single JSONL file. Returns a report dict."""
    content = path.read_text(encoding="utf-8")
    original_len = len(content)
    counts: dict[str, int] = {}

    # Apply literal replacements.
    for old, new in REPLACEMENTS:
        n = content.count(old)
        if n > 0:
            counts[new] = n
            content = content.replace(old, new)

    # Apply regex replacements using lambda to avoid backslash escaping.
    for pattern, replacement in REGEX_REPLACEMENTS:
        matches = pattern.findall(content)
        if matches:
            label = replacement[:40]
            counts[label] = counts.get(label, 0) + len(matches)
            repl = replacement  # capture for lambda
            content = pattern.sub(lambda m: repl, content)

    # Write sanitized output.
    out_path = path.with_suffix(".sanitized.jsonl")
    out_path.write_text(content, encoding="utf-8")

    return {
        "file": path.name,
        "output": out_path.name,
        "original_size": original_len,
        "sanitized_size": len(content),
        "replacements": counts,
    }


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_or_glob>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        files = sorted(target.glob("*.jsonl"))
        # Exclude already-sanitized files.
        files = [f for f in files if ".sanitized" not in f.name]
    else:
        files = [Path(a) for a in sys.argv[1:]]

    if not files:
        print("No .jsonl files found.")
        sys.exit(1)

    print(f"Sanitizing {len(files)} file(s)...\n")

    total_replacements = 0
    for f in files:
        report = sanitize_file(f)
        n = sum(report["replacements"].values())
        total_replacements += n

        status = f"  {n} replacements" if n > 0 else "  clean"
        print(f"{report['file']}: {status}")
        if report["replacements"]:
            for label, count in sorted(report["replacements"].items()):
                print(f"    {label}: {count}")
        print(f"    -> {report['output']}")
        print()

    print(f"Done. {total_replacements} total replacements across {len(files)} files.")
    print("\nSanitized files have .sanitized.jsonl suffix.")
    print("Review them, then rename to replace the originals if satisfied.")


if __name__ == "__main__":
    main()

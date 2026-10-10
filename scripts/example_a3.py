"""Regenerate the example A3 under ``docs/examples/a3/`` (linked from the README).

The example is a synthetic session (no real user data) analysed with the
offline adapters ``ter a3`` uses by default and placed against the control
limits of the golden corpus, so anyone can rebuild it byte for byte::

    python scripts/example_a3.py               # a3.html, a3.json, limits.json
    python scripts/example_a3.py --screenshots # also the PNGs the README shows

``tests/docs/test_example_a3.py`` fails when the committed page no longer
matches what TER produces; ``TER_UPDATE_GOLDEN=1`` on that test rewrites the
same files as this script, without the screenshots. Screenshots need
Playwright and a Chromium (``--chrome`` names its executable).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs" / "examples" / "a3"
SESSION = EXAMPLE / "session.jsonl"
LIMITS = EXAMPLE / "limits.json"
GOLDEN_LIMITS = (
    ROOT / "tests" / "golden" / "snapshots" / "control" / "corpus.limits.json"
)
HTML = EXAMPLE / "a3.html"
JSON = EXAMPLE / "a3.json"
#: Screenshot file -> the page element it shows (None: the top of the page).
SCREENSHOTS: dict[str, str | None] = {
    "a3-summary.png": None,
    "a3-process-control.png": "#s-control",
}


def a3_argv(html: Path, json: Path) -> list[str]:
    """The ``ter a3`` command the example is built with."""
    return [
        "a3",
        str(SESSION),
        "--limits",
        str(LIMITS),
        "--html",
        str(html),
        "--json",
        str(json),
    ]


def build() -> None:
    from ter import bootstrap

    shutil.copyfile(GOLDEN_LIMITS, LIMITS)
    code = bootstrap.main(a3_argv(HTML, JSON))
    if code != 0:
        raise SystemExit(code)


def screenshots(chrome: str | None) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chrome)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(HTML.as_uri())
        page.emulate_media(color_scheme="light")
        for name, selector in SCREENSHOTS.items():
            target = EXAMPLE / name
            if selector is None:
                page.screenshot(path=str(target))
            else:
                page.locator(selector).screenshot(path=str(target))
            print(f"Wrote {target.relative_to(ROOT)}")
        browser.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--screenshots", action="store_true")
    parser.add_argument("--chrome", help="Chromium executable for Playwright")
    args = parser.parse_args(argv)
    build()
    if args.screenshots:
        screenshots(args.chrome)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())

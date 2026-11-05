"""The Claude Code plugin loads: manifests are valid and every skill is named.

The commands its skills tell Claude to run are checked with the guides' in
test_docs.py (TER-REQ-008); this module checks the files Claude Code itself
reads when the plugin is installed (TER-REQ-014).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _json(path: Path, problems: list[str]) -> dict[str, object] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        problems.append(f"{path}: missing")
        return None
    except json.JSONDecodeError as error:
        problems.append(f"{path}: not valid JSON ({error})")
        return None
    if not isinstance(data, dict):
        problems.append(f"{path}: not a JSON object")
        return None
    return data


def _frontmatter(path: Path) -> dict[str, object] | None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None
    head, sep, _ = text[4:].partition("\n---\n")
    if not sep:
        return None
    data = yaml.safe_load(head)
    return data if isinstance(data, dict) else None


def _skill_problems(plugin: Path) -> list[str]:
    problems: list[str] = []
    skills = sorted((plugin / "skills").glob("*/SKILL.md"))
    if not skills:
        problems.append(f"{plugin}: no skills/<name>/SKILL.md")
    for skill in skills:
        meta = _frontmatter(skill)
        if meta is None:
            problems.append(f"{skill}: no YAML frontmatter")
            continue
        if meta.get("name") != skill.parent.name:
            problems.append(f"{skill}: name is not {skill.parent.name!r}")
        description = meta.get("description")
        if not isinstance(description, str) or not description.strip():
            problems.append(f"{skill}: no description")
    return problems


def plugin_problems(root: Path) -> list[str]:
    """Everything that would stop Claude Code loading the marketplace at root."""
    problems: list[str] = []
    market = _json(root / ".claude-plugin" / "marketplace.json", problems)
    if market is None:
        return problems
    owner = market.get("owner")
    if not isinstance(market.get("name"), str) or not (
        isinstance(owner, dict) and isinstance(owner.get("name"), str)
    ):
        problems.append("marketplace.json: needs a name and an owner name")
    entries = market.get("plugins")
    if not isinstance(entries, list) or not entries:
        problems.append("marketplace.json: lists no plugins")
        return problems
    for entry in entries:
        if not isinstance(entry, dict):
            problems.append(
                f"marketplace.json: plugin entry {entry!r} is not an object"
            )
            continue
        name, source = entry.get("name"), entry.get("source")
        if not isinstance(source, str) or not source.startswith("./") or ".." in source:
            problems.append(
                f"marketplace.json: {name}: source must be ./<path> in this repo"
            )
            continue
        plugin = root / source
        manifest = _json(plugin / ".claude-plugin" / "plugin.json", problems)
        if manifest is None:
            continue
        if manifest.get("name") != name:
            problems.append(
                f"{plugin / '.claude-plugin' / 'plugin.json'}: name "
                f"{manifest.get('name')!r} differs from the marketplace entry {name!r}"
            )
        problems.extend(_skill_problems(plugin))
    return problems


@pytest.mark.req("TER-REQ-014")
def test_the_plugin_loads() -> None:
    assert plugin_problems(ROOT) == []


@pytest.mark.req("TER-REQ-014")
def test_plugin_check_names_each_broken_file(tmp_path: Path) -> None:
    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin" / "marketplace.json").write_text(
        json.dumps(
            {
                "name": "m",
                "owner": {"name": "o"},
                "plugins": [
                    {"name": "good", "source": "./p"},
                    {"name": "gone", "source": "./nowhere"},
                    {"name": "up", "source": "../outside"},
                ],
            }
        ),
        encoding="utf-8",
    )
    plugin = tmp_path / "p"
    (plugin / ".claude-plugin").mkdir(parents=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text(
        '{"name": "other"}', encoding="utf-8"
    )
    (plugin / "skills" / "a").mkdir(parents=True)
    (plugin / "skills" / "a" / "SKILL.md").write_text(
        "---\nname: b\n---\nbody\n", encoding="utf-8"
    )
    (plugin / "skills" / "c").mkdir()
    (plugin / "skills" / "c" / "SKILL.md").write_text(
        "no frontmatter\n", encoding="utf-8"
    )

    problems = plugin_problems(tmp_path)

    assert any("name 'other' differs" in p for p in problems)
    assert any("a/SKILL.md: name is not 'a'" in p for p in problems)
    assert any("a/SKILL.md: no description" in p for p in problems)
    assert any("c/SKILL.md: no YAML frontmatter" in p for p in problems)
    assert any("nowhere" in p and "missing" in p for p in problems)
    assert any("up: source must be" in p for p in problems)


@pytest.mark.req("TER-REQ-014")
def test_plugin_check_rejects_invalid_json(tmp_path: Path) -> None:
    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin" / "marketplace.json").write_text("{", encoding="utf-8")

    assert [p for p in plugin_problems(tmp_path) if "not valid JSON" in p]


def test_the_plugin_registers_no_hooks() -> None:
    """Hooks record prompts and run on every tool call, and Claude Code cannot
    turn off one plugin component on its own, so /ter:setup-hooks writes them
    with the user's consent instead (plugins/ter/README.md)."""
    for plugin in (ROOT / "plugins").iterdir():
        assert not (plugin / "hooks").exists(), plugin
        manifest = json.loads(
            (plugin / ".claude-plugin" / "plugin.json").read_text("utf-8")
        )
        assert "hooks" not in manifest

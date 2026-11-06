# TER plugin for Claude Code (exploratory)

> Exploratory: skills, names and install steps may change. Guide:
> [docs/guides/claude-plugin.md](../../docs/guides/claude-plugin.md).

Two skills that let Claude run TER for you:

| Skill | What it does |
|---|---|
| `/ter:analyze [path \| this \| last]` | Runs the Lean A3 on a session, explains the findings (confident and uncertain kept apart) and offers to apply the countermeasures it recommends to CLAUDE.md, hooks and settings. Claude also uses it on its own when you ask how wasteful a session was. |
| `/ter:setup-hooks [capture \| advice \| check]` | Turns on live capture, and live advice once you have approved a policy file. Only you can start it. |

## Install

The plugin is instructions; TER itself is a Python package. Install both:

```bash
python -m pip install "ter-calculator @ git+https://github.com/lgriffin/TER"
```

Then in Claude Code:

```text
/plugin marketplace add lgriffin/TER
/plugin install ter@ter
```

## Why no hooks ship in the plugin

Claude Code turns on every hook in a plugin when the plugin is enabled, and
cannot switch one off on its own. TER's hooks record your prompts and tool
output and run on every tool call, and live advice needs a policy file you
approved yourself. So the plugin registers none: `/ter:setup-hooks` writes
them into a settings file you can read, after you agree.

# The Claude Code plugin (exploratory)

> **Exploratory.** The plugin is an experiment in letting Claude run TER for
> you. Its skills, names and install steps may change or be removed, and no
> vision point depends on it. The CLI and the hooks described in the other
> guides are the supported way to use TER.

TER is a command line tool. To use it well you need to know where Claude
Code keeps transcripts, which command writes the A3, how to read uncertain
findings, and how to turn a countermeasure into a CLAUDE.md line or a hook.
The plugin puts that knowledge into two
[skills](https://code.claude.com/docs/en/skills), so you can ask Claude
instead:

| Skill | Who starts it | What it does |
|---|---|---|
| `/ter:analyze [path \| this \| last]` | You, or Claude when you ask how wasteful a session was | Writes the A3 for a session, explains it, and offers to apply the countermeasures you pick |
| `/ter:setup-hooks [capture \| advice \| check]` | Only you | Writes TER's capture and live-advice hooks into a settings file, after showing you the JSON |

The skills live in [plugins/ter/](../../plugins/ter/README.md).

## Install

The plugin holds instructions only. TER itself is a Python package, so it is
installed in two steps.

1. Install TER into the Python that Claude Code will run (3.11 to 3.13):

   ```bash
   python -m pip install "ter-calculator @ git+https://github.com/lgriffin/TER"
   python -m ter --version
   ```

2. In Claude Code, add this repository as a plugin marketplace and install
   the plugin:

   ```text
   /plugin marketplace add lgriffin/TER
   /plugin install ter@ter
   ```

To try a local checkout without installing, start Claude Code with
`claude --plugin-dir plugins/ter`.

## Analyse a session

Ask in your own words ("how wasteful was this session?") or name the skill:

```text
/ter:analyze this
/ter:analyze last
/ter:analyze ~/.claude/projects/my-project/SESSION_ID.jsonl
```

Claude checks TER is installed, finds the transcript and runs the same
command you would:

```bash
python -m ter a3 SESSION.jsonl --json a3.json --html a3.html
```

It writes the output to a temporary folder, not your repository, and adds
`--outcome`, `--limits` or `--repo` only when they are true for the session
(see [the A3 guide](a3-report.md)). Then it reports the result under the
same rules TER uses everywhere:

- An **uncertain** finding (confidence below 0.70) is labelled as uncertain
  and counted as waste until verified ([ADR 0006](../decisions/0006-uncertain-waste-counts-until-verified.md)). Claude
  doesn't treat it as fact, and doesn't apply its countermeasure until you
  have checked it.
- **Value is never inferred from tokens.** Without test results
  (`--outcome`), value efficiency is reported as unknown.
- Every finding is quoted with its evidence, so you can check it.

Last, Claude lists the A3's countermeasures (CLAUDE.md lines, hook scripts,
harness settings) and asks which ones to apply. It shows each change before
writing it, and doesn't touch a settings file without your yes.

## Turn on hooks

`/ter:setup-hooks` is the [hooks guide](hooks.md) as a conversation. Claude
finds the absolute path of the Python that TER is installed in, asks which
settings file to use, shows the JSON it will merge, and writes it only after
you agree.

For live advice it walks through `policy check`, your approval, a shadow
run and, later, `policy effectiveness`. **It never runs `policy approve`.** Approval is your
sign-off, and the skill tells you to type that command yourself.

## Why the plugin ships no hooks

When a plugin is enabled, Claude Code turns on every hook in it, and it can't
switch off one part of a plugin on its own. TER's hooks run on every tool
call and record your prompts and tool output, and live advice needs a policy
file that you approved. So the plugin registers no hooks, and
`/ter:setup-hooks` writes them into a settings file you can read.
`tests/docs/test_plugin.py` pins this.

## How it is tested

- `tests/docs/test_docs.py` reads the skills like a guide. Every relative
  link must resolve, and every `python -m ter` or `ter` command in a shell
  block must exist with the options it uses (TER-REQ-008).
- `tests/docs/test_plugin.py` checks that the marketplace and plugin
  manifests are valid JSON, that their names agree, and that every skill has
  a name and a description (TER-REQ-014).
- `claude plugin validate .` and `claude plugin validate plugins/ter` check
  what Claude Code itself reads. They aren't run in CI, so run them when you
  change the manifests.

## Known gaps

- Installing takes two steps. Claude Code installs only npm dependencies for
  plugins, so TER has to be installed with pip first.
- TER 4 has no console script of its own, so the skills call
  `python -m ter`. That needs the right `python` on PATH, or the interpreter
  named in full.
- The skills are checked for structure and commands, not for how well Claude
  follows them. There is no evaluation suite yet.

# Security Policy

## Supported versions

Security updates are provided for the latest public release line.

| Version | Supported |
| --- | --- |
| 2.x | Yes |
| Earlier internal iterations | No |

## Reporting a vulnerability

Do not disclose suspected vulnerabilities in a public issue. Use the repository host's private security-advisory mechanism when available, and include:

- the affected TER version;
- reproduction steps or a minimal sample;
- expected and observed behavior;
- potential impact;
- any suggested mitigation.

Avoid attaching private Claude session data. Replace sensitive prompts, outputs, paths, credentials, and identifiers with a minimal synthetic reproduction.

## Generated HTML reports

TER HTML reports embed analyzed session content. Treat generated reports as potentially sensitive artifacts. Review them before sharing, store them with access controls appropriate for the underlying session, and do not publish reports containing confidential prompts or model outputs.

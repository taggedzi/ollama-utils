# Security Policy

## Supported Versions

This project currently supports only the latest published `0.6.x` release line for security fixes.

| Version | Supported |
| --- | --- |
| `0.6.x` | Yes |
| `< 0.6.0` | No |
| Unreleased local/dev builds | Best effort only |

If you report a security issue against an older release, the expected resolution may be "upgrade to the latest supported version".

## Reporting A Vulnerability

For vulnerabilities that could lead to unauthorized access, unintended model/library modification, data exposure, or remote-code-execution concerns:

- Do not open a public GitHub issue with exploit details.
- Use GitHub's private vulnerability reporting flow for this repository if it is available.
- If private reporting is not available, open a minimal public issue that only requests a private contact path and does not include exploit details, sample secrets, or proof-of-concept payloads.

Include the following where possible:

- affected version and install method
- operating system
- whether the issue requires a local, remote, or GUI workflow
- exact command, option set, or screen path involved
- steps to reproduce
- impact and any required preconditions

## Maintainer Availability

This project is maintained on a best-effort basis.

- There is no guaranteed response time for security reports.
- There is no guarantee that every report will receive an individual reply, a fix, or a public advisory.
- Periods of maintainer inactivity may be long, including months at a time.
- Reports are still appreciated and may be reviewed and acted on when the maintainer is able.

If you need guaranteed response times, contractual support, or formal security handling obligations, do not rely on this repository alone.

## Disclosure

If a security issue is reviewed and a fix or mitigation is prepared, public disclosure may happen after that work is available. Some reports may remain unresolved, may be closed without action, or may need to be handled by an upstream project instead.

## Scope Notes

This repository wraps and drives a local or user-selected Ollama installation. Reports about third-party models, upstream Ollama behavior, local operating-system configuration, or externally hosted Ollama servers may need to be forwarded to the appropriate upstream owner in addition to this repository.

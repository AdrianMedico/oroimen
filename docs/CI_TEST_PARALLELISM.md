# CI test parallelism

Oroimen deliberately uses different pytest worker settings for local
development and the public CI gate.

## Current policy

| Context | Command shape | Purpose |
|---|---|---|
| Local development | `pytest ... -n 4` | Fast feedback on the measured local development environment |
| Public unit CI | `pytest ...` (no xdist) | Reproducible baseline on GitHub-hosted runners |
| Public integration CI | `pytest ...` (no xdist) | Avoid concurrency masking lifecycle, SQLite, or ordering defects |
| Provider-backed F2 | `pytest ...` (no xdist) | Strictly manual, bounded external validation |

The previous CI configuration combined duration-balanced shards with xdist.
That layout was tuned for different hardware and depended on a generated
`.test_durations` file. It is not copied into the public workflow because its
hardware assumptions and timing data are not portable.

[GitHub currently documents](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
four vCPUs and 16 GB of memory for standard
Linux runners used by public repositories. That means `-n 4` may eventually
be appropriate, but CPU count alone does not establish that it is reliable:
xdist can expose or conceal ordering, SQLite, filesystem, and lifecycle
interactions.

## Promotion rule

The serial public gate is the initial correctness baseline. Parallelism
should change only in a dedicated CI change after collecting timings and
outcomes from at least five representative runs on `ubuntu-latest`.

Record:

- runner image and observed logical CPU count;
- dependency-install time and pytest wall time;
- pass, failure, retry, and cancellation outcomes;
- any difference between serial and parallel test selection or results.

Then compare no xdist, `-n 2`, and `-n 4`. Adopt the fastest setting that has
identical collected tests and stable results. Introduce `pytest-split` only
with a freshly generated, committed `.test_durations` file and documented
shard balance.

The local `-n 4` command remains a developer convenience, not release
evidence for the public runner.

## Related gate scope

Public CI enforces `ruff check`, while `ruff format --check` remains a tracked
follow-up because the imported codebase has pre-existing format drift. Enable
the formatter gate in the dedicated cleanup change that first establishes a
clean formatting baseline.

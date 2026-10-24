<!--
Thank you for contributing to AntroPy! Please read the contributing guidelines first:
https://raphaelvallat.com/antropy/contributing.html

Every PR must state its user impact (P1-P3). P4 changes (refactors, type hints, style,
test-only changes, edge cases nobody hits) are not accepted as standalone PRs.
If you used an AI tool, also follow the guidelines for AI-assisted pull requests.
PRs that don't follow these guidelines may be closed without review.
-->

## Summary

<!-- A short paragraph, written by you (not by an AI), explaining in plain English what this PR does and why. -->

Closes #

<!-- If fixing a bug: add the exact code to reproduce the bug, and show the output before and after the fix. -->

## Impact

<!-- Pick one and justify it in one or two sentences. See the "Assessing the impact of your change" section of the contributing guidelines.
P1: critical (wrong results or crash for most users, broken install, incompatibility with a new NumPy/SciPy/Numba release)
P2: important (bug in a less common but legitimate use, well-established new measure, benchmarked speed-up, misleading docs)
P3: minor (rare edge case, modest speed-up, clearer error message) - requires an issue first
P4: negligible - do not submit -->

**Rating:** P

**Why:**

## AI assistance

<!-- If you used an AI tool, report the exact model (e.g. "Claude Opus 5.5 via Claude Code"). Otherwise, write "None". -->

**Model used:**

<!-- Post a summary of the AI review (e.g. `/review` in Claude Code) as a comment on this PR: what it found, what you fixed, and anything you chose not to fix and why. -->

## Checklist

- [ ] The impact rating is stated and justified above (P1-P3)
- [ ] An issue was opened or discussed first (required for P3 and new features)
- [ ] The branch is up to date with `master`, and I checked open PRs for conflicts
- [ ] The PR is small and focused, or stacked
- [ ] The description starts with my own summary
- [ ] Only necessary tests were added, and all tests pass (`pytest`)
- [ ] Code style passes (`uvx ruff check` and `uvx ruff format --check`)
- [ ] The documentation is updated
- [ ] Performance claims are backed by a benchmark from `benchmarks/`
- [ ] The changelog was not modified (it is updated by the maintainer before each release)

If AI-assisted:

- [ ] The exact model used is reported
- [ ] An AI review was run, the issues were fixed, and a summary was posted
- [ ] `/simplify` was run (preferred, Claude Code)

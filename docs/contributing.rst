.. _contributing:

Contributing
============

AntroPy is an open-source project and contributions are welcome! Here's how you can help.

Reporting bugs
--------------

If you find a bug, please open an issue on the `GitHub repository <https://github.com/raphaelvallat/antropy/issues>`_ with:

- A clear description of the problem
- A minimal reproducible example
- Your Python and AntroPy version

Assessing the impact of your change
-----------------------------------

AntroPy has a single maintainer, and reviewing a pull request often takes far longer than
writing it. To keep review time focused on changes that matter to users, every pull request
must state its **user impact** using the scale below, with one or two sentences explaining
the rating. The maintainer may re-rate a PR; the final rating decides whether and how fast it
is reviewed.

.. list-table::
   :header-rows: 1
   :widths: 8 32 60

   * - Rating
     - Definition
     - Examples
   * - **P1**
     - Critical. Affects most users, or silently produces wrong results.
     - Wrong numerical output for common inputs or default parameters; crash or
       exception on standard usage; package fails to install or import; breakage with
       a new release of NumPy, SciPy or Numba; security issue.
   * - **P2**
     - Important. A clear benefit for a meaningful share of users.
     - Wrong output for a less common but legitimate parameter combination; a new
       entropy or fractal measure that is well established in the literature and has
       been requested; a substantial, benchmarked speed-up on realistic input sizes;
       misleading documentation (wrong formula, parameter description or reference).
   * - **P3**
     - Minor. Correct and useful, but only for rare edge cases.
     - Incorrect handling of degenerate inputs (constant signal, NaN, very short
       arrays) that are unlikely in practice; modest speed-up; clearer error message;
       small optional convenience parameter.
   * - **P4**
     - Negligible. No measurable benefit to users.
     - Refactoring or renaming without a behavior change; type hints or docstring
       wording only; test-only changes; defensive checks for inputs nobody passes;
       micro-optimizations without a benchmark; style or formatting tweaks.

Rules of thumb:

- **P4 changes should not be submitted as standalone pull requests.** They will be closed
  without review. If a P4 change is a natural part of a P1-P3 PR touching the same code,
  include it there and mention it in the description. Otherwise, open a Discussion if you
  think the change is worth doing anyway.
- **P3 changes and all new features require an issue first**, so we can agree the change is
  worth doing before anyone writes or reviews code.
- **P1 changes are fast-tracked.** Please say "P1" in the PR title and include a minimal
  reproducible example showing the bug.
- **Performance claims need a benchmark.** Use or extend the scripts in the ``benchmarks/``
  folder and paste the before/after numbers in the PR description.
- **New or modified numerics need a reference.** When adding a measure or changing an
  algorithm, show that the output matches a published formula, a reference implementation
  or the previous AntroPy version on representative inputs.

When in doubt, rate lower rather than higher, and ask in an issue before writing code.

Submitting a pull request
--------------------------

1. Fork the `repository <https://github.com/raphaelvallat/antropy>`_ and clone your fork locally.

2. Install in editable mode with test dependencies using `uv <https://docs.astral.sh/uv/>`_:

   .. code-block:: shell

       git clone https://github.com/<your-username>/antropy.git
       cd antropy
       uv pip install --group=test --editable .

3. Create a branch for your changes:

   .. code-block:: shell

       git checkout -b my-feature

4. Make your changes, add tests, and verify everything passes:

   .. code-block:: shell

       pytest --verbose

5. Check code style with ruff:

   .. code-block:: shell

       uvx ruff check
       uvx ruff format --check

6. Open a pull request against the ``master`` branch, following the PR template. State
   the impact rating of your change and, if you used an AI tool, follow the
   guidelines for AI-assisted pull requests below.

Guidelines for AI-assisted pull requests
----------------------------------------

AI tools are welcome. But a PR that takes a minute to generate can take hours to review, so
contributors who use AI are asked to do more of that work before requesting a review. PRs
that don't follow these guidelines may be closed without review.

**Before opening a PR**

- Rate the impact of your change using the scale above. Do not submit P4 changes.
- Open an issue first (preferred, required for P3 and new features). Describe the problem and
  the change you're proposing.
- Check other open PRs that touch the same files or functions, to avoid duplicate work and
  merge conflicts.
- Base your branch on the latest ``master``. Rebase if ``master`` moves while your PR is open.

**Opening the PR**

1. Open it as a draft.
2. Run an AI review and fix the issues it finds. If you use Claude Code, run ``/review``;
   with other tools, use their equivalent. Then post a summary of the review as a comment:
   what it found, what you fixed, and anything you chose not to fix and why.
3. Run ``/simplify`` (preferred, Claude Code). This helps keep the implementation small and
   efficient.
4. Mark the PR as ready for review only after these steps are done.

**PR description**

- Start with a summary written by you, not by the AI: a short paragraph explaining in your
  own words what the PR does and why.
- State the impact rating (P1-P3) and justify it in one or two sentences.
- Keep it concise and in plain English. If fixing a bug, add the exact code to reproduce it
  and show the output before and after the fix.
- Report the exact model you used, e.g. "Claude Opus 5.5 via Claude Code".

**Scope and size**

- Keep PRs small and focused. A PR should do one thing. If your change is large, split it
  into a series of small PRs stacked on top of each other.
- Add tests only when they're needed. AI agents tend to write too many tests, which slows
  down the test suite and makes it harder to maintain. Add a test only if it covers new
  behavior, a bug fix (a regression test), or a critical edge case. Remove tests that
  duplicate existing ones or only check trivial things.
- Do not modify the changelog. It is updated by the maintainer before each release.

**Everything else still applies**

You're responsible for every line you submit. If you can't explain a change, please don't
include it.

Updating the documentation
--------------------------

The documentation is built with `Sphinx <https://www.sphinx-doc.org/>`_. To build it locally:

1. Install the documentation dependencies:

   .. code-block:: shell

       uv pip install --group=docs --editable .

2. Build the HTML docs:

   .. code-block:: shell

       make -C docs clean
       make -C docs html

3. Open ``docs/build/html/index.html`` in your browser to preview the result.

When you open a pull request, a ``docs`` CI job automatically builds the documentation and
uploads the result as a GitHub Actions artifact. To verify the rendered docs from a PR:

1. Go to the PR on GitHub and click on the **Checks** tab.
2. Open the **Build documentation and upload as artifact to GitHub Actions** workflow run.
3. Click **Summary** in the left sidebar, then scroll down to the **Artifacts** section.
4. Download the ``docs-artifact`` zip file and open ``index.html`` in your browser.

Questions
---------

If you have questions, please use `GitHub Discussions <https://github.com/raphaelvallat/antropy/discussions>`_ rather than opening an issue.

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

6. Open a pull request against the ``master`` branch.

Questions
---------

If you have questions, please use `GitHub Discussions <https://github.com/raphaelvallat/antropy/discussions>`_ rather than opening an issue.

![alt text](https://github.com/PyDFLT/PyDFLT/blob/update-readme/images/logo.png?raw=true)
<img src="https://github.com/PyDFLT/PyDFLT/blob/update-readme/images/logo.png?raw=true" alt="description" width="400"/>

[![CI](https://github.com/PyDFLT/PyDFLT/actions/workflows/CI.yml/badge.svg)](https://github.com/PyDFLT/PyDFLT/actions/workflows/CI.yml)


PyDFLT is a (to be published) Python package for benchmarking decision-focused learning (DFL) algorithms.
In the near future, we will publish PyDFLT on the Python Package Index, after which you can install it by running:

`pip install pydflt`

For now, this repository is still in development. Please see the instructions below for contributing:

### Installation with uv (after cloning or forking the repository)

We make use of uv (https://github.com/astral-sh/uv) for the installation and testing.

`cd decision-focused-learning-codebase`

`uv sync --all-extras --all-groups`

`source .venv/bin/activate`

`uv run pytest`

### Before committing / pushing

We make use of pre-commit (https://pre-commit.com/) for formatting of our files.

`uv run pre-commit install`

`uv run pre-commit run --all-files`

### Documentation

We use Sphinx (https://www.sphinx-doc.org/en/master/) for the documentation.  The Makefile in this directory can be used to build the documentation.

You can run `uv run make html --directory=docs` rom the project root as well, which will build the documentation in the exact same way as it will be displayed on the website.

Then, go to docs/build/html/api/src.html and drag the file into a browser.

### Using Weights & Biases
If you want to use Weights & Biases, either set an environment variable named `WANDB_KEY` with your key,
or create a `.env` file with `WANDB_KEY = 'your-key-here'`.

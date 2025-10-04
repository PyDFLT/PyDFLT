[![CI](https://github.com/PyDFLT/PyDFLT/actions/workflows/CI.yml/badge.svg)](https://github.com/PyDFLT/PyDFLT/actions/workflows/CI.yml)

![alt text](https://github.com/PyDFLT/PyDFLT/blob/update-readme/images/logo.png?raw=true)


## A Python-based Decision-Focused Learning Toolbox 
**PyDFLT** is designed to help researchers apply and develop Decision Focused Learning (DFL) tools in Python.
In the near future, we will publish PyDFLT on the Python Package Index, after which you can install it by running:

`pip install pydflt`

### Documentation

Documentation can be found here.

### Contributing
If you want to contribute, you can fork the repository and send a pull request. We make use of **uv** (https://github.com/astral-sh/uv) for the installation and testing. Install uv [here](https://docs.astral.sh/uv/getting-started/installation/). To create the virtual environment:

`uv sync --all-extras --all-groups`

#### Before committing

We make use of **pre-commit** (https://pre-commit.com/) and **pytest** to ensure code is consistent and functioning properly. Both are part of the dev dependencies and therefore installed in the virtual environment. Before committing make sure to run both:

`uv run pre-commit run --all-files`

`uv run pytest`

#### Documentation

We use **Sphinx** (https://www.sphinx-doc.org/en/master/) for the documentation.  The Makefile in this directory can be used to build the documentation.

You can run `uv run make html --directory=docs` rom the project root as well, which will build the documentation in the exact same way as it will be displayed on the website.

Then, go to docs/build/html/api/src.html and drag the file into a browser.


### Using Weights & Biases
If you want to use Weights & Biases, either set an environment variable named `WANDB_KEY` with your key,
or create a `.env` file with `WANDB_KEY = 'your-key-here'`.

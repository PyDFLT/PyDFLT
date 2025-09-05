# PyDFLT

[![CI](https://github.com/NoahJSchutte/decision-focused-learning-codebase/actions/workflows/CI.yml/badge.svg)](https://github.com/NoahJSchutte/decision-focused-learning-codebase/actions/workflows/CI.yml)


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

### Running experiments on HPC cluster [outdated]
Step one is to make sure the repository 'delftblue-dfl-codebase' is moved to DelftBlue. We suppose that the repo is located in Desktop. Then on DelftBlue a virtual environment needs to be created with all the requirements installed. On DelftBlue I created a folder for the environments and locate the new environment here (/home/kvandenhouten/venvs/delftblue-dfl-codebase).

Move to DelftBLue:
```bash
cd Desktop
scp -pr delftblue-dfl-codebase kvandenhouten@login.delftblue.tudelft.nl:~/
```

Login:
```bash
ssh kvandenhouten@login.delftblue.tudelft.nl
```

Load 2024r1:
```bash
module load 2024r1
```

Load Python:
```bash
module load python/3.10.12
```

Create venv where you want:
```bash
python -m venv /home/kvandenhouten/venvs/delftblue-dfl-codebase
```

Activate venv
```bash
source /home/kvandenhouten/venvs/delftblue-dfl-codebase/bin/activate
```

Load pip
```bash
module load py-pip
```

Install packages from requirements.txt
```bash
python -m pip install -r /home/kvandenhouten/delftblue-dfl-codebase/requirements.txt
```

Go to the directory of the shell script
```bash
cd delftblue-dfl-codebase
```

Run bash script
```bash
sbatch shell_script.sh
```

Make sure the shell script has the correct venv, so for me the following line is in the shell script
```bash
source /home/kvandenhouten/venvs/delftblue-dfl-codebase/venv/bin/activate
```

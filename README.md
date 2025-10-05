[![CI](https://github.com/PyDFLT/PyDFLT/actions/workflows/CI.yml/badge.svg)](https://github.com/PyDFLT/PyDFLT/actions/workflows/CI.yml)

![alt text](https://github.com/PyDFLT/PyDFLT/blob/main/images/logo.png?raw=true)


## A Python-based Decision-Focused Learning Toolbox
**PyDFLT** is designed to help researchers apply and develop Decision Focused Learning (DFL) tools in Python. It uses **CVXPYLayers** [1] for differentiable models, **PyEPO** [2] for models with a linear objective and has an implementation of **SFGE** [3] and **Lancer** [4]. To help with research, it supports Weights & Biases (https://wandb.ai/) and Optuna (https://optuna.org).
In the near future, we will publish PyDFLT on the Python Package Index, after which you can install it by running:

`pip install pydflt`

### Documentation

Documentation can be found https://pydflt.github.io/documentation.

### Contributing
If you want to contribute, you can fork the repository and send a pull request. We make use of **uv** (https://github.com/astral-sh/uv) for the installation and testing. Install uv [here](https://docs.astral.sh/uv/getting-started/installation/). To create the virtual environment:

`uv sync --all-extras --all-groups`

Notice that your IDE might automatically create the environment, but does only install the basic package dependencies. Make sure to run above command to install all dependencies.

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


### References

[1] Akshay Agrawal, Brandon Amos, Shane Barratt, Stephen Boyd, Steven Diamond, and J Zico Kolter. Differentiable convex optimization layers. Advances in neural information processing systems, 32, 2019. doi:10.48550/arXiv.1910.12430.

[2] Bo Tang and Elias B. Khalil. Pyepo: a pytorch-based end-to-end predict-then-optimize library for linear and integer programming. Mathematical Programming Computation, 16(3):297–335, 2024. doi:10.1007/s12532-024-00255-x.

[3] Mattia Silvestri, Senne Berden, Jayanta Mandi, Ali ˙Irfan Mahmuto˘gulları, Maxime Mulamba, Allegra De Filippo, Tias Guns, and Michele Lombardi. Score function gradient estimation to widen the applicability of decision-focused learning. CoRR, abs/2307.05213, 2023. doi:10.48550/arXiv.2307.05213.

[4] Arman Zharmagambetov, Brandon Amos, Aaron Ferber, Taoan Huang, Bistra Dilkina, and Yuandong Tian. Landscape surrogate: Learning decision losses for mathematical optimization under partial information. Advances in Neural Information Processing Systems, 36:27332–27350, 2023. doi:10.48550/arXiv.2307.08964.

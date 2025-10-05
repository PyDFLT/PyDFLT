.. image:: _static/logo.png
   :alt: PyDFLT logo (light)
   :class: hero-logo hero-logo--light

.. image:: _static/logo_dark.png
   :alt: PyDFLT logo (dark)
   :class: hero-logo hero-logo--dark

**PyDFLT** is designed to help researchers apply and develop Decision Focused Learning (DFL) tools in Python. It uses **CVXPYLayers** [1] for differentiable models, **PyEPO** [2] for models with a linear objective and has an implementation of **SFGE** [3] and **Lancer** [4]. To help with research, it supports Weights & Biases (https://wandb.ai/) and Optuna (https://optuna.org).


DFL aims at learning a predictive model that maximizes the decision quality of a downstream decision-making task or optimization model. This problem as a whole is named a predict & optimize or contextual optimization problem. In this setting, the optimization model has some uncertain parameters that correlate with some context or features. If seen as a pure predictive problem, one could aim to maximize predictive accuracy. However, since we know we will use the predictions to make decisions, we aim at maximizing decision quality.


Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :glob:

   examples/*

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/src


References:

[1] Akshay Agrawal, Brandon Amos, Shane Barratt, Stephen Boyd, Steven Diamond, and J Zico Kolter. Differentiable convex optimization layers. Advances in neural information processing systems, 32, 2019. doi:10.48550/arXiv.1910.12430.

[2] Bo Tang and Elias B. Khalil. Pyepo: a pytorch-based end-to-end predict-then-optimize library for linear and integer programming. Mathematical Programming Computation, 16(3):297–335, 2024. doi:10.1007/s12532-024-00255-x.

[3] Mattia Silvestri, Senne Berden, Jayanta Mandi, Ali ˙Irfan Mahmuto˘gulları, Maxime Mulamba, Allegra De Filippo, Tias Guns, and Michele Lombardi. Score function gradient estimation to widen the applicability of decision-focused learning. CoRR, abs/2307.05213, 2023. doi:10.48550/arXiv.2307.05213.

[4] Arman Zharmagambetov, Brandon Amos, Aaron Ferber, Taoan Huang, Bistra Dilkina, and Yuandong Tian. Landscape surrogate: Learning decision losses for mathematical optimization under partial information. Advances in Neural Information Processing Systems, 36:27332–27350, 2023. doi:10.48550/arXiv.2307.08964.

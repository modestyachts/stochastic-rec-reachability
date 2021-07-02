# Quantifying Reachability in Recommender Systems

This repository contains the code associated with the paper *Quantifying Availability and Discovery in Recommender Systems via Stochastic Reachability* by [Mihaela Curmei](https://mcurmei627.github.io/), [Sarah Dean](https://people.eecs.berkeley.edu/~sarahdean/), and [Benjamin Recht](https://people.eecs.berkeley.edu/~brecht/index.html).

## Dependencies

The code is tested with python 3.7.4.

We make use of [RecLab](https://berkeley-reclab.github.io/) for data and recommenders. To install RecLab with these recommenders, it is necessary to have `g++` 5.0 or higher and `python3-dev` installed. Then run
```
pip install reclab[recommenders]==0.1.2
```

Then running `pip install -r requirements.txt` is sufficient to install the remaining dependencies.

## Getting Started

For an example of how reachability can be computed in a toy setting, see [`Reachability Demo.ipynb`](Reachability%20Demo.ipynb).

## Reproducing Experiments

To reproduce experiments presented in the paper *Quantifying Availability and Discovery in Recommender Systems via Stochastic Reachability*:
1. Install dependencies.
2. (Optional) Download and preprocess MIND data by running the [`utils/get_mind_data.py`](utils/get_mind_data.py).
3. (Optional) Run [`icml2021/run_experiments.py`](icml2021/run_experiments.py).
4. Plot figures and create tables using the notebook files in [`icml2021/`](icml2021/). (You can recreate the plots from the static files in the [`results`](icml2021/results) folder.)
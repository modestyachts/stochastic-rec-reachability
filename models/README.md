## Model folder

This folder contains a dictionary of optimal parameters for each recommendation environment. The settings in [`best_params.json`](best_params.json) were found via hyper-parameter tuning and were used in our experiments. Recall that each environment consists of a dataset and a recommender.

Upon first training a recommender for a dataset the default behaviour is to save a `.pkl` file containing the trained recommender model. This makes subsequent instantiations of [Environments](../utils/environment.py) much faster.

"""Implements ModelTuner, a class that automatically tunes a model's hyperparameters."""
import datetime
import os

import numpy as np
import pandas as pd

import json


class ModelTuner:
    """The tuner allows for easy tuning.

    Provides functionality for n-fold cross validation to
    assess the performance of various model parameters.

    Parameters
    ----------
    data : triple of iterables
        The (user, items, ratings) data.
    default_params : dict
        Default model parameters.
    recommender_class : class Recommender
        The class of the recommender on which we wish to tune parameters.
    n_fold : int, optional
        The number of folds for cross validation.
    verbose : bool, optional
        Mode for printing results, defaults to True.
    data_dir : str
        The name of the directory under which to store the tuning logs.
    environment_name : str
        The name of the environment snapshot on which we are tuning the recommender.
    recommender_name : str
        The name of the recommender for which we are storing the tuning logs.
    overwrite : bool
        Whether to overwrite tuning logs if they already exist.
    fold repeat : int or None
        The number of times to repeat each fold.
    """

    def __init__(self,
                 data,
                 default_params,
                 recommender_class,
                 n_fold=5,
                 verbose=True,
                 data_dir=None,
                 environment_name=None,
                 recommender_name=None,
                 overwrite=False,
                 mse_only=True,
                 fold_repeat=None,
                 num_evaluations=0,
                 seed=0):
        """Create a model tuner."""
        self.users, self.items, self.ratings = data
        self.default_params = default_params
        self.num_users = len(self.users)
        self.num_items = len(self.items)
        self.verbose = verbose
        self.recommender_class = recommender_class
        self.data_dir = data_dir
        self.environment_name = environment_name
        self.recommender_name = recommender_name
        self.overwrite = overwrite
        self.num_evaluations = num_evaluations
        self.mse_only = mse_only

        # hacky way to shorten CV
        self.n_fold_repeat = self.n_fold if fold_repeat is None else fold_repeat

        np.random.seed(seed)
        self._generate_n_folds(n_fold)

    def _generate_n_folds(self, n_fold):
        """Generate indices for n folds."""
        indices = np.random.permutation(len(self.ratings))
        size_fold = len(self.ratings) // n_fold
        self.train_test_folds = []
        for i in range(n_fold):
            test_ind = indices[i*size_fold:(i+1)*size_fold]
            train_ind = np.append(indices[:i*size_fold], indices[(i+1)*size_fold:])
            self.train_test_folds.append((train_ind, test_ind))

    def evaluate(self, params):
        """Train and evaluate a model for parameter setting."""
        # constructing model with given parameters
        defaults = {key: self.default_params[key] for key in self.default_params.keys()
                    if key not in params.keys()}
        recommender = self.recommender_class(**defaults, **params)
        metrics = []
        if self.verbose:
            print('Evaluating:', params)
        for i in range(self.n_fold_repeat):
            fold = self.train_test_folds[i]
            if self.verbose:
                print('Fold {}/{}, '.format(i+1, len(self.train_test_folds)),
                      end='')
            train_ind, test_ind = fold

            # splitting data dictionaries
            keys = list(self.ratings.keys())
            ratings_test = {key: self.ratings[key] for key in [keys[i] for i in test_ind]}
            ratings_train = {key: self.ratings[key] for key in [keys[i] for i in train_ind]}

            recommender.reset(self.users, self.items, ratings_train)

            # constructing test inputs
            ratings_to_predict = []
            true_ratings = []
            for user, item in ratings_test.keys():
                true_r, context = self.ratings[(user, item)]
                ratings_to_predict.append((user, item, context))
                true_ratings.append(true_r)
            predicted_ratings = recommender.predict(ratings_to_predict)

            
            mse = np.mean((predicted_ratings - true_ratings)**2)
            if self.verbose:
                print('mse={}, rmse={}'.format(mse, np.sqrt(mse)))
            if not self.mse_only:
                # Note that this is not quite a traditional NDCG
                # normally we would consider users individually
                # this computation lumps all predictions together.
                def get_ranks(array):
                    array = np.array(array)
                    temp = array.argsort()
                    ranks = np.empty_like(temp)
                    ranks[temp] = np.arange(len(array))
                    return len(ranks) - ranks

                def get_dcg(ranks, relevances, cutoff=5):
                    dcg = 0
                    for rank, relevance in zip(ranks, relevances):
                        if rank <= cutoff:
                            dcg += relevance / np.log2(rank+1)
                    return dcg

                cutoff = int(len(true_ratings) / 5)
                idcg = get_dcg(get_ranks(true_ratings), true_ratings, cutoff=cutoff)
                dcg = get_dcg(get_ranks(predicted_ratings), true_ratings, cutoff=cutoff)
                ndcg = dcg / idcg
                if self.verbose:
                    print('dcg={}, ndcg={}'.format(dcg, ndcg))
                metrics.append(np.array([mse, ndcg]))
            else:
                metrics.append(np.array([mse]))

        if self.verbose:
            print('Average Metric:', np.mean(metrics, axis=0))
        return np.array(metrics)

    def evaluate_grid(self, **params):
        """Train over a grid of parameters."""
        def recurse_grid(fixed_params, grid_params):
            if len(grid_params) == 0:
                result = fixed_params
                result['metric'] = self.evaluate(fixed_params)
                result['average_metric'] = np.mean(result['metric'], axis=0)
                return [result]

            curr_param, curr_values = list(grid_params.items())[0]
            new_grid_params = dict(grid_params)
            del new_grid_params[curr_param]
            results = []
            for value in curr_values:
                results += recurse_grid({**fixed_params, curr_param: value}, new_grid_params)
            return results

        results = recurse_grid({}, params)
        results = pd.DataFrame(results)
        if self.data_dir is not None:
            self.save(results, params)
        self.num_evaluations += 1
        return results

    def save(self, results, params):
        """Save the current hyperparameter tuning results."""
        dir_name = os.path.join(self.data_dir, self.environment_name, self.recommender_name,
                                'tuning', 'evaluation_' + str(self.num_evaluations), '')
        if os.path.isdir(dir_name):
            if not self.overwrite:
                if self.verbose:
                    print('Directory:', dir_name, 'already exists. Results will not be saved.')
                return
        else:
            os.makedirs(dir_name)

        if self.verbose:
            print('Saving to directory:', dir_name)
        info = {
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'recommender': self.recommender_name,
        }

        with open(os.path.join(dir_name, 'info.json'), 'w') as fp:
            json.dump(info, fp, sort_keys=True, indent=4)
        with open(os.path.join(dir_name, 'params.json'), 'w') as fp:
            json.dump(params, fp,sort_keys=True, indent=4)
        results.to_pickle(os.path.join(dir_name, 'results.pkl')) 

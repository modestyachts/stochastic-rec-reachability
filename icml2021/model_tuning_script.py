import numpy as np
import sys

import os
from reclab.recommenders import LibFM, KNNRecommender
from reclab.recommenders.sparse import EASE

from tuner import ModelTuner

sys.path.append('../utils/')
from config import MODELPATH
from environment import AggregateMIND, LastFMEnvironment,  MovieLensEnvironment, ML_DATASETS, YouTubeEnvironment

# Parsing Arguments
if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
    if len(sys.argv) > 2:
        models = sys.argv[2:]
    else:
        models = ['ease', 'knn', 'libfm']
else:
    dataset_name = 'ml-100k'
print(dataset_name, models)

# Defining parameter search ranges
refined_grid = {}
refined_grid['ml-100k','ease'] = [dict(binarize=[True], lam=np.logspace(2.1, 3.5, 20).tolist()),
                                  dict(binarize=[False], lam=np.logspace(3.1, 4.5, 20).tolist())]
refined_grid['ml-100k','knn'] = [dict(shrinkage=np.linspace(0, 100, 10).tolist(),
                                      neighborhood_size=np.linspace(10, 200, 20, dtype=int).tolist())]
refined_grid['ml-100k','libfm'] = [dict(reg=np.logspace(-1.3, -0.9, 4).tolist(),
                                   learning_rate=np.linspace(2e-4, 0.025, 10).tolist())]
refined_grid['ml-1m','ease'] = [dict(binarize=[True], lam=np.logspace(2.1, 3.5, 20).tolist()),
                                  dict(binarize=[False], lam=np.logspace(3.1, 4.5, 20).tolist())]
refined_grid['ml-1m','knn'] = [dict(shrinkage=np.linspace(0, 100, 10).tolist(),
                                      neighborhood_size=np.linspace(10, 200, 20, dtype=int).tolist())]
refined_grid['ml-1m','libfm'] = [dict(reg=np.logspace(-1.3, -0.9, 4).tolist(),
                                   learning_rate=np.linspace(2e-4, 0.025, 10).tolist())]
refined_grid['lastfm-360k','ease'] = [dict(binarize=[False], lam=np.logspace(2.1, 3.5, 20).tolist())]
refined_grid['lastfm-360k','knn'] = [dict(shrinkage=np.linspace(300, 500, 10).tolist(),
                                      neighborhood_size=np.linspace(100, 2000, 20, dtype=int).tolist())]
refined_grid['lastfm-360k','libfm'] = [dict(reg=np.linspace(0.05, 0.25, 10).tolist(),
                                   learning_rate=np.linspace(0.02, 0.07, 10).tolist())]
refined_grid['youtube','ease'] = [dict(binarize=[False], lam=np.logspace(2, 3, 20).tolist())]
refined_grid['youtube','knn'] = [dict(shrinkage=[0],
                                      neighborhood_size=np.linspace(10, 200, 20, dtype=int).tolist())]
refined_grid['youtube','libfm'] = [dict(reg=np.logspace(-6, -3.5, 10).tolist(),
                                   learning_rate=np.linspace(0.03, 0.08, 10).tolist())]
refined_grid['mind_aggregate_small_dev_cat','libfm'] = [dict(reg=np.logspace(-1, -5, 20).tolist(),
                                   learning_rate=np.linspace(0.001, 0.12, 20).tolist())]
refined_grid['mind_aggregate_small_train_cat','libfm'] = [dict(reg=np.logspace(-1, -5, 20).tolist(),
                                   learning_rate=np.linspace(0.001, 0.12, 20).tolist())]
refined_grid['mind_aggregate_small_dev_subcat','libfm'] = [dict(reg=np.logspace(-4, -2, 10).tolist(),
                                   learning_rate=np.linspace(0.05, 0.12, 10).tolist())]
refined_grid['mind_aggregate_small_train_subcat','libfm'] = [dict(reg=np.logspace(-4, -2, 10).tolist(),
                                   learning_rate=np.linspace(0.05, 0.15, 10).tolist())]
# Loading data
if dataset_name in ML_DATASETS:
    data = MovieLensEnvironment(dataset_name)
elif dataset_name == 'lastfm-360k':
    data = LastFMEnvironment()
elif dataset_name == 'youtube':
    data = YouTubeEnvironment()
elif dataset_name.startswith('mind_aggregate'):
    name_parts = dataset_name.split('_')
    size = name_parts[2]
    kind = name_parts[3]
    agg_by = name_parts[4]
    data = AggregateMIND(size=size, kind=kind, agg_by=agg_by)
data.load_data()
users, items, ratings = data._users, data._items, data._observed_ratings_dict


## EASE
if 'ease' in models:
    recommender_class = EASE
    default_params = {}
    tuner = ModelTuner((users, items, ratings),
                default_params,
                recommender_class,
                n_fold=10,
                verbose=True,
                data_dir=MODELPATH,
                environment_name=dataset_name,
                recommender_name='EASE',
                overwrite=True,
                mse_only=True,
                fold_repeat=1,
                num_evaluations=0)
    ### gridding params
    for refined_params in refined_grid[dataset_name,'ease']:
        tuner.evaluate_grid(**refined_params) # refined

## User KNN
if 'knn' in models:
    recommender_class = KNNRecommender
    default_params = dict(user_based=False)
    tuner = ModelTuner((users, items, ratings),
                default_params,
                recommender_class,
                n_fold=10,
                verbose=True,
                data_dir=MODELPATH,
                environment_name=dataset_name,
                recommender_name='KNN',
                overwrite=True,
                mse_only=True,
                fold_repeat=1,
                num_evaluations=0)
    ### gridding params
    for refined_params in refined_grid[dataset_name,'knn']:
        tuner.evaluate_grid(**refined_params) # refined

## LibFM
if 'libfm' in models:
    recommender_class = LibFM
    default_params = dict(num_user_features=0,
                    num_item_features=0,
                    num_rating_features=0,
                    max_num_users=len(users),
                    max_num_items=len(items),
                    method='sgd',
                    num_iter=128, # following https://arxiv.org/pdf/1905.01395.pdf
                    num_two_way_factors=64, # following https://arxiv.org/pdf/1905.01395.pdf
                    init_stdev=1.0)
    tuner = ModelTuner((users, items, ratings),
                default_params,
                recommender_class,
                n_fold=10,
                verbose=True,
                data_dir=MODELPATH,
                environment_name=dataset_name,
                recommender_name='LibFM',
                overwrite=True,
                mse_only=True,
                fold_repeat=1,
                num_evaluations=0)

    ### gridding params
    for refined_params in refined_grid[dataset_name,'libfm']:
        tuner.evaluate_grid(**refined_params) # refined

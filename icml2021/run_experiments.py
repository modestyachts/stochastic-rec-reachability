"""Reproducting paper experiments"""
import json
import os
import sys

folder , _ = os.path.split(__file__)
utilspath = os.path.normpath(os.path.join(folder, '../utils'))
sys.path.append(utilspath)

from experiment_setup_utils import run_dataset_future_experiment, run_dataset_history_experiment, run_dataset_nextk_experiment
from environment import MovieLensEnvironment, LastFMEnvironment, AggregateMIND

# load user subset
with open('user_subset.json') as json_file:
    user_subset_dict = json.load(json_file)

# ===================================================================================================
# ========== MovieLens 1M Experiments (LibFM) ======================================================
# ===================================================================================================
env = MovieLensEnvironment('ml-1m')
# Initialize LibFM recommender
env.initialize('libfm', save=True) # <- switch to False if you are repeating this experimental run
env.compute_dense_predictions()
run_users = user_subset_dict['user_subset_ml1m']

# Run reachability experiments
run_dataset_history_experiment(env, target_densities = [1], action_counts= [5, 10, 20], goal_counts = [500],
                               betas = [1, 2, 4, 10], title_str='history', run_users = run_users)
run_dataset_future_experiment(env, target_densities = [1], action_counts= [5, 10, 20], goal_counts = [500],
                               betas = [1, 2, 4, 10], title_str='future', run_users = run_users)
run_dataset_nextk_experiment(env, target_densities = [1], action_counts= [5, 10, 20], goal_counts = [500],
                             betas = [1, 2, 4, 10], title_str='future', run_users = run_users)

# ===================================================================================================
# ========== MovieLens 1M Experiments (KNN) ========================================================
# ===================================================================================================
env = MovieLensEnvironment('ml-1m')
# Initialize KNN recommender
env.initialize('knn', save=True) # <- switch to False if you are repeating this experimental run
env.compute_dense_predictions()
run_users = user_subset_dict['user_subset_ml1m']

# Run reachability experiments
run_dataset_history_experiment(env, target_densities = [1], action_counts= [5, 10, 20], goal_counts = [500],
                               betas = [1, 2, 4, 10], title_str='history', run_users = run_users)
run_dataset_future_experiment(env, target_densities = [1], action_counts= [5, 10, 20], goal_counts = [500],
                               betas = [1, 2, 4, 10], title_str='future', run_users = run_users)
run_dataset_nextk_experiment(env, target_densities = [1], action_counts= [5, 10, 20], goal_counts = [500],
                             betas = [1, 2, 4, 10], title_str='future', run_users = run_users)

# ===================================================================================================
# ==========  LastFM Experiments (LibFM) ============================================================
# ===================================================================================================
env = LastFMEnvironment()
# Initialize LibFM recommender
env.initialize('libfm', save=True) # <- switch to False if you are repeating this experimental run
env.compute_dense_predictions()
run_users = user_subset_dict['user_subset_lastfm']

# Run reachability experiments
run_dataset_history_experiment(env, target_densities = [1], action_counts= [5, 10, 20], goal_counts = [500],
                               betas = [1, 2, 4, 10], title_str='history', common_goals = True, run_users = run_users)
run_dataset_future_experiment(env, target_densities = [1],  action_counts= [5, 10, 20], goal_counts = [500],
                               betas = [1, 2, 4, 10], title_str='future', common_goals = True, run_users = run_users)
run_dataset_nextk_experiment(env, target_densities = [1], action_counts= [5, 10, 20], goal_counts = [500],
                             betas = [1, 2, 4, 10], title_str='nextk', common_goals = True, run_users = run_users)

# ===================================================================================================
# ==========  LastFM Experiments (KNN) ==============================================================
# ===================================================================================================
env = LastFMEnvironment()
# Initialize KNN recommender
env.initialize('knn', save=True) # <- switch to False if you are repeating this experimental run
env.compute_dense_predictions()
run_users = user_subset_dict['user_subset_lastfm']

# Run reachability experiments
run_dataset_history_experiment(env, target_densities = [1], action_counts= [5, 10, 20], goal_counts = [500],
                               betas = [1, 2, 4, 10], title_str='history', common_goals = True, run_users = run_users)
run_dataset_future_experiment(env, target_densities = [1],  action_counts= [5, 10, 20], goal_counts = [500],
                               betas = [1, 2, 4, 10], title_str='future', common_goals = True, run_users = run_users)
run_dataset_nextk_experiment(env, target_densities = [1], action_counts= [5, 10, 20], goal_counts = [500],
                             betas = [1, 2, 4, 10], title_str='nextk', common_goals = True, run_users = run_users)

# ===================================================================================================
# ==========  MIND Experiments (LibFM) ==============================================================
# ===================================================================================================
env = AggregateMIND(size='small', kind='dev', agg_by='subcat')
# Initialize LibFM recommender
env.initialize('libfm', save=True) # <- switch to False if you are repeating this experimental run
env.compute_dense_predictions()

# Run reachability experiments
run_dataset_history_experiment(env, target_densities = [1], action_counts= [5, 10, 20], goal_counts = [500],
                               betas = [1, 2, 4, 10], title_str='history', common_goals = True)
run_dataset_future_experiment(env, target_densities = [1],  action_counts= [5, 10, 20], goal_counts = [500],
                               betas = [1, 2, 4, 10], title_str='future', common_goals = True)
run_dataset_nextk_experiment(env, target_densities = [1], action_counts= [5, 10, 20], goal_counts = [500],
                             betas = [1, 2, 4, 10], title_str='nextk', common_goals = True)

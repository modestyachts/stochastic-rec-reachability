""" Functionality for testing item availability
Experiment instances are named run_***_experiment"""
from collections import defaultdict
from datetime import datetime
import itertools
import json
import numpy as np
import os
import pickle
import re
from scipy.sparse import csr_matrix
import scipy.special
from scipy.stats import skewnorm
import sys
from time import time

from config import EXPPATH
from environment import RandomEnvironment, RandomLatentEnvironment, ML_DATASETS
from helper_utils import negate_list, negate_matrix, get_non_zero_entries, softmax_transform, make_sparse_matrix
from mask_utils import make_nextk_action_mask, make_random_mask, make_common_columns_mask
import stochastic_reachability_utils as sru

# ========== Basic functionality for running reachability experiments ============
def run_reachability_experiment(env, beta, target_mask, action_mask, goal_mask = None,
                                save_results=False, save_dir=None, returns = True,
                                run_users = None, action_range = (0,5)):

    """ Computes two N_users x N_items matrices which contain the max
        rho and baseline rho values.

    Parameters
    ----------
    env : Environment containing data and recommender
    beta : float
        softmax policy  parameter
    target_mask : N_users x N_items bool matrix
        True - item i is target item for user u
    action_mask : N_users x N_items bool matrix
        True - item i is action item for user u
    goal_mask : N_users x N_items bool matrix
        True - item i is target item for user u
    save_results : bool, optional
        if True, saves results in a folder on a per-user basis
        by  default False
    save_dir : path, optional
        directory where to save the rech reachability output
        by default False
    returns : bool, optional
        if True the function returns experimental outputs
        by default False
    run_users : list, optional
        subset of users for whom reachability analysis is performed,
        by default None
    action_range : tuple, optional
        lower and upper limit of the action space,
        by default (0,5)

    Returns
    ------- (if returns= = True)
    max_rho_mat : N_users x N_items matrix
        entry (u, i) contains the max achievable selection
        probability for item i by user u (under the action model)
    baseline_rho_mat : N_users x N_items matrix
        entry (u, i) contains the baseline selection probability
        for item i by user u (without strategic modifications)
    opt_actions : list of 2d arrays
        opt_actions[i] is Num_target_items x Num_action_items 2d array
        containing the optimal actions (ratings) for the action items
        corresponding to user i
    max_ranks : list
        entry (u,i) contains the best achievable rank for the item i
        among target items for user u.
    baseline_ranks : list
        entry (u,i) contains the baseline rank for the item i
        among target items for user u.
    """
    if goal_mask is None:
        goal_mask = target_mask
    seen_dict = get_non_zero_entries(env.seen_mask, how='row')
    action_dict = get_non_zero_entries(action_mask, how='row')
    target_dict = get_non_zero_entries(target_mask, how='row')
    goal_dict = get_non_zero_entries(goal_mask, how ='row')
    if save_results:
        assert save_dir is not None
        assert os.path.exists(save_dir), "First create a directory to save the experiments"
    if returns:
        max_rhos = []
        baseline_rhos = []
        opt_actions = []
        max_ranks = []
        baseline_ranks = []
    if run_users is None:
        user_subset = set(action_dict.keys())
    else:
        user_subset = run_users.intersection(set(action_dict.keys()))
    for uid in user_subset:
        tic = time()
        seen_iids = seen_dict[uid]
        action_iids = action_dict[uid]
        target_iids = target_dict[uid]
        goal_iids = goal_dict[uid]

        user_baseline_actions = env.dense_predictions[uid, action_iids]
        user_baseline_rhos, user_baseline_ranks = get_user_baseline_rho(uid, goal_iids, target_iids, action_iids, beta,env.trained_model_dict,
                                                                        user_baseline_actions, action_range=action_range)

        (user_max_rhos, user_opt_actions, user_max_ranks, _) = get_user_max_rho(uid, goal_iids, target_iids, action_iids, beta, env.trained_model_dict,
                                               return_opt_actions=True, return_ranks=True, return_updated_factors=False, action_range=action_range)
        if save_results:
            data = defaultdict()
            data['seen_iids'] = seen_iids
            data['target_iids'] = target_iids
            data['goal_iids'] = goal_iids
            data['action_iids'] = action_iids
            data['baseline_rhos'] = user_baseline_rhos
            data['max_rhos'] = user_max_rhos
            data['opt_actions'] = user_opt_actions
            data['max_ranks'] = user_max_ranks
            data['baseline_ranks'] = user_baseline_ranks
            data['uid'] = uid

            fname = "user_{}.pkl".format(uid)
            fpath = os.path.join(save_dir, fname)
            pickle.dump(data, open(fpath,"wb"))

        if returns:
            baseline_rhos.extend(user_baseline_rhos)
            max_rhos.extend(user_max_rhos)
            opt_actions.extend(user_opt_actions)
            max_ranks.extend(user_max_ranks)
            baseline_ranks.extend(user_baseline_ranks)
        toc = time()
        print("\t\t\t User {} out of {} was processed in {:.2f} minutes".format(uid, len(user_subset), (toc-tic)/60))

    if returns:
        max_rho_mat = make_sparse_matrix(target_dict, shape = (env.num_user, env.num_item), values = max_rhos)
        baseline_rho_mat = make_sparse_matrix(target_dict, shape = (env.num_user, env.num_item), values = baseline_rhos)
        return(max_rho_mat, baseline_rho_mat, opt_actions, max_ranks, baseline_ranks)

def get_user_baseline_rho(uid, goal_iids, target_iids, action_iids, beta, trained_model, baseline_actions, action_range = (0,5)):
    """ Wrapper method for computing max rhos for a user given target ids and action ids

    Parameters
    ----------
    uid : int
        The id of the user for which max rhos (max reachability is being computed)
    goal_iids : iterable of int
        list containing the ids of the goal items (not relative goal_ids)
    target_iids : iterable of int
        list containing the ids of the target items
        must be a superset of goal_iids
    action_iids : interable of int
        list containing the ids of the action items
    beta : float
        beta parameter of the item selection policy
    trained_model : dict
        dictionary containing the trained model
    baseline_actions : iterable of int (num_action_items)
        choice of values of the action items used to update user factors
        the order of the columns is assumed to be the same as the order

    Returns
    -------
    baseline_rho_vec: array of size len(goal_iids)
        an array that contains the baseline rho associated to each goal item
        with respect to the target items under the baseline action
    baseline_rank_vec: array of size len(goal_iids)
        an array that contains the positional rank associated for which goal item
        with respect to the target items asumming that the user takes the baseline action
    """

    assert set(goal_iids).issubset(target_iids)

    # set up score update parameters based on model kind
    kind = trained_model['kind']
    assert kind in ['libfm', 'knn', 'ease']
    if kind == 'libfm':
        # TODO specify regularization and step size as appropriate!
        Bs, cs = sru.setup_stochastic_reachability_libfm(
                                                trained_model['item_factors'],
                                                trained_model['item_bias'],
                                                trained_model['user_bias'][uid],
                                                trained_model['global_bias'],
                                                trained_model['user_factors'][uid],
                                                target_iids, action_iids, method='sgd')
    if kind == 'knn':
        Bs, cs = sru.setup_stochastic_reachability_knn(trained_model["ratings"][uid],
                                                   trained_model["weights"],
                                                   trained_model["means"],
                                                   target_iids,
                                                   action_iids)
    if kind == 'ease':
        Bs, cs = sru.setup_stochastic_reachability_ease(trained_model["ratings"][uid],
                                                trained_model["weights"],
                                                target_iids,
                                                action_iids)
    baseline_actions = np.clip(baseline_actions, *action_range)
    baseline_scores = Bs@baseline_actions.reshape(-1, 1) + cs.reshape(-1, 1)
    relative_goal_items = [target_iids.index(goal_id) for goal_id in goal_iids]
    baseline_rhos = softmax_transform(baseline_scores, beta = beta, ignore_zero = False)
    baseline_ranks = np.argsort(baseline_rhos)
    baseline_rho_vec = baseline_rhos[relative_goal_items]
    baseline_rank_vec = baseline_ranks[relative_goal_items]
    return baseline_rho_vec, baseline_rank_vec

def get_user_max_rho(uid, goal_iids, target_iids, action_iids, beta, trained_model,
                    return_opt_actions=False, return_ranks=False, return_updated_factors=False,
                    parallel=True, action_range = (0,5), **kwargs):
    """ Wrapper method for computing max rhos for a user given target ids and action ids

    Parameters
    ----------
    uid : int
        The id of the user for which max rhos (max reachability is being computed)
    goal_iids : iterable of int
        list containing the ids of the goal items
        (absolute goal ids (rec inner ids) not relative goal ids)
    target_iids : iterable of int
        list containing the ids of the target items
        must be a superset of goal_iids
    action_iids : interable of int
        list containing the ids of the action items
    beta : float
        beta parameter of the item selection policy
    trained_model : dict
        dictionary containing the trained model
    return_opt_actions : bool, optional
        if True, the function returns the optimal action for each target item,
        by default False
    return_ranks : bool, optional
        if True, the function returns the rank of the goal items among target items
    return_updated_factors : bool, optional
        if True, the function returns the updated factors for the user after taking
        the action that maximizes thr rho of the target item, by default False
    parallel : bool, optional
        if True use parallel execution, by default True
    action_range : tuple, optional
        fraseable domain for user action (strategic rating).
        default (0,5)

    Returns
    -------
    max_rho_vec: array of size len(goal_iids)
        an array that contains the max rho associated to each goal item
    opt_user_ratings_mat: None/ 2D array of size len(target_iids) x len(action_iids)
        the i'th entry contains the optimal ratings for the action items that maximize the rho associated
        with the target item
    rank_vec: array of size len(goal_iids)
        an array that contains the rank associated to each goal item (among total target items)
    updated_user_factor_mat: None/2D array of size len(target_iids) x trained_model["latent_dim"]
        the i'th entry contains the updated item factors resulting from takin the optimal actions for
        maximizing the rho reachability of item i
    """
    assert set(goal_iids).issubset(target_iids)

    # set up score update parameters based on modek kind
    kind = trained_model['kind']
    assert kind in ['libfm', 'knn', 'ease']
    if kind == 'libfm':
        # TODO specify regularization and step size as appropriate!
        Bs, cs = sru.setup_stochastic_reachability_libfm(
                                                trained_model['item_factors'],
                                                trained_model['item_bias'],
                                                trained_model['user_bias'][uid],
                                                trained_model['global_bias'],
                                                trained_model['user_factors'][uid],
                                                target_iids, action_iids, method='sgd')
    if kind == 'knn':
        Bs, cs = sru.setup_stochastic_reachability_knn(trained_model["ratings"][uid],
                                                   trained_model["weights"],
                                                   trained_model["means"],
                                                   target_iids,
                                                   action_iids)
    if kind == 'ease':
        Bs, cs = sru.setup_stochastic_reachability_ease(trained_model["ratings"][uid],
                                                trained_model["weights"],
                                                target_iids,
                                                action_iids)

    if parallel:
        relative_goal_items = [target_iids.index(goal_id) for goal_id in goal_iids]
        max_rhos, opt_user_ratings, rho_ranks = sru.parsolve_maxprob_stochastic_reachability(relative_goal_items, Bs, cs, beta, action_range=action_range)
        max_rho_vec = [max_rhos[i] for i in relative_goal_items]
        opt_user_ratings_mat = [opt_user_ratings[i] for i in relative_goal_items]
        rank_vec = [rho_ranks[i] for i in relative_goal_items]
    else:
        # solve the reachability problem for each target item
        max_rho_vec = [] # array of length len(goal_iids)
        opt_user_ratings_mat = [] # matrix of dimension len(goal_iids) x len(action_iids)
        rank_vec = [] # array of length len(goal_iids)
        for goal_id in goal_iids:
            relative_iid = target_iids.index(goal_id)
            max_rho, opt_user_rating, rho_rank = sru.solve_maxprob_stochastic_reachability(relative_iid, Bs, cs, beta, action_range=action_range)
            max_rho_vec.append(max_rho)
            if return_opt_actions:
                opt_user_ratings_mat.append(opt_user_rating)
            rank_vec.append(rho_rank)

    if return_updated_factors and kind == 'libfm':
        updated_user_factor_mat = [] # matrix of dimension len(goal_iids) x latent_dim
        for opt_user_rating in opt_user_ratings_mat:
            updated_user_factor = sru.get_updated_user_factor(opt_user_rating,
                                                              action_iids,
                                                              trained_model['user_factors'][uid],
                                                              trained_model['item_factors'])
            updated_user_factor_mat.append(updated_user_factor)
    else:
        updated_user_factor_mat = None

    if not return_opt_actions:
        opt_user_ratings_mat = None
    if not return_ranks:
        rank_vec = None

    return (max_rho_vec, opt_user_ratings_mat, rank_vec, updated_user_factor_mat)

# ========== Dataset experiments =========
def run_dataset_history_experiment(env,
                                    action_counts = [10],
                                    goal_counts = None,
                                    target_densities = [1],
                                    betas = [1],
                                    title_str='history',
                                    run_users = None,
                                    common_goals = False,
                                    exppath = EXPPATH):
    """Stochastic reachability for random subset of history action items

    Parameters
    ----------
    env : Environment object
        instance of Environment
    target_densities : array of floats, optional
        target parameters to consider in the experiment, by default [1]
    action_counts : array of int, optional
        action parameters to consider in the experiment,
        number of editable history items, by default [10]
    goal_counts : iterable of int, optional
        a list the number of goal items to be considered, the goal items are random
        unseen items
    betas : array of floats, optional
        softmax selection probability parameters to consider, by default [1]
    title_str : str, optional
        title string for the json files, by default 'history'
    run_users : list, optional
        set of users for which to run experiments
        if None, then run reachability for all users,
        by default None
    common_goals : bool, optional
        if True, then the sampled goal items are identical for all users
        if False, then for each user the goal items are sampled independently.
        by default False
    exppath : str, path; optional
        by default EXPPATH (defined in config.py)

    """
    parent_dir = os.path.join(exppath, "{}_{}".format(env.dataset_name, env.recommender_model))
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    datestr = datetime.today().strftime('%Y-%m-%d-%H:%M')
    reach_exp_dir = os.path.join(parent_dir, "{}_{}".format(title_str, datestr))
    os.makedirs(reach_exp_dir)

    params = defaultdict()
    params['dataset'] = env.dataset_name
    params['recomender'] = env.recommender_model
    params['target_densities'] = target_densities
    params['action_counts'] = action_counts
    params['goal_counts'] = goal_counts
    params['betas'] = betas
    params['date'] = datestr
    pickle.dump(params, open(os.path.join(reach_exp_dir, 'params.pkl'),"wb"))

    for action_count_val in action_counts:
        print("Action count parameter: {}".format(action_count_val))
        action_mask = make_random_mask(negate_matrix(env.seen_mask), count=action_count_val, random_state=0)
        for target_density in target_densities:
            print("\tTarget parameter: {}".format(target_density))
            target_mask = make_random_mask(env.seen_mask, density = target_density, random_state=7)
            if goal_counts is None:
                goal_masks = [target_mask]
            else:
                goal_masks = []
                for goal_count in goal_counts:
                    if common_goals:
                        goal_masks.append(make_common_columns_mask(target_mask, nrow = env.num_user, ncol = env.num_item,
                                                                   count = goal_count, active_rows = run_users, random_state=9))
                    else:
                        goal_masks.append(make_random_mask(negate_matrix(target_mask), count = goal_count, random_state=9))

            for i, goal_mask in enumerate(goal_masks):
                if goal_counts is None:
                    goal_count = 'all'
                else:
                    goal_count = goal_counts[i]
                print("\t\tGoal count: {}".format(goal_count))
                for beta in betas:
                    data = defaultdict()
                    data['action_mask'] = action_mask
                    data['target_mask'] = target_mask
                    data['goal_mask'] = goal_mask
                    data['beta'] = beta
                    data['action_count'] = action_count_val
                    data['target_density'] = target_density
                    data['goal_count'] = goal_count
                    exp_dir = os.path.join(reach_exp_dir, "action_{}_goal_{}_target_{}_beta_{}".format(action_count_val, goal_count, target_density, beta))
                    if not os.path.exists(exp_dir):
                        os.makedirs(exp_dir)
                    pickle.dump(data, open(os.path.join(exp_dir, 'experiment.pkl'),"wb"))

                    tic = time()
                    run_reachability_experiment(env, beta, target_mask, action_mask, goal_mask = goal_mask,
                                                save_results=True, save_dir=exp_dir, returns = False, run_users=run_users)
                    toc = time()
                    run_time = toc-tic

                    print("\t\t\tBeta: {}; Runtime: {:.2f} min".format(beta, run_time/60))

def run_dataset_future_experiment(env,
                                    action_counts = [10],
                                    goal_counts = None,
                                    target_densities = [1],
                                    betas = [1],
                                    title_str='future',
                                    run_users = None,
                                    common_goals = False,
                                    exppath = EXPPATH):
    """Stochastic reachability for random action items that have not been previously seen

    Parameters
    ----------
    env : Environment object
        instance of Environment
    target_densities : array of floats, optional
        target parameters to consider in the experiment, by default [0.01]
    action_counts : array of int, optional
        action parameters to consider in the experiment, by default [10]
    goal_counts : iterable of int, optional
        a list the number of goal items to be considered
    betas : array of floats, optional
        softmax selection probability parameters to consider, by default [1]
    title_str : str, optional
        title string for the json files, by default 'future'
    run_users : list, optional
        set of users for which to run experiments
        if None, then run reachability for all users,
        by default None
    common_goals : bool, optional
        if True, then the sampled goal items are identical for all users
        if False, then for each user the goal items are sampled independently.
        by default False
    exppath : str, path; optional
        by default EXPPATH (defined in config.py)
    """
    parent_dir = os.path.join(exppath, "{}_{}".format(env.dataset_name, env.recommender_model))
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    datestr = datetime.today().strftime('%Y-%m-%d-%H:%M')
    reach_exp_dir = os.path.join(parent_dir, "{}_{}".format(title_str, datestr))
    os.makedirs(reach_exp_dir)

    params = defaultdict()
    params['dataset'] = env.dataset_name
    params['recomender'] = env.recommender_model
    params['target_densities'] = target_densities
    params['action_counts'] = action_counts
    params['goal_counts'] = goal_counts
    params['beta'] = betas
    params['date'] = datestr
    pickle.dump(params, open(os.path.join(reach_exp_dir, 'params.pkl'),"wb"))

    for action_count_val in action_counts:
        print("Action count parameter: {}".format(action_count_val))
        action_mask = make_random_mask(env.seen_mask, count=action_count_val, random_state=0)
        for target_density in target_densities:
            print("\tTarget parameter: {}".format(target_density))
            target_mask = make_random_mask(env.seen_mask+action_mask, density = target_density, random_state=7)
            if goal_counts is None:
                goal_masks = [target_mask]
            else:
                goal_masks = []
                for goal_count in goal_counts:
                    if common_goals:
                        goal_masks.append(make_common_columns_mask(target_mask, nrow = env.num_user, ncol = env.num_item,
                                                                   count = goal_count, active_rows = run_users, random_state = 18))
                    else:
                        goal_masks.append(make_random_mask(negate_matrix(target_mask), count = goal_count, random_state = 18))

            for i, goal_mask in enumerate(goal_masks):
                if goal_counts is None:
                    goal_count = 'all'
                else:
                    goal_count = goal_counts[i]
                print("\t\tGoal count: {}".format(goal_count))
                for beta in betas:
                    data = defaultdict()
                    data['action_mask'] = action_mask
                    data['target_mask'] = target_mask
                    data['goal_mask'] = goal_mask
                    data['beta'] = beta
                    data['action_count'] = action_count_val
                    data['target_density'] = target_density
                    data['goal_count'] = goal_count
                    exp_dir = os.path.join(reach_exp_dir, "action_{}_goal_{}_target_{}_beta_{}".format(action_count_val, goal_count, target_density, beta))
                    if not os.path.exists(exp_dir):
                        os.makedirs(exp_dir)
                    pickle.dump(data, open(os.path.join(exp_dir, 'experiment.pkl'),"wb"))

                    tic = time()
                    run_reachability_experiment(env, beta, target_mask, action_mask, goal_mask = goal_mask,
                                                save_results=True, save_dir=exp_dir, returns = False, run_users=run_users)
                    toc = time()
                    run_time = toc-tic

                    print("\t\t\tBeta: {}; Runtime: {:.2f} min".format(beta, run_time/60))

def run_dataset_nextk_experiment(env,
                                action_counts = [10],
                                goal_counts = None,
                                target_densities = [1],
                                betas = [1],
                                title_str= 'next_k',
                                run_users = None,
                                common_goals = False,
                                exppath = EXPPATH):
    """Main function for running stochastic reachability experiments,
    where the action items are the

    Parameters
    ----------
    env : Environment object
        instance of Environment object
    action_counts : iterable of int, optional
        a list with number of next k items to be used as action items in the experiment
    goal_counts : iterable of int, optional
        a list the number of goal items to be considered
    target_densities : iterable of float, optional
        a list of target parameters to consider in the experiment, by default [1] (full unseen item set)
    betas : list, optional
        a list of softmax selection probability parameters to consider, by default [1]
        title_str : str, optional
    title string for the json files, by default 'next_k'
    run_users : list, optional
        set of users for which to run experiments
        if None, then run reachability for all users,
        by default None
    common_goals : bool, optional
        if True, then the sampled goal items are identical for all users
        if False, then for each user the goal items are sampled independently.
        by default False
    exppath : str, path; optional
        by default EXPPATH (defined in config.py)
    """

    parent_dir = os.path.join(exppath, "{}_{}".format(env.dataset_name, env.recommender_model))
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    datestr = datetime.today().strftime('%Y-%m-%d-%H:%M')
    reach_exp_dir = os.path.join(parent_dir, "{}_{}".format(title_str, datestr))
    os.makedirs(reach_exp_dir)
    params = defaultdict()
    params['dataset'] = env.dataset_name
    params['recomender'] = env.recommender_model
    params['target_densities'] = target_densities
    params['action_counts'] = action_counts
    params['goal_counts'] = goal_counts
    params['beta'] = betas
    params['date'] = datestr
    pickle.dump(params, open(os.path.join(reach_exp_dir, 'params.pkl'),"wb"))

    for action_count_val in action_counts:
        print("K parameter: {}".format(action_count_val))
        action_mask = make_nextk_action_mask(env.dense_predictions, seen_mask = env.seen_mask, count = action_count_val)
        for target_density in target_densities:
            print("\tTarget parameter: {}".format(target_density))
            target_mask = make_random_mask(env.seen_mask+action_mask, density = target_density, random_state=17)
            if goal_counts is None:
                goal_masks = [target_mask]
            else:
                goal_masks = []
                for goal_count in goal_counts:
                    if common_goals:
                        goal_masks.append(make_common_columns_mask(target_mask, nrow = env.num_user, ncol = env.num_item,
                                                                   count = goal_count, active_rows = run_users, random_state = 23))
                    else:
                        goal_masks.append(make_random_mask(negate_matrix(target_mask), count = goal_count, random_state=23))

            for i, goal_mask in enumerate(goal_masks):
                if goal_counts is None:
                    goal_count = 'all'
                else:
                    goal_count = goal_counts[i]
                print("\t\tGoal count: {}".format(goal_count))
                for beta in betas:
                    data = defaultdict()
                    data['action_mask'] = action_mask
                    data['target_mask'] = target_mask
                    data['goal_mask'] = goal_mask
                    data['beta'] = beta
                    data['action_count'] = action_count_val
                    data['target_density'] = target_density
                    data['goal_count'] = goal_count
                    exp_dir = os.path.join(reach_exp_dir, "action_{}_goal_{}_target_{}_beta_{}".format(action_count_val, goal_count, target_density, beta))
                    if not os.path.exists(exp_dir):
                        os.makedirs(exp_dir)
                    pickle.dump(data, open(os.path.join(exp_dir, 'experiment.pkl'),"wb"))

                    tic = time()
                    run_reachability_experiment(env, beta, target_mask, action_mask, goal_mask = goal_mask,
                                                save_results=True, save_dir=exp_dir, returns = False, run_users=run_users)
                    toc = time()
                    run_time = toc-tic

                    print("\t\t\tBeta: {}; Runtime: {:.2f} min".format(beta, run_time/60))

def load_dataset_experiment(dataset_name='ml-100k',
                           recommender_model='libfm',
                           title_str='', datestr='latest', exppath = EXPPATH):
    """Function to load a full experimental run:
    Experimental runs are saved in folders of the form <dataset_name>_<recommender_model>
        Each run is saved in a folder of the form <title_str>_<timestamp>
            Each run folder has a params.pkl file which contains all the parameters used in the run
            Each run folder has folder corresponding to multiple experiments
                Within each experiment folder there is  an `experiment.pkl` file which contains the action/target/goal masks
                and files of the form 'user_<USER_ID>.pkl' which contain baseline/max rho/ranks

    Parameters
    ----------
    dataset_name : str, optional
        dataset name, by default 'ml-100k'
    recommender_model : str, optional
        recommender model, by default 'libfm'
    title_str : str, optional
        identifier of the experimental run, by default ''
    datestr : str, optional
        if 'latest' then the latest file which starts with the title_str will be used,
        by default 'latest'
    exppath : str, path; optional
        by default EXPPATH (defined in config.py)

    Returns
    -------
    params: dict
        dictionary containing all the parameters of the experimental run
    exp_list: list
        list of dictionaries, where each dictionary corresponds to an experiment in the experimental run
    """
    parent_dir = os.path.join(exppath, "{}_{}".format(dataset_name, recommender_model))
    if not os.path.exists(parent_dir):
        raise IOError('No such directory exists: {}'.format(parent_dir))
    if datestr != "latest":
        folder_stem = "{}_{}".format(title_str, datestr)
        possible_folders = [fn for fn in os.listdir(parent_dir) if fn.startswith(folder_stem)]
        if len(possible_folders) == 0:
            raise IOError('No folders matching stem {} in {}'.format(folder_stem, parent_dir))
        elif len(possible_folders) == 1:
            foldername = possible_folders[0]
        else:
            options_dict = {i:option for i,option in enumerate(possible_folders)}
            res = None
            while res not in options_dict.keys():
                res = int(input('{}\nPlease select a folder by number:'.format(json.dumps(options_dict,
                                                                                    indent=4))))
            print('You chose folder {}: {}'.format(res, options_dict[res]))
            foldername = options_dict[res]
    else:
        folder_stem = "{}_".format(title_str)
        possible_folders = [fn for fn in os.listdir(parent_dir) if fn.startswith(folder_stem)]
        matched_folders = defaultdict()

        for fn in possible_folders:
            matched = re.search(r"\d\d\d\d-\d\d-\d\d-\d\d:\d\d$", fn)
            if bool(matched):
                datestr = matched.group()
                matched_folders[datestr] = fn
        date_obj_array = [datetime.strptime(k, '%Y-%m-%d-%H:%M') for k in matched_folders.keys()]
        latest_date_idx = np.argmax(date_obj_array)
        latest_datestr = list(matched_folders.keys())[latest_date_idx]
        foldername = matched_folders[latest_datestr]
    folderpath = os.path.join(parent_dir, foldername)
    params = pickle.load(open(os.path.join(folderpath, 'params.pkl'),"rb"))

    exp_list = []
    exp_folders = [fn for fn in os.listdir(folderpath) if 'params.pkl' not in fn]
    for exp_folder in exp_folders:
        print(exp_folder)
        exp_folderpath = os.path.join(folderpath, exp_folder)
        try:
            exp_dict = _load_exp(exp_folderpath)
            exp_list.append(exp_dict)
            print("Loaded experiment data from {}".format(exp_folderpath))
        except:
            print("Failed to load experiment from {}".format(exp_folderpath))

    return params, exp_list

def _load_exp(folder):
    """Load experiment data from and experiment folder
    A folder is an experimental folder if it contains
    an `experiment.pkl` file which contains the action/target/goal masks
    and files of the form 'user_<USER_ID>.pkl' which contain baseline/max rho/ranks

    Parameters
    ----------
    folder : path
        folder that contains an experimental run

    Returns
    -------
    exp_dict : dict
        dictionary with experimental data
    """
    assert os.path.exists(folder)
    exp_dict = pickle.load(open(os.path.join(folder, 'experiment.pkl'),"rb"))

    user_files = [fn for fn in os.listdir(folder) if 'user_' in fn]
    all_baseline_rhos = []
    all_max_rhos = []
    all_max_ranks = []
    all_baseline_ranks = []
    goal_row_ids = []
    goal_col_ids = []
    all_opt_actions = []
    baseline_rank_flag = True
    for f in user_files:
        user_data = pickle.load(open(os.path.join(folder, f), 'rb'))
        if 'baseline_ranks' not in user_data.keys():
            baseline_rank_flag = False
        uid = user_data['uid']
        goal_row_ids.extend([uid]*len(user_data['goal_iids'])) # add goal iids
        goal_col_ids.extend(user_data['goal_iids'])
        all_baseline_rhos.extend(user_data['baseline_rhos'])
        all_max_rhos.extend(user_data['max_rhos'])
        all_max_ranks.extend(user_data['max_ranks'])
        if baseline_rank_flag:
            all_baseline_ranks.extend(user_data['baseline_ranks'])
        all_opt_actions.extend(user_data['opt_actions'])

    exp_dict['baseline_rho_mat'] = csr_matrix((all_baseline_rhos, (goal_row_ids, goal_col_ids)))
    exp_dict['max_rho_mat'] = csr_matrix((all_max_rhos, (goal_row_ids, goal_col_ids)))
    exp_dict['max_rank_mat'] = csr_matrix((all_max_ranks, (goal_row_ids, goal_col_ids)))
    if baseline_rank_flag:
        exp_dict['baseline_rank_mat'] = csr_matrix((all_baseline_ranks, (goal_row_ids, goal_col_ids)))
    keys = zip(goal_row_ids, goal_col_ids)
    exp_dict['opt_actions'] = dict(zip(keys, all_opt_actions))
    return exp_dict

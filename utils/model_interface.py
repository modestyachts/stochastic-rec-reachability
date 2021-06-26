"""Functions for interfacing with, saving, loading RecLab recommender models."""
import numpy as np
import os
import pickle
from scipy.sparse import csr_matrix
import time

from config import MODELPATH
from helper_utils import get_seen_mask, nlargest_indices

from reclab.recommenders import LibFM, KNNRecommender
from reclab.recommenders.sparse import EASE
from reclab import data_utils

def save_trained_recommender(filename, recommender, returns_dict = False):
    """Extracts key attributes of RecLab recommender and saves in dictionary.

    Parameters
    ----------
    filename : dict
        name for recommender file to be saved at
    recommender : reclab.recommender
        recommender object to save
    returns_dict : bool
        whether or not to return the dictionary file

    Returns
    -------
    save_dict : dict
        dictionary summarizing key attributes of recommender
    """
    if not filename.endswith('.pkl'):
        filename = "{}.pkl".format(filename)
    save_dict = {}
    save_dict['trained_model'] = extract_model_parameters(recommender, use_one_way=True)
    save_dict['seen_mask'] = get_seen_mask(recommender)
    save_dict['inner_to_outer_uid'] = recommender._inner_to_outer_uid
    save_dict['inner_to_outer_iid'] = recommender._inner_to_outer_iid
    save_dict['observed_ratings'] = csr_matrix(recommender._ratings)

    pickle.dump(save_dict, open(os.path.join(MODELPATH, filename),"wb"))
    print("saving recommender model to", "{}".format(os.path.join(MODELPATH, filename)))
    if returns_dict:
        return save_dict

def extract_model_parameters(recommender):
    """Extracts key attributes of RecLab recommender and saves in dictionary.

    Parameters
    ----------
    recommender : reclab.recommender
        recommender object to extract model parameters from

    Returns
    -------
    trained_model : dict
        dictionary summarizing parameters of preference model, depending on recommender type
    """
    if recommender.name == 'libfm':
        n_user = len(recommender._users)
        n_item = len(recommender._items)
        (global_bias, biases, factors) = recommender.model_parameters()

        # select the info of users
        user_indices = np.arange(n_user)

        use_one_way = recommender.hyperparameters['use_one_way']
        if use_one_way:
            user_biases = biases[user_indices]
        else:
            user_biases = np.zeros(n_user)
        user_factors = factors[user_indices]

        # select the info of items
        item_indices = np.arange(n_user, n_item+n_user)
        if use_one_way:
            item_biases = biases[item_indices]
        else:
            item_biases = np.zeros(n_item)
        item_factors = factors[item_indices]

        trained_model = {"user_factors": user_factors,
                        "user_bias": user_biases,
                        "item_factors": item_factors,
                        "item_bias": item_biases,
                        "global_bias": global_bias,
                        "kind": 'libfm',
                        }
    elif recommender.name == 'knn':
        weights = csr_matrix(recommender._similarity_matrix.shape)
        for i in range(weights.shape[0]):
            neighbor_idx = nlargest_indices(recommender._neighborhood_size,
                                            recommender._similarity_matrix[:,i])
            weights[i, neighbor_idx] = recommender._similarity_matrix[i, neighbor_idx]
        trained_model = {"ratings": recommender._ratings,
                         "weights": weights,
                         "means": recommender._means,
                         "kind": 'knn',
                         }
    elif recommender.name == 'ease':
        trained_model = {"ratings": recommender._ratings,
                         "weights": np.array(recommender._weights),
                         "kind": 'ease'
                         }

    return trained_model

def get_predicted_scores(trained_model_dict, **kwargs):
    """Computed predicted scores from latent model

    Parameters
    ----------
    trained_model_dict : dict
        dictionary containing a trained model

    Returns
    -------
    np.array (N_selected_users x N_items)
        predicted scores for each user-item pair
    """
    kind = trained_model_dict['kind']
    assert kind in ["libfm", "knn", "ease"]
    if kind == "libfm":
        return _get_predicted_scores_libfm(trained_model_dict, **kwargs)
    if kind == "knn":
        return _get_predicted_scores_knn(trained_model_dict, **kwargs)
    if kind == "ease":
        return _get_predicted_scores_ease(trained_model_dict, **kwargs)

def _get_predicted_scores_libfm(trained_model_dict, clip = False, min_score = 1, max_score = 5, uid_vec = None):
    """Computes predicted scores for LibFM latent model

    Parameters
    ----------
    trained_model_dict : dict
        libfm latent model
    clip : bool, optional
        [description], by default False
    min_rat : int, optional
        lower bound for ratings, by default 1
    max_rat : int, optional
        upper bound for ratings, by default 5
    uid_vec : array of int, optional
        which users to compute score predictions for
    """

    if uid_vec is None:
        user_bias = trained_model_dict['user_bias']
        user_factors = trained_model_dict['user_factors']
    else:
        user_bias = trained_model_dict['user_bias'][uid_vec]
        user_factors = trained_model_dict['user_factors'][uid_vec]

    item_bias = trained_model_dict['item_bias']
    item_factors = trained_model_dict['item_factors']
    global_bias = trained_model_dict['global_bias']
    n_user = len(user_bias)
    n_item = len(item_bias)
    u_bias_mat = np.tile(user_bias.reshape(n_user,1), (1, n_item))
    i_bias_mat = np.tile(item_bias.reshape(1, n_item), (n_user, 1))
    ratings = user_factors @ item_factors.T + u_bias_mat + i_bias_mat + global_bias
    if clip:
        ratings = np.clip(ratings, min_score, max_score)
    return np.asarray(ratings)

def _get_predicted_scores_knn(trained_model_dict, clip = False, min_score = 1, max_score = 5, uid_vec = None):
    """Computes predicted scores for KNN latent model

    Parameters
    ----------
    trained_model_dict : dict
        knn model dictionary
    clip : bool, optional
        [description], by default False
    min_rat : int, optional
        lower bound for ratings, by default 1
    max_rat : int, optional
        upper bound for ratings, by default 5
    uid_vec : array of int, optional
        which users to compute score predictions for
    """
    if uid_vec is None:
        uid_vec = np.arange(trained_model_dict['ratings'].shape[0])


    offset_ratings = trained_model_dict['ratings'][uid_vec]
    item_means = trained_model_dict['means'].reshape(-1, 1)
    tiled_means = np.tile(item_means, (1, len(uid_vec))).T
    offset_ratings[offset_ratings.nonzero()] -= tiled_means[offset_ratings.nonzero()]

    weights = trained_model_dict['weights']
    normalize_by = []
    for uid in uid_vec:
        rel_idx = trained_model_dict['ratings'][uid].nonzero()[1]
        row_sum = np.array(weights[:, rel_idx].sum(axis=1))
        row_sum[np.isclose(row_sum, 0)] = 1.0
        normalize_by.append(np.array(row_sum).ravel())

    normalize_by = np.array(normalize_by)
    ratings = (offset_ratings @ weights.T) / normalize_by + tiled_means

    if clip:
        ratings = np.clip(ratings, min_score, max_score)

    return np.asarray(ratings)

def _get_predicted_scores_ease(trained_model_dict, clip = False, min_score = 1, max_score = 5, uid_vec = None):
    """Computes predicted scores for EASE latent model

    Parameters
    ----------
    trained_model_dict : dict
        ease model dictionary
    clip : bool, optional
        [description], by default False
    min_rat : int, optional
        lower bound for ratings, by default 1
    max_rat : int, optional
        upper bound for ratings, by default 5
    uid_vec : array of int, optional
        which users to compute score predictions for
    """
    if uid_vec is None:
        ratings = trained_model_dict["ratings"] @ trained_model_dict["weights"]
    else:
        ratings = trained_model_dict["ratings"][uid_vec] @ trained_model_dict["weights"]
    if clip:
        ratings = np.clip(ratings, min_score, max_score)
    # should already be np.array because weights are, but just in case
    return np.asarray(ratings)

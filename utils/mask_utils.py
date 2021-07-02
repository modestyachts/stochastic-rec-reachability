""" Functionality for creating sparse matrices"""
from collections import defaultdict
import itertools
import numpy as np
from scipy.sparse import csr_matrix

from helper_utils import get_non_zero_entries, make_sparse_matrix, negate_list

def make_random_mask(blocked_mask, density = None, count = None, random_state = None):
    """Sample target items that are disjoined from the blocked items
    Parameters
    ----------
    blocked_mask :  N_row x N_col bool sparse matrix
        True - location is blocked
    density : float, optional
        percentage of unblocked entries to be selected in each row; default None
    count : int, optional
        number of unblocked entries to be selected in each row; default None
    random_state : int, optional
        random seed for reproducibility; default None

    Returns
    -------
    random_mask :  sparse N_row x N_col bool matrix
    """
    a = density is not None
    b = count is not  None
    assert (a or b) and not (a and b), "Exactly one of density/count must be not None"

    r = np.random.RandomState(seed=random_state)
    N_row, N_col = blocked_mask.shape

    blocked_ids_dict = get_non_zero_entries(blocked_mask, how='row')
    sampled_ids_dict = {}

    for r_id in range(N_row):
        blocked_items = blocked_ids_dict[r_id]
        mask = np.ones(N_col, dtype=bool)
        mask[blocked_items] = False
        if count is not None:
            size = min(count, sum(mask))
        else:
            if density > 1:
                density = 1
            size = int(density*np.sum(mask))
        sampled_ids = r.choice(np.arange(N_col)[mask], replace = False, size = size)
        sampled_ids_dict[r_id] = sampled_ids

    random_mat = make_sparse_matrix(sampled_ids_dict, (N_row, N_col))
    return random_mat

def make_common_columns_mask(available_mask, nrow = None, ncol = None, count = None, active_rows = None, random_state = None):
    r = np.random.RandomState(seed=random_state)
    if ncol is None:
        _, ncol = available_mask.shape
    if nrow is None:
        nrow, _ = available_mask.shape
    if active_rows is None:
        active_rows = np.arange(nrow)
    available_entries = get_non_zero_entries(available_mask, how = "row")
    available_col_set_list = []
    for row_id in active_rows:
        available_col_set_list.append(set(available_entries[row_id]))
    common_available_columns = list(set.intersection(*available_col_set_list))
    sampled_columns = r.choice(common_available_columns, count, replace=False)
    row_ids = list(itertools.chain.from_iterable(itertools.repeat(i, len(sampled_columns)) for i in active_rows))
    col_ids = list(sampled_columns)*len(active_rows)
    mask = csr_matrix(([True]*len(row_ids), (row_ids, col_ids)), shape = (nrow, ncol))
    return mask


def make_nextk_action_mask(dense_scores, seen_mask = None, density = None, count = None, run_users = None):
    """ Make an action mask where for each user we select the k next items as actions

    Parameters
    ----------
    seen_mask : csr_matrix n_users x n_items
        bool matrix, if entry (i,j) is True it means that user i has seen item j
    dense_scores : np.array n_users x n_items
        matrix containing the scores for all user-item pairs
    density : float, optional
        percentage of unblocked entries to be selected in each row; default None
    count : int, optional
        number of unblocked entries to be selected in each row; default None
    run_users: iterable of int
        list of indices corresponding to users for which to create action mask

    Returns
    -------
    action_mask: csr_matrix
        sparse matrix containing the ids of the target items
    """
    a = density is not None
    b = count is not  None
    assert (a or b) and not (a and b), "Exactly one of density/count must be not None"

    if seen_mask is not None:
        assert(seen_mask.shape == dense_scores.shape), "Dense predicted score and seen mask are of different sizes"
        seen_items_dict = get_non_zero_entries(seen_mask, how='row')

    N_USERS, N_ITEMS = dense_scores.shape
    action_items_dict = defaultdict(list)
    if run_users is None:
        run_users = range(N_USERS)
    for uid in run_users:
        if seen_mask is not None:
            available_iids = negate_list(seen_items_dict[uid], N_ITEMS)
        else:
            available_iids = range(N_ITEMS)
        iids_sorted_by_prediction = np.argsort(dense_scores[uid, :])
        available_iids_sorted_by_prediction = iids_sorted_by_prediction[np.isin(iids_sorted_by_prediction, available_iids)]
        if count is not None:
            size = min(count, len(available_iids))
        else:
            if density > 1:
                density = 1
            size = int(density*len(available_iids))
        action_iids = available_iids_sorted_by_prediction[-size:]
        action_items_dict[uid] = action_iids
    action_mask = make_sparse_matrix(action_items_dict, (N_USERS, N_ITEMS))
    return action_mask

def validate_masks(target_mask, action_mask, seen_mask):
    """Validate that target items, action items and seen items are
    pairwise disjoint

    Parameters
    ----------
    target_mask : N_users x N_items bool csr matrix
        True - item is target
    action_mask : N_users x N_items bool csr matrix
        True - item is action item
    seen_mask : N_users x N_items bool csr matrix
        True - item is seen
    """
    assert(target_mask.shape == action_mask.shape == seen_mask.shape)
    # make sure target items are not action items
    assert(np.sum(target_mask*action_mask) == 0)
    # make sure action items are not seen items
    assert(np.sum(target_mask*seen_mask) == 0)
    # make sure that the list of users for which we defined target items is
    # the same as the list of users for which we defined action items
    action_dict = get_non_zero_entries(action_mask, how='row')
    target_dict = get_non_zero_entries(target_mask, how='row')
    assert(set(action_dict.keys()) == set(target_dict.keys()))

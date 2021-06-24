"""Helper functions for working with sparse matrices"""
from collections import defaultdict
import heapq
import numpy as np
from scipy.sparse import csr_matrix
import sys


def streamprinter(text):
    """Stream Printer

    Parameters
    ----------
    text : str
    """
    sys.stdout.write(text)
    sys.stdout.flush()


def softmax_transform(x, beta=1, ignore_zero = True):
    """Computes softmax probabilities of items associated with the
    their ratings

    Parameters
    ----------
    x : array
        sparse array of predicted ratings
    beta : int, optional
        parameter, by default 1
    ignore_zero : bool, optional
        if True remove 0 ratings from the predictions and
        do not consider them, by default True

    Returns
    -------
    array of softmax probabilities
    """
    x = np.array(x)
    if ignore_zero:
        mask = np.argwhere(x!=0)
        x = x[mask]
    exp = np.exp(beta * x)
    return (exp / np.sum(exp)).reshape(-1)



def get_non_zero_entries(sparse_mat, how='row'):
    """Helper function to return all indices non-zero entries of a sparse matrix
       in three convenient formats

    Parameters
    ----------
    sparse_mat : sparse matrix
    how : string
        'zip': return a list of tuples corresponding to non-zero entries
        'row': return a dict of the form {r_id:[c_ids]}, default
        'col': return a dict of the form {c_id:[r_ids]}

    Returns
    -------
    if 'zip' it returns a list of (row_id, col_id) tuples
    if 'row' it returns a dictionary of the form {row_id: [col_ids]}
    if 'col' it returns a dictionary of the form {col_id: [row_ids]}
    """
    assert how in ['zip', 'row', 'col'], "Available formats are 'zip', 'row', 'column'"
    row_ids, col_ids = sparse_mat.nonzero()
    if how == 'zip':
        return zip(row_ids, col_ids)
    non_zero_dict = defaultdict(list)
    if how == 'row':
        for r, c in zip(row_ids, col_ids):
            non_zero_dict[r].append(c)
        return non_zero_dict

    if how == 'col':
        for c, r in zip(col_ids, row_ids):
            non_zero_dict[c].append(r)
        return non_zero_dict

def entrywise_division(mat1, mat2):
    """Performs entrywise division on sparse matrices

    Parameters
    ----------
    mat1 : csr matrix
    mat2 : csr matrix

    Returns
    -------
    res: csr matrix
    """
    mat2 = mat2.power(-1)
    res = mat2.multiply(mat1)
    return res

def extract_sampled_mask(sparse_mat, sample_ids, how='row'):
    """Downsamples row/columns from a sparse matrix

    Parameters
    ----------
    sparse_mat : csr matrix
        input sparse matrix
    sample_ids : iterable of int
        list of indices to keep
    how : str, optional
        'row', subsamples the rows corresponding to the sample_ids,
        by default 'row'
        'col', subsamples the column corresponding to the sample_ids

    Returns
    -------
    sampled_mask: a sparse matrix of the same size as the input
    """
    assert how in ['row', 'col'], "The supported formats are 'row' and 'col"
    sparse_dict = get_non_zero_entries(sparse_mat, how=how)
    sampled_dict = defaultdict(list)
    for sid in sample_ids:
        sampled_dict[sid] = sparse_dict[sid]
    sampled_mask = make_sparse_matrix(sampled_dict, sparse_mat.shape, how='row')
    return sampled_mask


def make_sparse_matrix(d, shape, values = None, how='row'):
    """ Creates a sparse matrix from row/colums based
    dictionary of keys

    Parameters
    ----------
    d : dict containing the non-zero entries in the form:
        {row_id: [col_ids]} or {col_id: [row_ids]}
    shape : tuple,
            shape of the output matrix
    values : array of values, default None
        if None, then fill with ones

    how : str, optional
        'row', subsamples the rows corresponding to the sample_ids,
        by default 'row'
        'col', subsamples the column corresponding to the sample_ids

    Returns
    -------
    mask : csr matrix
    """
    assert how in ['row', 'col'], "The supported formats are 'row' and 'col"
    if how == 'row':
        row_ids = [k for k, v in d.items() for _ in range(len(v))]
        col_ids= [i for ids in d.values() for i in ids]
    else:
        col_ids = [k for k, v in d.items() for _ in range(len(v))]
        row_ids= [i for ids in d.values() for i in ids]
    if values is None:
        values = [True]*len(row_ids)
    assert len(values) == len(row_ids)
    mask = csr_matrix((values, (row_ids, col_ids)), shape = shape)
    return mask

def get_seen_mask(recommender):
    """ Creates a sparse matrix of seen entries ids

    Parameters
    ----------
    recommender : PredictRecommender
        trained recommender object

    Returns
    -------
    seen_mask : csr matrix
    """
    seen_idx = recommender._ratings.nonzero()
    seen_mask = csr_matrix((np.ones(len(seen_idx[0]), dtype=bool), seen_idx),
                           shape=(len(recommender._users), len(recommender._items)))
    return seen_mask

def negate_list(lst, N):
    """ Returns all the indices from range(N) that are not in the list

    Parameters
    ----------
    lst : list of int
        list of indices
    N : int
        size of the range

    Returns
    -------
    np.array:
        list of indices not contained in the input list
    """
    assert np.max(lst) <= N
    temp = np.ones(N).astype(bool)
    temp[lst] = False
    return np.where(temp)

def negate_matrix(mat):
    """ Returns all the negation of a boolean sparse matrix

    Parameters
    ----------
    mat : sparse matrix
    """
    one_mat = csr_matrix(np.ones(mat.shape))
    neg_mat = one_mat > mat
    return neg_mat

def union_lists(lst1, lst2):
    """ Produces the union of items in each list

    Parameters
    ----------
    lst1 : np.array
        ordered list of unique indices
    lst2 : np.array
        ordered list of unique indices

    Returns
    -------
    np.array
        ordered  list of unique indices present in either list
    """
    N = max(max(lst1), max(lst2))
    temp = np.zeros(N).astype(bool)
    temp[lst1] = True
    temp[lst2] = True
    return np.where(temp)

def intersection_lists(lst1, lst2):
    """ Produces the intersection of items in each list

    Parameters
    ----------
    lst1 : np.array
        ordered list of unique indices
    lst2 : np.array
        ordered list of unique indices

    Returns
    -------
    np.array
        ordered  list of unique indices present in both lists
    """
    N = max(max(lst1), max(lst2))
    temp1 = np.zeros(N).astype(bool)
    temp2 = np.zeros(N).astype(bool)
    temp1[lst1] = True
    temp2[lst2] = True
    temp = temp1*temp2
    return np.where(temp)

def nlargest_indices(n, iterable):
    """Given an iterable, computes the indices of the n largest items.

    Parameters
    ----------
    n : int
        How many indices to retrieve.
    iterable : iterable
        The iterable from which to compute the n largest indices.

    Returns
    -------
    largest : list of int
        The n largest indices where largest[i] is the index of the i-th largest index.

    """
    nlargest = heapq.nlargest(n, enumerate(iterable),
                              key=lambda x: x[1])
    return [i[0] for i in nlargest]
""" Setting up stochastic reachability as optimization problems and solving them"""
import cvxpy as cvx
import mosek as m
from multiprocessing import Process, Queue, Pool, cpu_count
import numpy as np
import scipy
import time

from helper_utils import negate_list, intersection_lists, streamprinter

# ==== Setup parameteres for reachability problem ======

def setup_stochastic_reachability_libfm(item_factors, item_biases, user_bias, global_bias,
                                           user_vector, target_iids, action_iids, lam=0,
                                           method='als', alpha=0.1):
    """Setup stochastic reachability problem parameters from model and action parameters
    for the LibFM recommender

    Parameters
    ----------
    item_factors : np.array
        An array of size num_items by latent_dim.
    item_biases : np.array
        An array of length num_items.
    user_bias : float
        The user's bias.
    global_bias : float
        The global bias.
    user_vector : np.array
        If method is ALS, this is the rating_vector, an array of size num_items, containing the users ratings.
        If method is SGD, this is the user_factor, an array of size latent_dim.
    target_iids :  np.array
        An array of indices coresponding to target items (i.e. those on which to evaluate reachability).
    action_iids : np.array
       An array of indices coresponding to action items (i.e. those whose ratings are actions).
    lam : float, optional
        regularization parameter used by ALS, by default 0
    method : string
        Indicates whether update is 'als' or'sgd', by default 'als'
    alpha : float
        Step size. Only used for SGD method, by de

    Returns
    -------
    Bs : np.array
        An array of size num_target_items by num_action_items, consisting of linear parameters.
    cs : np.array
        An array of length num_target_items, consisting of affine parameters.
    """

    if method == 'als':
        num_items, latent_dim = item_factors.shape()
        rating_vector = user_vector
        seen_items = negate_list(target_iids,num_items)
        Q_seen = item_factors[seen_items]
        W = np.linalg.inv( Q_seen.T @ Q_seen + lam * np.eye(Q_seen.shape[1]) )
        action_matrix = W @ item_factors[action_iids].T

        fixed_items = intersection_lists(seen_items, negate_list(action_iids, num_items))
        action_vector = W @ item_factors[fixed_items].T @ rating_vector[fixed_items]
        action_vector += W @ Q_seen.T @ ( item_biases[seen_items] + user_bias + global_bias )
    elif method == 'sgd':
        user_factor = user_vector
        latent_dim = len(user_factor)
        Q_action = item_factors[action_iids]
        action_matrix = alpha * Q_action.T
        action_vector = (np.eye(latent_dim) - alpha * Q_action.T @ Q_action) @ user_factor
        action_vector -= alpha * Q_action.T @ (item_biases[action_iids] + user_bias + global_bias)
    else:
        raise AssertionError("The method can be either 'als' or 'sgd'")


    Bs = item_factors[target_iids] @ action_matrix
    cs = item_factors[target_iids] @ action_vector + item_biases[target_iids]
    return Bs, cs

def setup_stochastic_reachability_knn(user_ratings, weights, item_means, target_iids, action_iids):
    """Setup stochastic reachability problem parameters from model and action parameters for the
    KNN recommender.

    Parameters
    ----------
    user_ratings : array of float of size num_items
        array containing predicted user ratings for all items corresponding to a user
    weights : array of floats of size num_items
        ease weights
    target_iids : array of int
        array containing the indices of the target items
    action_iids : array of int
        array containing the indices of the action items

    Returns
    -------
    Bs : np.array
        An array of size num_target_items by num_action_items, consisting of linear parameters.
    cs : np.array
        An array of length num_target_items, consisting of affine parameters.
    """
    user_ratings = user_ratings.reshape(-1, 1)
    weights_user = scipy.sparse.lil_matrix(weights.shape)
    for i in target_iids:
        for j in user_ratings.nonzero()[0]:
            weights_user[i, j] = weights[i, j]
        for j in action_iids:
            weights_user[i, j] = weights[i, j]


    row_sum = np.array(weights_user[target_iids].sum(axis=1))
    row_sum[np.isclose(row_sum, 0)] = 1.0
    weights_user[target_iids] /= row_sum

    item_means = item_means.reshape(-1, 1)

    cs = weights_user[target_iids] @ user_ratings - weights_user[target_iids] @ item_means + item_means[target_iids]
    cs = np.squeeze(np.asarray(cs))
    if len(action_iids) > 0:
        Bs = weights_user[target_iids][:,action_iids].toarray()
    else:
        Bs = np.empty(shape=(len(target_iids),0))
    return Bs, cs

def setup_stochastic_reachability_ease(user_ratings, weights, target_iids, action_iids):
    """Setup stochastic reachability problem parameters from model and action parameters for the
    EASE recommender.

    Parameters
    ----------
    user_ratings : array of float of size num_items
        array containing predicted user ratings for all items corresponding to a user
    weights : array of floats of size num_items
        ease weights
    target_iids : array of int
        array containing the indices of the target items
    action_iids : array of int
        array containing the indices of the action items

    Returns
    -------
    Bs : np.array
        An array of size num_target_items by num_action_items, consisting of linear parameters.
    cs : np.array
        An array of length num_target_items, consisting of affine parameters.
    """
    user_ratings = user_ratings.reshape(-1, 1)
    cs = np.squeeze(weights[target_iids] @ user_ratings)
    if len(action_iids) > 0:
        Bs = weights[target_iids][:,action_iids]
    else:
        Bs = np.empty(shape=(len(target_iids),0))
    return Bs, cs

def solve_maxprob_stochastic_reachability_cvxpy(relative_goal_id, Bs, cs, beta, action_range=(0,5),
                                          solver_kwargs={'solver':cvx.ECOS, 'verbose':False}):
    """

    Parameters
    ----------
    relative_goal_id : int
        relative position of the current goal item within the array of targetable items
    Bs : np.array
        An array of size num_target_items by num_action_items, consisting of linear parameters.
    cs : np.array
        An array of length num_target_items, consisting of affine parameters.
    beta : float
        Stachasticity parameter of the item selection rule (beta)
    action_range : tuple, optional
        the lower and upper limit of the action space, by default (0,5)
    solver_kwargs : dict, optional
        CVXPY solver arguments, by default {'solver':cvx.ECOS, 'verbose':False}

    Returns
    -------
    best_rho : float
        The max rho achievable
    opt_arg : np.array
        The minimizing argument
    rho_rank : int
        The ranked position of the maximized rho.
    """
    num_action_items = Bs.shape[1]
    a = cvx.Variable(num_action_items)

    Bs_not_goal = np.vstack([Bs[:relative_goal_id], Bs[(relative_goal_id+1):]])
    cs_not_goal = np.hstack([cs[:relative_goal_id], cs[(relative_goal_id+1):]])
    cons = []
    lower_bd, upper_bd = action_range
    if lower_bd > -np.inf:
        cons += [a >= lower_bd]
    if upper_bd < np.inf:
        cons += [a <= upper_bd]

    prob = cvx.Problem(cvx.Minimize(cvx.log_sum_exp( beta*(Bs_not_goal @ a + cs_not_goal) )
                                    - beta * (Bs[relative_goal_id] @ a + cs[relative_goal_id])), cons)
    try:
        prob.solve(**solver_kwargs) #, abstol=1e-6, feastol=1e-6)
        if prob.status not in [cvx.INFEASIBLE, cvx.INFEASIBLE_INACCURATE, cvx.UNBOUNDED_INACCURATE, cvx.UNBOUNDED]:
            best_rho = 1 / (np.exp(prob.value) + 1)
            opt_arg = a.value
            # rank of rho_max
            opt_scores = Bs @ opt_arg + cs
            opt_rhos = np.exp(opt_scores) / np.sum(np.exp(opt_scores))
            temp = opt_rhos.argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(opt_rhos))[::-1]
            rho_rank = ranks[relative_goal_id]
        else:
            print(prob.status)
            best_rho = 0.
            opt_arg = None
            rho_rank = -1
    except cvx.error.SolverError as e:
        print("SolverError", e)
        best_rho = 0.
        opt_arg = None
        rho_rank = -1
    return (best_rho, opt_arg, rho_rank)

def get_updated_user_factor(user_action, user_action_items, old_user_factor, item_factors, alpha=0.1):
    """ Computes the updated user factor for SGD-MF

    Parameters
    ----------
    user_action : np.array
        array, where i-th entry is the rating corresponding to the i-th non-zero entry in the action_mask
    user_action_items : Boolean np.array (num_items x 1)
        True - item is an action item for this user
    old_user_factor : np.array (latent_dim x 1)
        Old latent factor corresponding to the user
    item_factors : np.array
        An array of size num_items by latent_dim.
    alpha : float
        Step size. Only used for SGD method.

    Returns
    -------
    updated_user_factor: np.array (latent_dim x 1)
    """
    Q_action = item_factors[user_action_items,:]
    try:
        updated_user_factor = old_user_factor - alpha * Q_action.T @ user_action - alpha * Q_action.T @ Q_action @ old_user_factor
    except:
        updated_user_factor = old_user_factor
    return (updated_user_factor)

# ======== Parallelize optimization with MOSEK =========

def _get_mosek_task(env, B, b, c, action_range, verbose=False):
    """ Sets up a mosek optimization for maxprob reachability.
    Equivalent optimizations:
        rho = max_a exp(beta*(Bs[goal_id] @ a + cs[goal_id])) / (sum exp(beta*(Bs @ a + cs)))
    and
        - log(rho) + beta * c[goal_id] = min_a LSE( B @ a + c) ) - b @ a
    and
        gamma = min_(a, t, u)  t - <b, a>
                s.t.          <1, u> <= 1
                              (u_i, 1, B_i @ a + c_i - t) in K_exp for all i
    for rho = 1 / exp(gamma - beta * c[goal_id]).
    Enforcing cone constraint requires auxiliary variables v = 1 and z_i = B_i @ a + c_i - t.
    Therefore, the decision variable is (a, t, u, v, z).

    Parameters
    ----------
    env : mosek environment
        a mosek environment for the task
    B : np.array (num_items x num_actions)
        equal to beta * Bs
    b : np.array (num_actions)
        equal to beta * Bs[goal_id]
    c : np.array (num_items)
        equal to beta * cs
    action_range : tuple
        upper and lower bounds on action
    verbose : bool
        flag for verbosity of optimization procedure

    Returns
    -------
    task : mosek task
    """
    num_items, num_actions = B.shape

    task = m.Task(env, 0, 0)
    task.putintparam(m.iparam.num_threads,1)

    if verbose:
        task.set_Stream(m.streamtype.log, streamprinter)

    # Add number variables and linear constraints
    task.appendvars(num_actions + 1 + 3*num_items)
    task.appendcons(1 + num_items)

    # Objective is linear in action items and auxiliary variable
    task.putobjsense(m.objsense.minimize)
    task.putcslice(0, num_actions+1, np.hstack([-b, 1]))

    # Specifying domain of each variable
    inf = 0.0 # defined for symbolic purposes only
    ## bounded actions
    lower_bd, upper_bd = action_range
    if lower_bd > -np.inf:
        if upper_bd < np.inf:
            task.putvarboundsliceconst(0, num_actions, m.boundkey.ra, *action_range)
        else:
            task.putvarboundsliceconst(0, num_actions, m.boundkey.lo, lower_bd, inf)
    else:
        if upper_bd < np.inf:
            task.putvarboundsliceconst(0, num_actions, m.boundkey.up, -inf, upper_bd)
        else:
            task.putvarboundsliceconst(0, num_actions, m.boundkey.fr, -inf, inf)
    ## unbounded auxiliary variable t
    task.putvarbound(num_actions, m.boundkey.fr, -inf, inf)
    ## unbounded LSE variable u
    task.putvarboundsliceconst(num_actions+1, num_actions+1+num_items, m.boundkey.fr, -inf, inf)
    ## auxiliary variable v equal to 1
    task.putvarboundsliceconst(num_actions+1+num_items, num_actions+1+2*num_items, m.boundkey.fx, 1., 1.)
    ## unbounded auxiliary variable z
    task.putvarboundsliceconst(num_actions+1+2*num_items, num_actions+1+3*num_items, m.boundkey.fr, -inf, inf)

    # Adding linear constraints
    ## LSE variable u adds less than 1
    task.putarow(0, np.arange(num_actions+1, num_actions+1+num_items), np.ones(num_items))
    task.putconbound(0, m.boundkey.up, -inf, 1.)
    ## auxiliary variable z equality definition
    for i in range(num_actions):
        task.putacol(i, np.arange(1,num_items+1), -B[:,i])
    task.putacol(num_actions, np.arange(1,num_items+1), np.ones(num_items))
    task.putaijlist(np.arange(1,num_items+1),
                    np.arange(num_actions+1+2*num_items,num_actions+1+3*num_items),
                    np.ones(num_items))
    task.putconboundslice(1, num_items+1, [m.boundkey.fx]*num_items, c, c)

    # Adding cone constraints on u, v, z
    for i in range(num_items):
        task.appendcone(m.conetype.pexp, 0., [num_actions+1+i, num_actions+1+i+num_items, num_actions+1+i+2*num_items])

    return task


def solve_maxprob_stochastic_reachability(relative_goal_id, Bs, cs, beta, action_range=(0,5), verbose=False, env=None, queue=None):
    """Maximize  stochastic reachability probability.

    Parameters
    ----------
    relative_goal_id : int
        the goal items for reachability problem.
        Note this is the position of the goal item among the other target items
    Bs : np.array
        An array of size num_target_items by num_action_items, consisting of linear score update parameters.
    cs : np.array
        An array of length num_target_items, consisting of affine score update parameters.
    beta : float
        The sharpness parameter of the stochastic reachability problem.
    action_range : tuple
        Lower and upper bounds on possible rating values, used to constrain the action space.
    verbose : bool
        verbosity flag for optimization solver
    env : mosek environment, optimal
        mosek optimization environment, used for running in parallel
    queue : optional
        A queue for results, used for running in parallel

    Returns
    -------
    best_rho : float
        The max rho achievable
    opt_arg : np.array
        The minimizing argument
    rho_rank : int
        The ranked position of the maximized rho.
    """
    _, num_actions = Bs.shape
    if env is None:
        env = m.Env()
    task = _get_mosek_task(env, beta * Bs, beta * Bs[relative_goal_id], beta * cs, action_range, verbose=verbose)
    try:
        task.optimize()
        solsta = task.getsolsta(m.soltype.itr)
        if solsta == m.solsta.optimal:
            # optimal solution
            best_gamma = task.getdouinf(m.dinfitem.intpnt_primal_obj)
            best_rho = 1 / np.exp(best_gamma - beta*cs[relative_goal_id])

            # optimizing argument
            opt_arg = [0.0] * num_actions
            task.getxxslice(m.soltype.itr, 0, num_actions, opt_arg)

            # rank of rho_max
            opt_scores = Bs @ opt_arg + cs
            opt_rhos = np.exp(opt_scores) / np.sum(np.exp(opt_scores))
            temp = opt_rhos.argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(opt_rhos))[::-1]
            rho_rank = ranks[relative_goal_id]
        else:
            print('not optimal', solsta)
            best_rho = 0.
            opt_arg = None
            rho_rank = -1
    except m.MosekException as e:
        print(e)
        best_rho = 0.
        opt_arg = None
        rho_rank = -1
    if queue is not None:
        queue.put((relative_goal_id, best_rho, opt_arg, rho_rank))
    return best_rho, opt_arg, rho_rank


def _wrapper_maxprob_stochastic_reachability(args):
    """Wrapper method used to parallelize reeachability

    Parameters
    ----------
    args : arguments for solve_maxprob_stochastic_reachability

    Returns
    -------
    args[0]: int
        relative goal item id
    best_rho : float
        The max rho achievable
    opt_arg : np.array
        The minimizing argument
    rho_rank : int
        The ranked position of the maximized rho.
    """
    best_rho, opt_arg, rho_rank = solve_maxprob_stochastic_reachability(*args)
    return args[0], best_rho, opt_arg, rho_rank

def parsolve_maxprob_stochastic_reachability(relative_goal_ids, Bs, cs, beta, action_range=(0,5), verbose=False):
    """Parralelized stochastic reachability optimization
    Parralelization is performed over the goal items

    Parameters
    ----------
    relative_goal_ids : np.array
        list of integers corresponding to the relative ids of the goal items within the targetable items
    Bs : np.array
        An array of size num_target_items by num_action_items, consisting of linear score update parameters.
    cs : np.array
        An array of length num_target_items, consisting of affine score update parameters.
    beta : float
        The sharpness parameter of the stochastic reachability problem.
    action_range : tuple
        Lower and upper bounds on possible rating values, used to constrain the action space.
    verbose : bool
        verbosity flag for optimization solver

    Returns
    -------
    best_rho : float
        The max rho achievable
    opt_arg : np.array
        The minimizing argument
    rho_rank : int
        The ranked position of the maximized rho.
    """
    best_rhos = {}
    opt_args = {}
    rho_ranks = {}

    queue = None
    env = None

    # with m.Env() as env:
    p = Pool(cpu_count())
    rets = p.map(_wrapper_maxprob_stochastic_reachability, [(relative_goal_id, Bs, cs, beta, action_range, verbose, env, queue) for relative_goal_id in relative_goal_ids])
    p.close()
    p.join()

    for ret in rets:
        relative_goal_id, best_rho, opt_arg, rho_rank = ret
        best_rhos[relative_goal_id] = best_rho
        opt_args[relative_goal_id] = opt_arg
        rho_ranks[relative_goal_id] = rho_rank
    return best_rhos, opt_args, rho_ranks

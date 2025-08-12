import numpy as np
from ot.utils import unif, dist, list_to_array
from ot.backend import get_backend
import warnings


def geometricBar(weights, alldistribT):
    """return the weighted geometric mean of distributions"""
    weights, alldistribT = list_to_array(weights, alldistribT)
    nx = get_backend(weights, alldistribT)
    assert (len(weights) == alldistribT.shape[1])
    return nx.exp(nx.dot(nx.log(alldistribT), weights.T))


def geometricMean(alldistribT):
    """return the  geometric mean of distributions"""
    alldistribT = list_to_array(alldistribT)
    nx = get_backend(alldistribT)
    return nx.exp(nx.mean(nx.log(alldistribT), axis=1))


def barycenter_sinkhorn(A, M, reg, weights=None, numItermax=1000,
                        stopThr=1e-4, verbose=False, log=False, warn=True):
    r"""Compute the entropic regularized wasserstein barycenter of distributions :math:`\mathbf{A}`

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein distance
      (see :py:func:`ot.bregman.sinkhorn`)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix
      :math:`\mathbf{A}`
    - `reg` and :math:`\mathbf{M}` are respectively the regularization term and
      the cost matrix for OT

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm as proposed in :ref:`[3]<references-barycenter-sinkhorn>`.

    Parameters
    ----------
    A : array-like, shape (dim, n_hists)
        `n_hists` training distributions :math:`\mathbf{a}_i` of size `dim`
    M : array-like, shape (dim, dim)
        loss matrix for OT
    reg : float
        Regularization term > 0
    weights : array-like, shape (n_hists,)
        Weights of each histogram :math:`\mathbf{a}_i` on the simplex (barycentric coodinates)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.


    Returns
    -------
    a : (dim,) array-like
        Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-barycenter-sinkhorn:
    References
    ----------

    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G. (2015).
    Iterative Bregman projections for regularized transportation problems.
    SIAM Journal on Scientific Computing, 37(2), A1111-A1138.

    """

    A, M = list_to_array(A, M)

    nx = get_backend(A, M)

    if weights is None:
        weights = nx.ones((A.shape[1],), type_as=A) / A.shape[1]
    else:
        assert (len(weights) == A.shape[1])

    if log:
        log = {'err': []}

    K = nx.exp(-M / reg)

    err = 1

    UKv = nx.dot(K, (A.T / nx.sum(K, axis=0)).T)

    u = (geometricMean(UKv) / UKv.T).T

    for ii in range(numItermax):
        UKv_prev = UKv
        u_prev = u
        UKv = u * nx.dot(K.T, A / nx.dot(K, u))
        u = (u.T * geometricBar(weights, UKv)).T / UKv

        if (nx.any(UKv == 0)
                or nx.any(nx.isnan(u))
                or nx.any(nx.isinf(u))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Warning: numerical errors at iteration %d' % ii)
            u = u_prev
            UKv = UKv_prev
            break

        if ii % 10 == 1:
            err = nx.sum(nx.std(UKv, axis=1))

            # log and verbose print
            if log:
                log['err'].append(err)

            if err < stopThr:
                break
            if verbose:
                if ii % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")
    if log:
        log['niter'] = ii
        return geometricBar(weights, UKv), log
    else:
        return geometricBar(weights, UKv)


def sinkhorn_knopp(a, b, M, reg, numItermax=1000, stopThr=1e-9,
                   verbose=False, log=False, warn=True, warmstart=None, **kwargs):
    r"""
    Solve the entropic regularization optimal transport problem and return the OT matrix

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0
    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      weights (histograms, both sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp
    matrix scaling algorithm as proposed in :ref:`[2] <references-sinkhorn-knopp>`


    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,) or array-like, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed :math:`\mathbf{M}` if :math:`\mathbf{b}` is a matrix
        (return OT loss + dual variables in log)
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the u,v sinkhorn scaling vectors)

    Returns
    -------
    gamma : array-like, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> ot.sinkhorn(a, b, M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])


    .. _references-sinkhorn-knopp:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation
        of Optimal Transport, Advances in Neural Information
        Processing Systems (NIPS) 26, 2013


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)

    if len(a) == 0:
        a = nx.full((M.shape[0],), 1.0 / M.shape[0], type_as=M)
    if len(b) == 0:
        b = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        if n_hists:
            u = nx.ones((dim_a, n_hists), type_as=M) / dim_a
            v = nx.ones((dim_b, n_hists), type_as=M) / dim_b
        else:
            u = nx.ones(dim_a, type_as=M) / dim_a
            v = nx.ones(dim_b, type_as=M) / dim_b
    else:
        u, v = nx.exp(warmstart[0]), nx.exp(warmstart[1])

    K = nx.exp(M / (-reg))

    Kp = (1 / a).reshape(-1, 1) * K

    err = 1
    for ii in range(numItermax):
        uprev = u
        vprev = v
        KtransposeU = nx.dot(K.T, u)
        v = b / KtransposeU
        u = 1. / nx.dot(Kp, v)

        if (nx.any(KtransposeU == 0)
                or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
                or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Warning: numerical errors at iteration %d' % ii)
            u = uprev
            v = vprev
            break
        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                tmp2 = nx.einsum('ik,ij,jk->jk', u, K, v)
            else:
                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = nx.einsum('i,ij,j->j', u, K, v)
            err = nx.norm(tmp2 - b)  # violation of marginal
            if log:
                log['err'].append(err)

            if err < stopThr:
                break
            if verbose:
                if ii % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")
    if log:
        log['niter'] = ii
        log['u'] = u
        log['v'] = v

    if n_hists:  # return only loss
        res = nx.einsum('ik,ij,jk,ij->k', u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))


def sinkhorn_knopp_new(a, b, M, reg, numItermax=1000, stopThr=1e-9,
                       verbose=False, log=False, warn=True, warmstart=None, **kwargs):
    r"""
    Solve the entropic regularization optimal transport problem and return the OT matrix

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0
    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      weights (histograms, both sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp
    matrix scaling algorithm as proposed in :ref:`[2] <references-sinkhorn-knopp>`


    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,) or array-like, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed :math:`\mathbf{M}` if :math:`\mathbf{b}` is a matrix
        (return OT loss + dual variables in log)
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the u,v sinkhorn scaling vectors)

    Returns
    -------
    gamma : array-like, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> ot.sinkhorn(a, b, M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])


    .. _references-sinkhorn-knopp:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation
        of Optimal Transport, Advances in Neural Information
        Processing Systems (NIPS) 26, 2013


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)

    if len(a) == 0:
        a = nx.full((M.shape[0],), 1.0 / M.shape[0], type_as=M)
    if len(b) == 0:
        b = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        if n_hists:
            u = nx.ones((dim_a, n_hists), type_as=M) / dim_a
            v = nx.ones((dim_b, n_hists), type_as=M) / dim_b
        else:
            u = nx.ones(dim_a, type_as=M) / dim_a
            v = nx.ones(dim_b, type_as=M) / dim_b
    else:
        u, v = nx.exp(warmstart[0]), nx.exp(warmstart[1])

    K = nx.exp(M / (-reg))

    Kp = (1 / a).reshape(-1, 1) * K

    err = 1
    for ii in range(numItermax):
        uprev = u
        vprev = v
        KtransposeU = nx.dot(K.T, u)
        v = b / KtransposeU
        u = 1. / nx.dot(Kp, v)

        if (nx.any(KtransposeU == 0)
                or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
                or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Warning: numerical errors at iteration %d' % ii)
            u = uprev
            v = vprev
            break
        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                tmp2 = nx.einsum('ik,ij,jk->jk', u, K, v)
            else:
                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = nx.einsum('i,ij,j->j', u, K, v)
            err = nx.norm(tmp2 - b)  # violation of marginal
            if log:
                log['err'].append(err)

            if err < stopThr:
                break
            if verbose:
                if ii % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")
    if log:
        log['niter'] = ii
        log['u'] = u
        log['v'] = v

    if n_hists:  # return only loss
        res = nx.einsum('ik,ij,jk,ij->k', u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))


def sinkhorn_gamma(a, bd, bu, M, reg, numItermax=1000, stopThr=1e-9,
                   verbose=False, log=False, warn=True, warmstart=None, **kwargs):
    r"""
    Solve the entropic regularization optimal transport problem and return the OT matrix

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0
    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      weights (histograms, both sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp
    matrix scaling algorithm as proposed in :ref:`[2] <references-sinkhorn-knopp>`


    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,) or array-like, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed :math:`\mathbf{M}` if :math:`\mathbf{b}` is a matrix
        (return OT loss + dual variables in log)
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the u,v sinkhorn scaling vectors)

    Returns
    -------
    gamma : array-like, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    """

    a, bu, bd, M = list_to_array(a, bu, bd, M)

    nx = get_backend(M, a, bu, bd)

    if len(a) == 0:
        a = nx.full((M.shape[0],), 1.0 / M.shape[0], type_as=M)
    if len(bu) == 0:
        bu = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)
    if len(bd) == 0:
        bd = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)

    # init data
    dim_a = len(a)
    dim_b = bu.shape[0]

    if len(bu.shape) > 1:
        n_hists = bu.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'change': []}

    gamma = nx.exp(M / (-reg))
    # print(gamma)
    err = 1
    for ii in range(numItermax):
        gamma_prev = gamma
        gamma = projR(gamma, a)
        gamma = projC_max(gamma, bd)
        gamma = projC_min(gamma, bu)

        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations

            change = nx.norm(gamma - gamma_prev)  # violation of marginal
            if log:
                log['change'].append(change)

            if change < stopThr:
                break
            if verbose:
                if ii % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, change))
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")

    return gamma


def sinkhorn_gammanew(a, bd, bu, M, reg, numItermax=1000, stopThr=1e-9,
                   verbose=False, log=False, warn=True, warmstart=None, **kwargs):
    a, bu, bd, M = list_to_array(a, bu, bd, M)

    nx = get_backend(M, a, bu, bd)

    if len(a) == 0:
        a = nx.full((M.shape[0],), 1.0 / M.shape[0], type_as=M)
    if len(bu) == 0:
        bu = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)
    if len(bd) == 0:
        bd = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)

    # init data
    dim_a = len(a)
    dim_b = bu.shape[0]

    if len(bu.shape) > 1:
        n_hists = bu.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'change': []}

    gamma = nx.exp(M / (-reg))
    gamma = np.array([gamma] * a.shape[1])
    # print(gamma)
    err = 1
    for ii in range(numItermax):
        gamma_prev = gamma
        gamma = projRnew(gamma, a)
        gamma = projC_maxnew(gamma, bd)
        gamma = projC_minnew(gamma, bu)

        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations

            change = nx.norm(gamma - gamma_prev)  # violation of marginal
            if log:
                log['change'].append(change)

            if change < stopThr:
                break
            if verbose:
                if ii % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, change))
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")

    return gamma


def sinkhorn_gamma_break(a, bd, bu, M, reg, numItermax=1000, stopThr=1e-9,
                         verbose=False, log=False, warn=True, warmstart=None, **kwargs):
    r"""
    Solve the entropic regularization optimal transport problem and return the OT matrix

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0
    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      weights (histograms, both sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp
    matrix scaling algorithm as proposed in :ref:`[2] <references-sinkhorn-knopp>`


    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,) or array-like, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed :math:`\mathbf{M}` if :math:`\mathbf{b}` is a matrix
        (return OT loss + dual variables in log)
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the u,v sinkhorn scaling vectors)

    Returns
    -------
    gamma : array-like, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    """

    a, bu, bd, M = list_to_array(a, bu, bd, M)

    nx = get_backend(M, a, bu, bd)

    if len(a) == 0:
        a = nx.full((M.shape[0],), 1.0 / M.shape[0], type_as=M)
    if len(bu) == 0:
        bu = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)
    if len(bd) == 0:
        bd = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)

    # init data
    dim_a = len(a)
    dim_b = bu.shape[0]

    if len(bu.shape) > 1:
        n_hists = bu.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'change': []}

    gamma = nx.exp(M / (-reg))
    # print(gamma)
    err = 1
    for ii in range(numItermax):
        gamma_prev = gamma
        gamma = projR(gamma, a)
        # gamma = projC_max(gamma, bd)
        # gamma = projC_min(gamma, bu)

        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations

            change = nx.norm(gamma - gamma_prev)  # violation of marginal
            if log:
                log['change'].append(change)

            if change < stopThr:
                break
            if verbose:
                if ii % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, change))
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")

    return gamma


def sinkhorn_gamma_breaknew(a, bd, bu, M, reg, numItermax=1000, stopThr=1e-9,
                         verbose=False, log=False, warn=True, warmstart=None, **kwargs):
    r"""
    Solve the entropic regularization optimal transport problem and return the OT matrix

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0
    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      weights (histograms, both sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp
    matrix scaling algorithm as proposed in :ref:`[2] <references-sinkhorn-knopp>`


    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,) or array-like, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed :math:`\mathbf{M}` if :math:`\mathbf{b}` is a matrix
        (return OT loss + dual variables in log)
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the u,v sinkhorn scaling vectors)

    Returns
    -------
    gamma : array-like, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    """

    a, bu, bd, M = list_to_array(a, bu, bd, M)

    nx = get_backend(M, a, bu, bd)

    if len(a) == 0:
        a = nx.full((M.shape[0],), 1.0 / M.shape[0], type_as=M)
    if len(bu) == 0:
        bu = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)
    if len(bd) == 0:
        bd = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)

    # init data
    dim_a = len(a)
    dim_b = bu.shape[0]

    if len(bu.shape) > 1:
        n_hists = bu.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'change': []}

    gamma = nx.exp(M / (-reg))
    gamma = np.array([gamma] * a.shape[1])
    # print(gamma)
    err = 1
    for ii in range(numItermax):
        gamma_prev = gamma
        gamma = projRnew(gamma, a)
        # gamma = projC_max(gamma, bd)
        # gamma = projC_min(gamma, bu)

        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations

            change = nx.norm(gamma - gamma_prev)  # violation of marginal
            if log:
                log['change'].append(change)

            if change < stopThr:
                break
            if verbose:
                if ii % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, change))
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")

    return gamma


def projR(gamma, p):
    """return the KL projection on the row constrints """
    gamma, p = list_to_array(gamma, p)
    nx = get_backend(gamma, p)
    return (gamma.T * p / nx.maximum(nx.sum(gamma, axis=1), 1e-10)).T


def projRnew(gamma, p):
    """return the KL projection on the row constrints """
    gamma, p = list_to_array(gamma, p)
    print(gamma.shape)
    nx = get_backend(gamma, p)
    g = nx.maximum(nx.einsum('kij->ik', gamma), 1e-10)
    print(g.shape)
    g = p / g
    g = nx.einsum('kij,ik->kij', gamma, g)
    return g


def projC(gamma, q):
    """return the KL projection on the column constrints """
    gamma, q = list_to_array(gamma, q)
    nx = get_backend(gamma, q)
    return gamma * q / nx.maximum(nx.sum(gamma, axis=0), 1e-10)


def projC_max(gamma, q):
    """return the KL projection on the column constrints """
    gamma, q = list_to_array(gamma, q)
    nx = get_backend(gamma, q)
    return gamma * nx.maximum(q / nx.maximum(nx.sum(gamma, axis=0), 1e-10), 1)


def projC_maxnew(gamma, q):
    """return the KL projection on the column constrints """
    gamma, q = list_to_array(gamma, q)
    nx = get_backend(gamma, q)
    g = nx.maximum(nx.einsum('kij->jk', gamma), 1e-10)
    g = nx.maximum(q / g, 1)
    g = nx.einsum('kij,jk->kij', gamma, g)
    return g


def projC_min(gamma, q):
    """return the KL projection on the column constrints """
    gamma, q = list_to_array(gamma, q)
    nx = get_backend(gamma, q)
    return gamma * nx.minimum(q / nx.maximum(nx.sum(gamma, axis=0), 1e-10), 1)


def projC_minnew(gamma, q):
    """return the KL projection on the column constrints """
    gamma, q = list_to_array(gamma, q)
    nx = get_backend(gamma, q)
    g = nx.maximum(nx.einsum('kij->jk', gamma), 1e-10)
    g = nx.minimum(q / g, 1)
    g = nx.einsum('kij,jk->kij', gamma, g)
    return g


def compute_optimal_transport(M, r, c, lam, eplison=1e-8):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm

    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter

    Outputs:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    n, m = M.shape  # 8, 5
    P = np.exp(-M / lam)  # (8, 5)
    P /= P.sum()  # 归一化
    u = np.zeros(n)  # (8, )
    # normalize this matrix
    while np.max(np.abs(u - P.sum(1))) > eplison:  # 这里是用行和判断收敛
        # 对行和列进行缩放，使用到了numpy的广播机制，不了解广播机制的同学可以去百度一下
        u = P.sum(1)  # 行和 (8, )
        P *= (r / u).reshape((-1, 1))  # 缩放行元素，使行和逼近r
        # print("P_after_u")
        # print(P)
        v = P.sum(0)  # 列和 (5, )
        P *= (c / v).reshape((1, -1))  # 缩放列元素，使列和逼近c
        # print("P_after_v")
        # print(P)

    return P, np.sum(P * M)  # 返回分配矩阵和Sinkhorn距离


def sinkhorn_uv(a, bd, bu, M, reg, numItermax=1000, stopThr=1e-9,
                verbose=False, log=False, warn=True, warmstart=None, **kwargs):

    a, bu, bd, M = list_to_array(a, bu, bd, M)

    nx = get_backend(M, a, bu, bd)

    if len(a) == 0:
        a = nx.full((M.shape[0],), 1.0 / M.shape[0], type_as=M)
    if len(bu) == 0:
        bu = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)
    if len(bd) == 0:
        bd = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)

    # init data
    dim_a = len(a)
    dim_b = bu.shape[0]

    if len(bu.shape) > 1:
        n_hists = bu.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        if n_hists:
            u = nx.ones((dim_a, n_hists), type_as=M) / dim_a
            vd = nx.ones((dim_b, n_hists), type_as=M) / dim_b / 2
            vu = nx.ones((dim_b, n_hists), type_as=M) / dim_b * 2
        else:
            u = nx.ones(dim_a, type_as=M) / dim_a
            vd = nx.ones(dim_b, type_as=M) / dim_b / 2
            vu = nx.ones(dim_b, type_as=M) / dim_b * 2
    else:
        u, vd, vu = nx.exp(warmstart[0]), nx.exp(warmstart[1]), nx.exp(warmstart[2])

    K = nx.exp(M / (-reg))

    err = 1
    tmp_new = nx.zeros((dim_a, dim_b))
    for ii in range(numItermax):
        uprev = u
        vdprev = vd
        vuprev = vu
        # KtransposeU = nx.dot(K.T, u)
        u = a / nx.dot(K, vu * vd)
        vd = nx.maximum(np.ones(dim_b), (bd / (nx.dot(u, K) * vu)))
        vu = nx.minimum(np.ones(dim_b), (bu / (nx.dot(u, K) * vd)))

        if (nx.any(nx.isnan(u)) or nx.any(nx.isnan(vd)) or nx.any(nx.isnan(vu)) or
                nx.any(nx.isinf(u)) or nx.any(nx.isinf(vd)) or nx.any(nx.isinf(vu))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Warning: numerical errors at iteration %d' % ii)
            u = uprev
            vd = vdprev
            vu = vuprev
            break
        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            tmp = tmp_new
            tmp_new = nx.dot(nx.dot(nx.diag(u), K), nx.diag(vu * vd))
            err = nx.norm(tmp_new - tmp)  # violation of marginal
            if log:
                log['err'].append(err)

            if err < stopThr:
                break
            if verbose:
                if ii % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")
    if log:
        log['niter'] = ii
        log['u'] = u
        log['vu'] = vu
        log['vd'] = vd

    if log:
        return tmp_new, log
    else:
        return tmp_new


def sinkhorn_log(a, bd, bu, M, reg, numItermax=1000, stopThr=1e-9, verbose=False,
                 log=False, warn=True, warmstart=None, **kwargs):
    r"""
    Solve the entropic regularization optimal transport problem in log space
    and return the OT matrix

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0
    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (histograms, both sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm  :ref:`[2] <references-sinkhorn-log>` with the
    implementation from :ref:`[34] <references-sinkhorn-log>`


    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,) or array-like, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed :math:`\mathbf{M}` if :math:`\mathbf{b}` is a matrix (return OT loss + dual variables in log)
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the u,v sinkhorn scaling vectors)

    Returns
    -------
    gamma : array-like, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> ot.sinkhorn(a, b, M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])


    .. _references-sinkhorn-log:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of
        Optimal Transport, Advances in Neural Information Processing
        Systems (NIPS) 26, 2013

    .. [34] Feydy, J., Séjourné, T., Vialard, F. X., Amari, S. I.,
        Trouvé, A., & Peyré, G. (2019, April). Interpolating between
        optimal transport and MMD using Sinkhorn divergences. In The
        22nd International Conference on Artificial Intelligence and
        Statistics (pp. 2681-2690). PMLR.


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    a, bu, bd, M = list_to_array(a, bu, bd, M)

    nx = get_backend(M, a, bu, bd)

    if len(a) == 0:
        a = nx.full((M.shape[0],), 1.0 / M.shape[0], type_as=M)
    if len(bu) == 0:
        bu = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)
    if len(bd) == 0:
        bd = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)

    # init data
    dim_a = len(a)
    dim_b = bu.shape[0]

    if len(bu.shape) > 1:
        n_hists = bu.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    K = nx.exp(M / (-reg))

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        f = nx.ones(dim_a, type_as=M) / dim_a
        g = -nx.ones(dim_b, type_as=M) / dim_b
        h = nx.ones(dim_b, type_as=M) / dim_b
    else:
        f, g, h = warmstart

    loga = nx.log(a)
    logbd = nx.log(bd)
    logbu = nx.log(bu)

    err = 1
    P_new = nx.zeros((dim_a, dim_b), type_as=M)

    def Loss(x, y, z):
        return nx.sum(x * a) + nx.sum(y * bd) + nx.sum(z * bu) \
               - reg * nx.sum(nx.exp(x / reg) * nx.dot(M, nx.exp((y + z) / reg)))

    def P(x, y, z):
        return nx.exp(-x / reg).reshape(-1, 1) * nx.exp(-M / reg) * nx.exp(-(y + z) / reg)

    for ii in range(numItermax):
        fprev = f
        gprev = g
        hprev = h

        f = reg * loga - reg * nx.log(nx.dot(K, nx.exp((g + h) / reg)))
        g = nx.maximum(reg * logbd - reg * nx.log(nx.dot(K.T, nx.exp(f / reg))) - h, 0)
        h = nx.minimum(reg * logbu - reg * nx.log(nx.dot(K.T, nx.exp(f / reg))) - g, 0)

        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations

            # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
            P = P_new
            P_new = nx.exp(f / reg).reshape(-1, 1) * nx.exp(-M / reg) * nx.exp((g + h) / reg).reshape(1, -1)
            err = nx.norm(P_new - P)  # violation of marginal
            if log:
                log['err'].append(err)

            if verbose:
                if ii % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))
            if err < stopThr:
                break
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")

    if log:
        log['niter'] = ii
        log['f'] = f
        log['g'] = g
        log['h'] = h
        log['L'] = nx.sum(f * a) + nx.sum(g * bd) + nx.sum(h * bu) \
                   - reg * nx.sum(nx.exp(f / reg) * nx.dot(M, nx.exp((g + h) / reg)))

        return P_new, log

    else:
        return P_new






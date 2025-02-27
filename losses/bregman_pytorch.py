# -*- coding: utf-8 -*-
"""
Rewrite ot.bregman.sinkhorn in Python Optimal Transport (https://pythonot.github.io/_modules/ot/bregman.html#sinkhorn)
using pytorch operations.
Bregman projections for regularized OT (Sinkhorn distance).
"""

import torch
import time

M_EPS = 1e-16


def sinkhorn(
    a,
    b,
    C,
    reg=1e-1,
    method="sinkhorn",
    maxIter=1000,
    tau=1e3,
    stopThr=1e-9,
    verbose=False,
    log=True,
    warm_start=None,
    eval_freq=10,
    print_freq=200,
    **kwargs
):
    """
    Solve the entropic regularization optimal transport
    The input should be PyTorch tensors
    The function solves the following optimization problem:

    .. math::
            \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
            s.t. \gamma 1 = a
                     \gamma^T 1= b
                     \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [1].

    Parameters
    ----------
    a : torch.tensor (na,)
            samples measure in the target domain
    b : torch.tensor (nb,)
            samples in the source domain
    C : torch.tensor (na,nb)
            loss matrix
    reg : float
            Regularization term > 0
    method : str
            method used for the solver either 'sinkhorn', 'greenkhorn', 'sinkhorn_stabilized' or
            'sinkhorn_epsilon_scaling', see those function for specific parameters
    maxIter : int, optional
            Max number of iterations
    stopThr : float, optional
            Stop threshol on error ( > 0 )
    verbose : bool, optional
            Print information along iterations
    log : bool, optional
            record log if True

    Returns
    -------
    gamma : (na x nb) torch.tensor
            Optimal transportation matrix for the given parameters
    log : dict
            log dictionary return only if log==True in parameters

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    See Also
    --------

    """

    if method.lower() == "sinkhorn":
        return sinkhorn_knopp(
            a,
            b,
            C,
            reg,
            maxIter=maxIter,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            warm_start=warm_start,
            eval_freq=eval_freq,
            **kwargs
        )
    else:
        raise ValueError("Unknown method '%s'." % method)


def sinkhorn_knopp(
    a,
    b,
    C,
    reg=1e-1,
    maxIter=1000,
    stopThr=1e-9,
    verbose=False,
    warm_start=None,
    eval_freq=10,
    **kwargs
):
    P, log, cost = sinkhorn_knopp_fast(
        a, b, C, reg, maxIter, stopThr, verbose, warm_start, eval_freq, **kwargs
    )

    if (
        torch.logical_or(torch.isnan(P), torch.isinf(P)).any()
        or torch.logical_or(torch.isnan(log["beta"]), torch.isinf(log["beta"])).any()
    ):
        P2, log2, cost2 = sinkhorn_knopp_slow(
            a, b, C, reg, maxIter, stopThr, verbose, warm_start, eval_freq, **kwargs
        )
        return P2, log2, cost + cost2

    return P, log, cost


def sinkhorn_knopp_slow(
    a,
    b,
    C,
    reg=1e-1,
    maxIter=1000,
    stopThr=1e-9,
    verbose=False,
    warm_start=None,
    eval_freq=10,
    **kwargs
):
    """
    Solve the entropic regularization optimal transport
    The input should be PyTorch tensors
    The function solves the following optimization problem:

    .. math::
            \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
            s.t. \gamma 1 = a
                     \gamma^T 1= b
                     \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [1].

    Parameters
    ----------
    a : torch.tensor (na,)
            samples measure in the target domain
    b : torch.tensor (nb,)
            samples in the source domain
    C : torch.tensor (na,nb)
            loss matrix
    reg : float
            Regularization term > 0
    maxIter : int, optional
            Max number of iterations
    stopThr : float, optional
            Stop threshol on error ( > 0 )
    verbose : bool, optional
            Print information along iterations
    log : bool, optional
            record log if True

    Returns
    -------
    gamma : (na x nb) torch.tensor
            Optimal transportation matrix for the given parameters
    log : dict
            log dictionary return only if log==True in parameters

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    See Also
    --------

    """

    device = a.device
    na, nb = C.shape

    assert na >= 1 and nb >= 1, "C needs to be 2d"
    assert (
        na == a.shape[0] and nb == b.shape[0]
    ), "Shape of a or b does't match that of C"
    assert reg > 0, "reg should be greater than 0"
    assert a.min() >= 0.0 and b.min() >= 0.0, "Elements in a or b less than 0"

    log = {"err": []}

    if warm_start is not None:
        u = warm_start["u"]
        v = warm_start["v"]
    else:
        u = torch.ones(na, dtype=a.dtype).to(device) / na
        v = torch.ones(nb, dtype=b.dtype).to(device) / nb

    K = torch.empty(C.shape, dtype=C.dtype).to(device)
    torch.div(C, -reg, out=K)
    torch.exp(K, out=K)

    b_hat = torch.empty(b.shape, dtype=C.dtype).to(device)

    it = 1
    err = 1

    # allocate memory beforehand
    KTu = torch.empty(v.shape, dtype=v.dtype).to(device)
    Kv = torch.empty(u.shape, dtype=u.dtype).to(device)

    cost = 0

    while err > stopThr and it <= maxIter:

        upre, vpre = u, v
        torch.matmul(u, K, out=KTu)
        v = torch.div(b, KTu + M_EPS)
        torch.matmul(K, v, out=Kv)
        u = torch.div(a, Kv + M_EPS)

        pre_mul = time.time()
        # if torch.isnan(u).any() or torch.isnan(v).any() or torch.isinf(u).any() or torch.isinf(v).any():
        if (
            torch.logical_or(torch.isnan(u), torch.isinf(u)).any()
            or torch.logical_or(torch.isnan(v), torch.isinf(v)).any()
        ):
            print("Warning: numerical errors at iteration", it)
            u, v = upre, vpre
            break
        cost += time.time() - pre_mul

        if it % eval_freq == 0:
            # we can speed up the process by checking for the error only all
            # the eval_freq iterations
            # below is equivalent to:
            # b_hat = torch.sum(u.reshape(-1, 1) * K * v.reshape(1, -1), 0)
            # but with more memory efficient
            b_hat = torch.matmul(u, K) * v
            err = (b - b_hat).pow(2).sum().item()
            # err = (b - b_hat).abs().sum().item()
            log["err"].append(err)

        it += 1

    log["beta"] = reg * torch.log(v + M_EPS)

    # transport plan
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)

    return P, log, cost


def sinkhorn_knopp_fast(
    a,
    b,
    C,
    reg=1e-1,
    maxIter=1000,
    stopThr=1e-9,
    verbose=False,
    warm_start=None,
    eval_freq=10,
    **kwargs
):
    device = a.device
    na, nb = C.shape

    assert na >= 1 and nb >= 1, "C needs to be 2d"
    assert (
        na == a.shape[0] and nb == b.shape[0]
    ), "Shape of a or b does't match that of C"
    assert reg > 0, "reg should be greater than 0"
    assert a.min() >= 0.0 and b.min() >= 0.0, "Elements in a or b less than 0"

    log = {"err": []}

    if warm_start is not None:
        u = warm_start["u"]
        v = warm_start["v"]
    else:
        u = torch.ones(na, dtype=a.dtype).to(device) / na
        v = torch.ones(nb, dtype=b.dtype).to(device) / nb

    K = torch.empty(C.shape, dtype=C.dtype).to(device)
    torch.div(C, -reg, out=K)
    torch.exp(K, out=K)

    b_hat = torch.empty(b.shape, dtype=C.dtype).to(device)

    it = 1
    err = 1

    # allocate memory beforehand
    KTu = torch.empty(v.shape, dtype=v.dtype).to(device)
    Kv = torch.empty(u.shape, dtype=u.dtype).to(device)

    cost = 0
    pre_mul = time.time()

    while err > stopThr and it <= maxIter:

        upre, vpre = u, v
        torch.matmul(u, K, out=KTu)
        v = torch.div(b, KTu + M_EPS)
        torch.matmul(K, v, out=Kv)
        u = torch.div(a, Kv + M_EPS)

        if it % eval_freq == 0:
            # we can speed up the process by checking for the error only all
            # the eval_freq iterations
            # below is equivalent to:
            # b_hat = torch.sum(u.reshape(-1, 1) * K * v.reshape(1, -1), 0)
            # but with more memory efficient
            b_hat = torch.matmul(u, K) * v
            err = (b - b_hat).pow(2).sum().item()
            # err = (b - b_hat).abs().sum().item()
            log["err"].append(err)

        it += 1

    log["beta"] = reg * torch.log(v + M_EPS)

    # transport plan
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    cost += time.time() - pre_mul

    return P, log, cost

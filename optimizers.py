from __future__ import annotations

import enum
import os
import statistics as stats
from itertools import takewhile
from typing import Iterable, Optional, Sequence, Union

import numpy as np
from attr import frozen
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from scipy import optimize
from scipy.optimize import _optimize
from scipy.optimize import _minimize as mini








from typing_extensions import Literal

from .core.interval import Interval
from .core.optimizer import ObjectiveFn, Optimizer
from .core.sample import Sample

Samples = Sequence[Sample]
Bounds = Sequence[Interval]


class Behavior(enum.IntEnum):
    """Behavior when falsifying case for system is encountered.

    Attributes:
        FALSIFICATION: Stop searching when the first falsifying case is encountered
        MINIMIZATION: Continue searching after encountering a falsifying case until iteration
                      budget is exhausted
    """

    FALSIFICATION = enum.auto()
    MINIMIZATION = enum.auto()


def _sample_uniform(bounds: Bounds, rng: Generator) -> Sample:
    return Sample([rng.uniform(bound.lower, bound.upper) for bound in bounds])


def _minimize(samples: Samples, func: ObjectiveFn[float], nprocs: Optional[int]) -> Iterable[float]:
    if nprocs is None:
        return func.eval_samples(samples)
    else:
        return func.eval_samples_parallel(samples, nprocs)


def _falsify(samples: Samples, func: ObjectiveFn[float]) -> Iterable[float]:
    costs = map(func.eval_sample, samples)
    return takewhile(lambda c: c >= 0, costs)


"""Can't use eval_samples as is I don't beleive"""
"""https://proceedings.neurips.cc/paper_files/paper/2018/file/4e4b5fbbbb602b6d35bea8460aa8f8e5-Paper.pdf
https://icml.cc/2012/papers/489.pdf"""
""""""
"""def _maximize(func: ObjectiveFn[float],x0,  samples: Samples,  zita: float) -> Iterable[float]:
    if(zita <= 0):
        throw ValueError("zita must be greater than zero")
    x = asarray(x0).flatten()
    maxiter = N * 1000
    maxfun = N * 1000
    rewards = Iterable[float]
    lambda = 0.99 
    j0 = samples
    for j_lambda in j0:
        reward = lambda x: func.eval_sample(j_lambda)
        rewards = chain(rewards, reward)
        var = find_varience(rewards)
        j_lambda = reward - lambda(var - zita)

def find_varience(samples: Iterable[float]) -> float:
    mean = stats.mean(samples)
    sigma = sum(map(lambda x:np.square(x - mean)),samples)
    return np.sqrt(sigma/len(samples))
"""

"""Using a maximization version of powell as my prebuilt function already works for powell, and I can't find any preference on how a function minimizes in the papers I have seen
Logic works by taking the negation of func, minimizing the points as such, and then inverting the function again
Ideally, this should let me find the maximum, though I would need to proof-write this maybe
Anyway, this is mainly copied from scipy.optimize(method = 'powell') with the aformentioned changes"""
def maximize_powell(func, x0, args=(), callback=None, bounds=None,
                     xtol=1e-4, ftol=1e-4, maxiter=None, maxfev=None,
                     disp=False, direc=None, return_all=False,
                     **options):

    
    maxfun = maxfev
    retall = return_all
    bounds = mini.standardize_bounds(bounds,x0,'powell')
    x = np.asarray(x0).flatten()
    if retall:
        allvecs = [x]
    N = len(x)

    # If neither are set, then set both to default
    if maxiter is None and maxfun is None:
        maxiter = N * 1000
        maxfun = N * 1000
    elif maxiter is None:
        # Convert remaining Nones, to np.inf, unless the other is np.inf, in
        # which case use the default to avoid unbounded iteration
        if maxfun == np.inf:
            maxiter = N * 1000
        else:
            maxiter = np.inf
    elif maxfun is None:
        if maxiter == np.inf:
            maxfun = N * 1000
        else:
            maxfun = np.inf

    # we need to use a mutable object here that we can update in the
    # wrapper function

    fcalls, func = _optimize._wrap_scalar_function_maxfun_validation(func, args, maxfun) 

    if direc is None:
        direc = np.eye(N, dtype=float)

    else:
        direc = np.asarray(direc, dtype=float)
        
        if np.linalg.matrix_rank(direc) != direc.shape[0]:
            print("direc input is not full rank, some parameters may "
                          "not be optimized",
                          optimize.OptimizeWarning, 3)

    if bounds is None:
        # don't make these arrays of all +/- inf. because
        # _linesearch_powell will do an unnecessary check of all the elements.
        # just keep them None, _linesearch_powell will not have to check
        # all the elements.
        lower_bound, upper_bound = None, None
    else:

        # bounds is standardized in _minimize.py.
        lower_bound, upper_bound = bounds.lb, bounds.ub
        if np.any(lower_bound > x0) or np.any(x0 > upper_bound):
            print("Initial guess is not within the specified bounds",
                          optimize.OptimizeWarning, 3)
    
    fval = np.squeeze(func(x))
    x1 = x.copy()
    iter = 0
    while True:
        try:
            fx = fval
            bigind = 0
            delta = 0.0
            for i in range(N):
                direc1 = direc[i]

               
                fx2 = fval
                #This is where its important that func is reversed
                fval, x, direc1 = _optimize._linesearch_powell(func, x, direc1,
                                                     tol=xtol * 100,
                                                     lower_bound=lower_bound,
                                                     upper_bound=upper_bound,
                                                     fval=fval)
                fval *= -1
                x *= -1
                direc1 *= -1
                #now these checks need to be reversed
                #original#
                """                if (fx2 - fval) > delta:
                    delta = fx2 - fval
                    bigind = i
            iter += 1
            if retall:
                allvecs.append(x)
            intermediate_result = optimize.OptimizeResult(x=x, fun=fval)
            if _optimize._call_callback_maybe_halt(callback, intermediate_result):
                break
            bnd = ftol * (np.abs(fx) + np.abs(fval)) + 1e-20
            if 2.0 * (fx - fval) <= bnd:
                break
            if fcalls[0] >= maxfun:
                break
            if iter >= maxiter:
                break
            if np.isnan(fx) and np.isnan(fval):
                # Ended up in a nan-region: bail out
                break

            # Construct the extrapolated point
            direc1 = x - x1
            x1 = x.copy()
            # make sure that we don't go outside the bounds when extrapolating
            if lower_bound is None and upper_bound is None:
                lmax = 1
            else:
                _, lmax = _optimize._line_for_search(x, direc1, lower_bound, upper_bound)
            x2 = x + min(lmax, 1) * direc1
            fx2 = squeeze(func(x2))

            if (fx > fx2):
                t = 2.0*(fx + fx2 - 2.0*fval)
                temp = (fx - fval - delta)
                t *= temp*temp
                temp = fx - fx2
                t -= delta*temp*temp
                if t < 0.0:
                    fval, x, direc1 = _optimize._linesearch_powell(
                        func, x, direc1,
                        tol=xtol * 100,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        fval=fval
                    )
                    if np.any(direc1):
                        direc[bigind] = direc[-1]
                        direc[-1] = direc1"""
                if (fval- fx2) > delta:
                    delta = fval - fx2
                    bigind = i
            iter += 1
            if retall:
                allvecs.append(x)
            intermediate_result = optimize.OptimizeResult(x=x, fun=fval)
            if _optimize._call_callback_maybe_halt(callback, intermediate_result):
                break
            bnd = ftol * (np.abs(fx) + np.abs(fval)) + 1e-20
            if 2.0 * (fval - fx) <= bnd:
                break
            if fcalls[0] >= maxfun:
                break
            if iter >= maxiter:
                break
            if np.isnan(fx) and np.isnan(fval):
                # Ended up in a nan-region: bail out
                break

            # Construct the extrapolated point
            direc1 = x + x1
            x1 = x.copy()
            # make sure that we don't go outside the bounds when extrapolating
            if lower_bound is None and upper_bound is None:
                lmax = 1
            else:
                _, lmax = _optimize._line_for_search(x, direc1, lower_bound, upper_bound)
            x2 = x + max(lmax, 1) * direc1
            fx2 = np.squeeze(func(x2))

            if (fx > fx2):
                t = 2.0*(fx + fx2 - 2.0*fval)
                temp = (fx - fval - delta)
                t *= temp*temp
                temp = fx - fx2
                t -= delta*temp*temp
                if t < 0.0:
                    #reswitch
                    fval, x, direc1 = _optimize._linesearch_powell(
                        (lambda x : -x ,func), x, direc1,
                        tol=xtol * 100,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        fval=fval
                    )
                    if np.any(direc1):
                        direc[bigind] = direc[-1]
                        direc[-1] = direc1
        except _optimize._MaxFuncCallError:
            break

    warnflag = 0
    msg = _optimize._status_message['success']
    # out of bounds is more urgent than exceeding function evals or iters,
    # but I don't want to cause inconsistencies by changing the
    # established warning flags for maxfev and maxiter, so the out of bounds
    # warning flag becomes 3, but is checked for first.
    if bounds and (np.any(lower_bound > x) or np.any(x > upper_bound)):
        warnflag = 4
        msg = _optimize._status_message['out_of_bounds']
    elif fcalls[0] >= maxfun:
        warnflag = 1
        msg = _optimize._status_message['maxfev']
    elif iter >= maxiter:
        warnflag = 2
        msg = _optimize._status_message['maxiter']
    elif np.isnan(fval) or np.isnan(x).any():
        warnflag = 3
        msg = _optimize._status_message['nan']

    if disp:
        _optimize._print_success_message_or_warn(warnflag, msg, RuntimeWarning)
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % iter)
        print("         Function evaluations: %d" % fcalls[0])

    result = optimize.OptimizeResult(fun=fval, direc=direc, nit=iter, nfev=fcalls[0],
                            status=warnflag, success=(warnflag == 0),
                            message=msg, x=x)
    if retall:
        result['allvecs'] = allvecs
    return result


@frozen(slots=True)
class UniformRandomResult:
    """Data class that represents the result of a uniform random optimization.

    Attributes:
        average_cost: The average cost of all the samples selected.
    """

    average_cost: float


class UniformRandom(Optimizer[float, UniformRandomResult]):
    """Optimizer that implements the uniform random optimization technique.

    This optimizer picks samples randomly from the search space until the budget is exhausted.

    Args:
        parallelization: Value that indicates how many processes to use when evaluating each
                            sample using the cost function. Acceptable values are a number,
                            "cores", or None

    Attributes:
        processes: The number of processes to use when evaluating the samples.
    """

    def __init__(
        self,
        parallelization: Union[Literal["cores"], int, None] = None,
        behavior: Behavior = Behavior.FALSIFICATION,
    ):
        if isinstance(parallelization, int):
            self.processes: Optional[int] = parallelization
        elif parallelization == "cores":
            self.processes = os.cpu_count()
        else:
            self.processes = None

        self.behavior = behavior

    def optimize(
        self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int
    ) -> UniformRandomResult:
        rng = default_rng(seed)
        samples = [_sample_uniform(bounds, rng) for _ in range(budget)]

        if self.behavior is Behavior.MINIMIZATION:
            costs = _minimize(samples, func, self.processes)
        else:
            costs = _falsify(samples, func)

        average_cost = stats.mean(costs)

        return UniformRandomResult(average_cost)


@frozen(slots=True)
class DualAnnealingResult:
    """Data class representing the result of a dual annealing optimization.

    Attributes:
        jacobian_value: The value of the cost function jacobian at the minimum cost discovered
        jacobian_evals: Number of times the jacobian of the cost function was evaluated
        hessian_value: The value of the cost function hessian as the minimum cost discovered
        hessian_evals: Number of times the hessian of the cost function was evaluated
    """

    jacobian_value: Optional[NDArray[np.float_]]
    jacobian_evals: int
    hessian_value: Optional[NDArray[np.float_]]
    hessian_evals: int


class DualAnnealing(Optimizer[float, DualAnnealingResult]):
    """Optimizer that implements the simulated annealing optimization technique.

    The simulated annealing implementation is provided by the SciPy library dual_annealing function
    with the no_local_search parameter set to True.
    """

    def __init__(self, behavior: Behavior = Behavior.FALSIFICATION):
        self.behavior = behavior

    def optimize(
        self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int
    ) -> DualAnnealingResult:
        def listener(sample: NDArray[np.float_], robustness: float, ctx: Literal[-1, 0, 1]) -> bool:
            if robustness < 0 and self.behavior is Behavior.FALSIFICATION:
                return True

            return False

        result = optimize.dual_annealing(
            func=lambda x: func.eval_sample(Sample(x)),
            bounds=[bound.astuple() for bound in bounds],
            seed=seed,
            maxfun=budget,
            no_local_search=True,  # Disable local search, use only traditional generalized SA
            callback=listener,
        )

        try:
            jac: Optional[NDArray[np.float_]] = result.jac
            njev = result.njev
        except AttributeError:
            jac = None
            njev = 0

        try:
            hess: Optional[NDArray[np.float_]] = result.hess
            nhev = result.nhev
        except AttributeError:
            hess = None
            nhev = 0

        return DualAnnealingResult(jac, njev, hess, nhev)

@frozen(slots = True)
class CoordinateDescentResult:
    average_cost: float

    """Uses Powell minimization method to undergo block coordinate descent
    Is it efficent? No.
    Does it work?  Decently
    Will I need to make a better minimization method?  Absolutely
    Will I do that soon? Absolutely not"""
class CoordinateDescent(Optimizer[float, CoordinateDescentResult]):
    def __init__(self, behavior: Behavior = Behavior.FALSIFICATION):
        self.behavior = behavior

    def optimize(
        self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int
    ) -> CoordinateDescentResult:
        rng = default_rng(seed)
        samples = [_sample_uniform(bounds, rng)]
        for sample in samples:
            result = optimize.minimize(fun = lambda x: -1*func.eval_sample(Sample(x)),method = 'Powell', x0 = np.array([0,0]) ,  bounds=[bound.astuple() for bound in bounds], options = {'disp': True, 'maxfev' : budget*1000 })
            print(type(result))
            sample = Sample(sample + result.x) 
        costs =_minimize(samples, func,None)
        
        average_cost = stats.mean(costs)
        return CoordinateDescentResult(average_cost)

"""Coordinate Ascent function
Will be exactly like Coordinate descent except for the method, which is the actual important part"""

@frozen(slots = True)
class CoordinateAscentResult:
    average_cost: float 

class CoordinateAscent:
    def __init__(self, behavior: Behavior = Behavior.FALSIFICATION):
        self.behavior = behavior

    def optimize(
        self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int
    ) -> CoordinateDescentResult:
        rng = default_rng(seed)
        samples = [_sample_uniform(bounds, rng)]
        for sample in samples:
            result = optimize.minimize(fun = lambda x: func.eval_sample(Sample(x)),method = maximize_powell, x0 = np.array([0,0]) ,  bounds=[bound.astuple() for bound in bounds], options = {'disp': True, 'maxfev' : budget*1000 })
            print(type(result))
            sample = Sample(sample + result.x) 
        costs =_minimize(samples, func,None)
        
        average_cost = stats.mean(costs)
        return CoordinateDescentResult(average_cost)
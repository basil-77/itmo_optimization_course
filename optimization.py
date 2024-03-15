import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """

    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        # TODO_: Implement line search procedures for Armijo, Wolfe and Constant steps.

        if self._method == 'Constant':
            return self.c

        alpha = self.alpha_0 if previous_alpha is None else previous_alpha
        betha = 0.5


        # phi = PhiLineSearch(oracle.func, oracle.grad, x_k, d_k)

        phi_func = lambda alpha: oracle.func(x_k + alpha * d_k)
        phi_deriv = lambda alpha: np.dot(oracle.grad(x_k + alpha * d_k), d_k)

        if self._method == 'Wolfe':
            alpha, _, _, _ = scipy.optimize.linesearch.scalar_search_wolfe2(phi=phi_func, derphi=phi_deriv, c1=self.c1,
                                                                            c2=self.c2)

        if alpha is None or self._method == 'Armijo':
            alpha = self.alpha_0 if alpha is None else alpha
            # back-tracking
            while phi_func(alpha) > phi_func(0) + self.c1 * alpha * phi_deriv(0):
                alpha = alpha * betha

        # Wolfe
        # while True:
        #     if self._phi_func(alpha, oracle, x_k, d_k) > self._phi_func(0, oracle, x_k, d_k) + self.c1*self._phi_deriv(0, oracle, x_k, d_k) or alpha > alpha_prev:
        #         ==decrease(alpha_prev, alpha)
        #     else:
        #         phi_deriv = self._phi_deriv(alpha, oracle, x_k, d_k)
        #         if check_wolfe:
        #             break # we got alpha
        #         if phi_deriv >= 0:
        #             ==decrease(alpha, alpha_prev)
        #         else:
        #             #next alpha
        #             alpha_prev = alpha
        #             alpha = betha * alpha

        return alpha


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    grad_x_0 = oracle.grad(x_0)

    # TODO_: Implement gradient descent

    alpha_prev = None

    time_start = datetime.now()

    message = None

    def stop_cond(grad_x_k, grad_x_0, eps=1e-5):
        return np.power(np.linalg.norm(grad_x_k), 2) <= eps * np.power(np.linalg.norm(grad_x_0), 2)

    for i in range(max_iter):
        func = oracle.func(x_k)
        grad = oracle.grad(x_k)
        if trace:
            time_s = (datetime.now() - time_start).total_seconds()
            history['time'].append(time_s)
            history['func'].append(func)
            history['grad_norm'].append(np.linalg.norm(grad))
            if x_k.size <= 2:
                history['x'].append(x_k)
            else:
                history['x'].append(None)
        if display:
            print(f'iter {i}/{max_iter}: grad_norm={np.linalg.norm(grad)} time: {time_s} s')
        #time_start = datetime.now()
        if stop_cond(grad, grad_x_0, tolerance):
            message = 'success'
            break
        d_k = -grad
        alpha = line_search_tool.line_search(oracle, x_k, d_k, alpha_prev)
        if alpha is None:
            message = 'computational_error'
            break
        # alpha_prev = alpha
        x_k = x_k - alpha * grad

    if not message:
        message = 'iterations_exceeded'

    # Use line_search_tool.line_search() for adaptive step size.
    return x_k, message, history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    grad_x_0 = oracle.grad(x_0)

    # TODO_: Implement Newton's method.
    # Use line_search_tool.line_search() for adaptive step size.

    time_start = datetime.now()

    message = None

    def stop_cond(grad_x_k, grad_x_0, eps=1e-5):
        return np.power(np.linalg.norm(grad_x_k), 2) <= eps * np.power(np.linalg.norm(grad_x_0), 2)

    for i in range(max_iter):
        grad = oracle.grad(x_k)
        hess = oracle.hess(x_k)
        try:
            cho_factor = scipy.linalg.cho_factor(hess)
            d_k_cho = scipy.linalg.cho_solve(cho_factor, -grad)
        except LinAlgError:
            message = 'newton_direction_error'
            break
        except Exception:
            message = 'computational_error'
            break
        alpha = line_search_tool.line_search(oracle, x_k, d_k_cho, previous_alpha=1.0)
        if alpha is None:
            message = 'computational_error'
            break
        x_prev = x_k
        x_k = x_k + alpha * d_k_cho
        func = oracle.func(x_k)
        grad_norm = np.linalg.norm(oracle.grad(x_k))
        if trace:
            time_s = (datetime.now() - time_start).total_seconds()
            history['time'].append(time_s)
            history['func'].append(func)
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(x_k)
            else:
                history['x'].append(None)
        if display:
            print(f'iter {i}/{max_iter}: grad_norm={np.linalg.norm(grad)} time: {time_s} s')
        #time_start = datetime.now()
        if stop_cond(grad , grad_x_0, tolerance): # if np.isclose(x_k, x_prev)
            message = 'success'
            break

    if not message:
        message = 'iterations_exceeded'

    return x_k, message, history




# class PhiLineSearch:
#     def __init__(self, f_func, f_deriv, x_k, d_k):
#         self.f_func = f_func
#         self.f_deriv = f_deriv
#         self.x_k = x_k
#         self.d_k = d_k
#
#     def func(self, alpha):
#         return self.f_func(self.x_k + alpha*self.d_k)
#
#     def deriv(self, alpha):
#         return np.dot(self.f_deriv(self.x_k + alpha*self.d_k), self.d_k)

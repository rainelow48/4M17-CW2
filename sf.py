"""
Schwefel's function
f = sum(-x_i * sin (sqrt(|x_i|)))

Global minimum of -418.9829 * len(x) at (420.9687, ..., 420.9687) 
----------
f1 = 418.9829 * len(x) + sum(-x_i * sin(sqrt(|x_i|)))
Global minimum of f1 is 0 at (420.9687, ..., 420.9687).
----------
source: 
https://arxiv.org/pdf/1308.4008.pdf
https://www.sfu.ca/~ssurjano/schwef.html
"""
import numpy as np


class SF:

    def __init__(self):
        self.lim = 500

    def is_feasible(self, x: np.ndarray) -> np.ndarray:
        '''Check if x is in the feasible region
        Parameters
        ----------
        x: m points to be checked for feasibility. Dimension: (m, n)
        
        Returns
        -------
        check: (m, ) array containing boolean indicating feasibility of each of the m points in x
        '''
        return np.amax(np.abs(x), axis=1) <= self.lim

    def generate_feasible(self, dim: int) -> np.ndarray:
        '''Generate a random point in the feasible region
        Parameters
        ----------
        dim: Dimension of the Schwefel's function
        
        Returns
        -------
        x: A feasible point of the correct dimension
        '''
        return np.random.uniform(-self.lim, self.lim, dim)

    def cost(self, x: np.ndarray) -> np.ndarray:
        '''Evaluate Schwefel's function at a single point
        
        Parameters
        ----------
        x: A single point to be evaluated for n-dimension Schwefel's function. Dimension: (n,)
        
        Returns
        -------
        f: Schwefel's function evaluations at point x
        '''
        f = np.sum(-x * np.sin(np.sqrt(np.abs(x))))
        return f

    def cost_es(self, x: np.ndarray) -> np.ndarray:
        '''Evaluate Schwefel's function at multiple points
        
        Parameters
        ----------
        x: m points to be evaluated for n-dimension Schwefel's function. Dimension: (m, n)
        
        Returns
        -------
        f: Schwefel's function evaluations at points specified in x
        '''

        assert self.is_feasible(x).all() == True

        f = np.sum(-x * np.sin(np.sqrt(np.abs(x))), axis=1)
        return f

from typing import Any
import numpy as np

class HebbianUpdate:
    """
    Empty class
    """
    
    def __call__(self, w, u, v) -> Any: 
        """
        HebbianUpdate should implement the call method that takes in input
        the weight vector w, the input vector u and the output v and returns the updated weight vector w.
        If there are other parameters that need to be passed to the update rule, they should be passed in the constructor.
        """
        raise NotImplementedError("Subclasses should implement this!")

class BasicHebbianUpdate(HebbianUpdate):
    """
    Basic Hebbian update rule
    """
    def __call__(self, w, u, v) -> np.ndarray:
        return v * u


class OjaUpdate(HebbianUpdate):
    """
    Oja's update rule: w(t+1) = w(t) + alpha * v * (u - v * w(t))
    """
    def __init__(self, alpha) -> None:
        self.alpha = alpha

    def __call__(self, w, u, v) -> np.ndarray:
        return self.alpha * v * (u - self.alpha * v * w)


class SubtractiveNormUpdate(HebbianUpdate):
    """
    Subtractive normalization update rule: w(t+1) += v * (u - (u^T ones) * ones / n_u)

    """
    def __call__(self, w, u, v) -> np.ndarray:
        n = np.ones_like(u)
        n_u = len(u)
        return v * (u - (np.dot(u, n)) * n / n_u)


class BCMUpdate(HebbianUpdate):
    """
    Bienenstock-Cooper-Munro update rule: w(t+1) = w(t) + v * (u - theta * v^2)
    """
    def __init__(self, theta, eta_th) -> None:
        self.theta = theta
        self.eta_th = eta_th

    def _update_theta(self, v)-> float:
        return v**2 - self.theta

    def __call__(self, w, u, v) -> np.ndarray:
         w_upd=  v * u * (v -self.theta)
         self.theta += self.eta_th * self._update_theta(v)
         return w_upd
        
        

        
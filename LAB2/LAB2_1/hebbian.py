from updates import HebbianUpdate
import numpy as np

class Hebbian:
    """
    Hebbian learning algorithm
    params:
    dim: int: dimension of the input data
    update_func: HebbianUpdate: update rule to use (implemented: basic hebbian, oja)
    """
    def __init__(self, dim: int, update_func: HebbianUpdate) -> None:
        self.w = (np.random.rand(1, dim) - 0.5) * 2 # weight vector in range [-1, 1]
        self.update_func = update_func

    def predict(self, data):
        return np.dot(data, self.w.T) 

    def train(self, data, eta,  epochs: int = 100):
        # randomly shuffle the data
        ws = []
        for epoch in range(epochs):
            np.random.shuffle(data)
            for u in data:
                v = self.predict(u)
                self.w += eta * self.update_func(w=self.w, u=u, v=v)
                ws.append(self.w.copy())
        return np.array(ws).squeeze()
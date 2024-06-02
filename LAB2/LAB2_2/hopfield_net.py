import numpy as np
import matplotlib.pyplot as plt


class HopfieldNetwork():
    def __init__(self) -> None:
        self.W = None

    def training(self, data: np.array):
        n_examples, dim = data.shape
        self.W = np.zeros((dim,dim ))
        for i in range(n_examples):
            self.W += np.outer(data[i], data[i])
        self.W = self.W - n_examples * np.eye(dim)
        self.W /= dim
        self.n_examples = n_examples
        self.dim = dim
    
    def retrieval(self, x, max_steps=5, bias=0.6, compute_history=False, orig=None):
        """
        Perform retrieval process in the Hopfield network.

        Parameters:
        - x (ndarray): The input pattern to retrieve.
        - data (ndarray): The training data used to train the network.
        - max_steps (int): The maximum number of retrieval steps to perform.
        - bias (float or ndarray, optional): The bias value(s) for the neurons. If a scalar, it is assumed to be the same for all neurons. If an ndarray, it should have the same shape as the number of neurons in the network. Default is 0.6.
        - compute_history (bool, optional): Whether to compute and store the energy and overlap history during retrieval. Default is False.
        - orig (ndarray, optional): The original pattern to compare the retrieved pattern with. Required if compute_history is True.
        Returns:
        - ndarray: The retrieved pattern
        """
        # check if the network is trained
        assert self.W is not None, 'network is not trained'
        x = x.reshape(-1, 1)

        if compute_history:
            assert orig is not None, 'original pattern must be provided to compute history'
            orig = orig.reshape(-1, 1)
            assert orig.shape == x.shape, 'original pattern must have the same shape as the input pattern'
            self.history = {'energy': [], 'overlap': []}  # store the energy over time
            Wx = np.dot(self.W, x)
            inefficient_new_energy = (-1 / 2 * x.T @ Wx)
            self.history['energy'].append(inefficient_new_energy)
            # compute overlap with original pattern
            overlap = np.sum(x == orig) / self.dim
            self.history['overlap'].append(overlap)
        # if the bias is a scalar, we assume it is the same for all neurons
        if np.isscalar(bias):
            bias = np.ones(self.dim) * bias
        # if the bias is a vector, it should have the same shape as the number of neurons
        else:
            print(bias.shape)
            bias = bias.reshape(-1,)
            assert bias.shape == (self.dim,), 'bias should have the same shape as the number of neurons'            
        x_new = x.copy()
        for i in range(max_steps):
            neurons = np.random.permutation(self.dim)
            x_old = x_new.copy()
            for neuron in neurons:
                new_value = np.dot(self.W[neuron, :], x_new) + bias[neuron]
                x_new[neuron] = np.sign(new_value)
                if compute_history:
                    energy = (-1 / 2 * x_new.T @ self.W @ x_new)  # TODO: is there a more efficient way to update from the last energy?
                    self.history['energy'].append(energy)
                    overlap = np.sum(x_new == orig) / self.dim
                    self.history['overlap'].append(overlap)

            if np.all(x_old == x_new):  # checks if after a change to every component, the vector is the same.
                print(f'converged in {i + 1} steps')
                break
        return x_new
    
    def plot_energy(self, title='Energy over iterations'):
        if not hasattr(self, 'history'):
            print('energy not computed')
            return
        self.history['energy'] = np.array(self.history['energy']).reshape(-1,)
        plt.plot(self.history['energy'])
        plt.title(title)
        plt.xlabel('iteration')
        plt.ylabel('energy')

    
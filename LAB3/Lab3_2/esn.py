import numpy as np 
from numpy import tanh
import sklearn
from scipy import sparse


class LeastMeanSquares():
    def __init__(self) -> None:
        pass
    def fit(self, H, y):
        H_pinv =  np.linalg.pinv(H)
        self.W = H_pinv @ y
        
    def predict(self, H):
        return  H @ self.W
    

import sklearn.linear_model

def random_matrix(n, m, density):
    """
    Generate a random n times m matrix with around density% non-zero elements
    """
    A = sparse.random(n, m, density=density)
    return A


class EchoStateNetwork():
    def __init__(self, input_size, hidden_size, rho_h, omega_x, density=0.1, head=None) -> None:
        # initialize the weights
        self.W_x = random_matrix(hidden_size, input_size, density)
        self.W_h = random_matrix(hidden_size, hidden_size, density)
        print(type(self.W_h))
        # initialize the hidden state and bias
        self.last_h = np.zeros([hidden_size, 1])
        self.bias = np.ones_like(self.last_h) # TODO: implement more initialization strategies
        # initialize weights at stability
        norm_W_x = sparse.linalg.norm(self.W_x)
        self.W_x *= omega_x / norm_W_x

        rho = max(abs(np.linalg.eigvals(self.W_h.toarray())))
        self.W_h *= rho_h/rho
        self.head=head
    
    def create_reservoir(self, X, alpha_leak = 1.0):
        self.hidden_states=[]
        for x_i in X:
            x_i = x_i.reshape(-1,1)
            self.last_h= alpha_leak * tanh(self.W_x.dot( x_i)  + self.W_h.dot(self.last_h) + self.bias) + (1-alpha_leak)*self.last_h
            self.hidden_states.append(self.last_h)
        self.hidden_states=np.stack( self.hidden_states, axis=0).squeeze()
        return self.hidden_states

    def fit(self, X, y, washout =10):
        """
            params:
            X: list of numpy arrays or numpy array of shape sequence_length x input_size
            y: labels of shape sequence_length x n_features
        """
        # create the reservoir
        self.create_reservoir(X)
        #discard the washout
        self.hidden_states=self.hidden_states[washout:,:] 
        # print(self.hidden_states.shape)
        y = y[washout:, :]
        if self.head is not None:
            self.head.fit(self.hidden_states, y)
        else:
            # throw error
            print("No head defined")


    def predict(self, X):
        self.hidden_states=[]
        # create the reservoir
        self.create_reservoir(X)
        
        y_pred = self.head.predict(self.hidden_states)
        return y_pred


        
if __name__ == "__main__":
    # Test the ESN
    A= random_matrix(300,30,0.1)
    print(A)
    print(len(A.nonzero()[0]))
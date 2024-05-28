import numpy as np 
from numpy import tanh
import sklearn
from scipy import sparse
import scipy.stats
from tqdm import tqdm
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
    Generate a random n times m matrix with around density% non-zero elements with values in [-1, 1]
    """
    rvs = scipy.stats.uniform(loc=-1, scale=2).rvs
    A = scipy.sparse.random(n, m, density=density, data_rvs=rvs)   

    return A

def random_orthogonal_matrix(n):
    A = np.random.rand(n,n)
    Q, _ = np.linalg.qr(A)
    return Q

class EchoStateNetwork():
    def __init__(self,
                input_size,
                hidden_size,
                rho_h, omega_x,
                density=0.1,
                orthogonal_matrix_leakage='identity',
                head=None) -> None:
        # initialize the weights
        self.W_x = random_matrix(hidden_size, input_size, density)
        self.W_h = random_matrix(hidden_size, hidden_size, density)
        # initialize the hidden state and bias
        self.last_h = np.zeros([hidden_size, 1])
        self.bias = np.zeros_like(self.last_h) # TODO: implement more initialization strategies
        # initialize weights at stability
        norm_W_x = sparse.linalg.norm(self.W_x)
        self.W_x *= omega_x / norm_W_x
        rho = max(abs(np.linalg.eigvals(self.W_h.toarray())))
        self.W_h *= rho_h/rho
        self.head=head
        # create the orthogonal matrix for the leakage part
        if isinstance(orthogonal_matrix_leakage, str): # random, identity, cycle
            self.Q = self.create_orthogonal_matrix(hidden_size, orthogonal_matrix_leakage)
        elif isinstance(orthogonal_matrix_leakage, np.ndarray):
            try:
                assert orthogonal_matrix_leakage.shape == (hidden_size, hidden_size)
                self.Q = orthogonal_matrix_leakage
            except AssertionError:
                raise ValueError("Orthogonal matrix leakage should be of shape (hidden_size, hidden_size)")
        else:
            raise ValueError("Orthogonal matrix leakage should be of type str (accepted values 'identity', 'random', 'cycle') or numpy.ndarray of shape (hidden_size, hidden_size)")
    
    def create_orthogonal_matrix(self, hidden_size, orthogonal_matrix_leakage):
        if orthogonal_matrix_leakage == 'random':
            return random_orthogonal_matrix(hidden_size)
        elif orthogonal_matrix_leakage == 'identity':
            return np.eye(hidden_size)
        elif orthogonal_matrix_leakage == 'cycle':
            subdiag = np.ones(hidden_size-1)
            Q = np.diag(subdiag, -1)
            Q[0,-1] = 1.
            return Q
        else:
            raise ValueError("Unknown orthogonal matrix leakage type")
    
    def create_reservoir(self, X, alpha_leak = 1.0):
        self.hidden_states=[]
        self.last_h = np.zeros([self.W_x.shape[0], 1])
        for x_i in X:
            x_i = x_i.reshape(-1,1) # FIXME: only works for 1D input
            self.last_h= alpha_leak * tanh(self.W_x.dot( x_i)  + self.W_h.dot(self.last_h) + self.bias) + (1-alpha_leak)* (np.matmul(self. Q,  self.last_h))
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

    def create_reservoir_batched(self, X, alpha_leak = 1.0):
        self.hidden_states=[]
        self.last_h = np.zeros((self.W_x.shape[0], X.shape[0]))
        X=np.transpose(X, (1,0,2)) # sequence_length x batch_size x input_size
        print(X.shape)
        for x_i in tqdm(X): # x_i is of shape batch_size x input_size
            """print(x_i.shape)
            print(self.W_x.shape)
            print(self.W_h.shape)
            print(self.last_h.shape)"""
            self.last_h= alpha_leak * tanh(self.W_x.dot( x_i.T)  + self.W_h.dot(self.last_h) + self.bias) + (1-alpha_leak)* (np.matmul(self. Q,  self.last_h))
            self.hidden_states.append(self.last_h)
        self.hidden_states=np.stack( self.hidden_states, axis=0).squeeze() # sequence_length x batch_size x hidden_size
        return self.hidden_states
    
    def fit_batch(self, X, y, washout =10):
        """
            params:
            X: list of numpy arrays or numpy array of shape batch_size x sequence_length x input_size
            y: labels of shape batch_size x sequence_length x n_features
        """
        # create the reservoir
        self.create_reservoir_batched(X)
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
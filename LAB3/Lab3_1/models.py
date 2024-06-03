import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
class TDNN(nn.Module):
    def __init__(self, window_size:int, hidden_size: int, output_size:int):
        super(TDNN, self).__init__()
        self.window_size =window_size
        self.td = nn.Linear(window_size, hidden_size)   
        self.relu=nn.ReLU()
        self.linear = nn.Linear(hidden_size, output_size)
    
    def pad_input(self, x:torch.Tensor):
        pad_size = self.window_size - x.shape[1]
        return torch.cat(torch.zeros(1, pad_size), x)

    def forward(self, x:torch.Tensor):
        """
        just a 2 layer feedforward network applied on time windows of the input
        """
        if x.shape[1] < self.window_size:
            x= self.pad_input(x)
        h = self.relu(self.td(x))
        o = self.linear(h)
        return o.squeeze(0)
        
    
class VanillaRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, nonlinearity='relu') -> None:
        super().__init__()
        self.rnn =nn.RNN(input_size, hidden_size, nonlinearity = nonlinearity, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h_0 =None):
        outputs, h_T = self.rnn(x, h_0)
        y = self.linear(outputs)
        return y, h_T
 

def train_tdnn(tdnn, optimizer, loss_fn, mae_fn, x_train, y_train, x_val, y_val, epochs=100, n_steps_update=32, patience=5, eps=1e-5):
    train_losses_tdnn = []
    train_maes_tdnn = []
    val_losses_tdnn = []
    val_maes_tdnn = []
    best_val_loss = np.inf
    patience_counter = 0
    
    for epoch in range(epochs):
        running_loss = 0
        running_mae = 0
        print(f'Epoch {epoch + 1}')
        
        # Training phase
        tdnn.train()
        train_progress = tqdm(range(tdnn.window_size - 1, x_train.shape[1]), desc='Training', leave=False)
        for i in train_progress: # iterate over the time series in windows
            x_i = x_train[:, i - tdnn.window_size + 1: i + 1]
            y_hat = tdnn(x_i)
            l = loss_fn(y_hat, y_train[:, i])
            m = mae_fn(y_hat, y_train[:, i])
            running_loss += l.item()
            running_mae += m.item()
            l.backward()
            if i % n_steps_update == 0 or i == x_train.shape[1] - 1:
                optimizer.step()
                optimizer.zero_grad()
            
                train_progress.set_postfix({'loss': running_loss / (i + 1), 'mae': running_mae / (i + 1)})
        
        running_loss /= x_train.shape[1]
        running_mae /= x_train.shape[1]
        train_losses_tdnn.append(running_loss)
        train_maes_tdnn.append(running_mae)
        print(f'Training loss: {running_loss:.4f}')
        print(f'Training MAE: {running_mae:.4f}')
        print('Relative MAE: ', running_mae / y_train.abs().mean().item())

        # Validation phase
        if x_val is None or y_val is None:
            continue
        # else, evaluate the model on the validation set
        tdnn.eval()
        val_loss = 0
        val_mae = 0
        y_hats = torch.Tensor()
        val_progress = tqdm(range(tdnn.window_size - 1, x_val.shape[1]), desc='Validation', leave=False)
        for i in val_progress:
            x_i = x_val[:, i - tdnn.window_size + 1: i + 1]
            with torch.no_grad():
                y_hat = tdnn(x_i)
                y_hats = torch.cat((y_hats, y_hat))
                val_loss += loss_fn(y_hat, y_val[:, i]).item()
                val_mae += mae_fn(y_hat, y_val[:, i]).item()
            
            val_progress.set_postfix({'val_loss': val_loss / (i + 1), 'val_mae': val_mae / (i + 1)})
        
        val_loss /= x_val.shape[1]
        val_mae /= x_val.shape[1]
        val_losses_tdnn.append(val_loss)
        val_maes_tdnn.append(val_mae)
        print(f'Validation loss: {val_loss:.4f}')
        print(f'Validation MAE: {val_mae:.4f}')
        print('Relative MAE: ', val_mae / y_val.abs().mean().item())
        print('-----------------------------------')
        # Early stopping
        if val_loss < best_val_loss - eps:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs.')
            break
    return train_losses_tdnn, train_maes_tdnn, val_losses_tdnn, val_maes_tdnn, epoch



def train_rnn(rnn, optimizer, loss_fn, mae_fn, x_train, y_train, x_val, y_val, prop_length=32, epochs=300, patience=20, eps=1e-6, gradient_clip=0):
    train_losses_rnn = []
    train_maes_rnn = []
    val_losses_rnn = []
    val_maes_rnn = []
    best_val_loss = np.inf
    loop = tqdm(range(epochs), desc='Training')
    for epoch in loop:
        loop.set_description('Training, epoch: {}'.format(epoch))
        # training phase
        rnn.train()
        h_last = None
        running_loss = 0
        running_mae = 0
        for i in range(0, x_train.shape[1], prop_length): # we backpropagate the gradient through time only for a fixed number of steps
            x_batch = x_train[:, i: i + prop_length]
            y_batch = y_train[:, i: i + prop_length]
            y_hat, h_last = rnn(x_batch, h_last)
            h_last = h_last.detach()
            l = loss_fn(y_hat, y_batch)
            m = mae_fn(y_hat, y_batch)
            optimizer.zero_grad()
            l.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(rnn.parameters(), gradient_clip)
            optimizer.step()
            running_loss += l.item() / prop_length
            running_mae += m.item() / prop_length

        train_losses_rnn.append(running_loss)
        train_maes_rnn.append(running_mae)
   
        if x_val is None or y_val is None:
            loop.set_postfix({'loss': running_loss, 'mae':running_mae})
            continue
        # validation phase
        rnn.eval()
        with torch.no_grad():
            y_hat, _ = rnn(x_val, h_last)
            v_l = loss_fn(y_hat, y_val)
            v_m = mae_fn(y_hat, y_val)
            val_losses_rnn.append(v_l.item())
            val_maes_rnn.append(v_m.item())
            loop.set_postfix({'loss': running_loss, 'mae':running_mae, 'val_loss': v_l.item(), 'val_mae': v_m.item()})
        # early stopping
        if v_l < best_val_loss - eps:
            best_val_loss = v_l
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs.')
            break

    
    return train_losses_rnn, train_maes_rnn, val_losses_rnn, val_maes_rnn


if __name__=='__main__':
    tdnn=TDNN(input_size=1, hidden_size=2, window_size=2, output_size=1)
    input = torch.ones(1,100)
    tdnn(input)
        
        

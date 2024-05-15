import torch
import torch.nn as nn
import pandas as pd
"""
    Class for TDNN
"""

class TDNN(nn.Module):
    def __init__(self, window_size:int, hidden_size: int, output_size:int):
        super(TDNN, self).__init__()
        self.window_size =window_size
        self.td = nn.Linear(window_size, hidden_size)   
        self.relu=nn.ReLU()
        self.mlp = nn.Linear(hidden_size, output_size)
    
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
        o = self.mlp(h)
        return o.squeeze(0)

"""
training loop TDNN
""" 
def training_loop_tdnn(model:TDNN, optimizer, x_train, y_train, x_val, y_val):
    # training loop
    val_mae =0
    val_loss=0
    epochs =100
    n_steps_update =32 #number of steps before updating the weights
    train_losses_tdnn =[]
    train_maes_tdnn=[]
    val_losses_tdnn =[]
    val_maes_tdnn=[]
    loss = torch.nn.MSELoss() 
    mae = torch.nn.L1Loss()
    y_val = torch.cat((y_train[:, -model.window_size:], y_val), dim=1) # append the last elements of training sequence to validation to perform tdnn 
    for epoch in range(epochs):
        running_loss =0
        running_mae =0
        print(f'Epoch {epoch+1}')
        for i in range( model.window_size -1, x_train.shape[1]): # from window_size to length of training data 
            x_i = x_train[:, i -model.window_size +1 : i+1]
            y_hat = model(x_i)
            l = loss( y_hat, y_train[:, i])
            m = mae( y_hat, y_train[:, i])
            running_loss += l
            running_mae += m
            l.backward()
            if i% n_steps_update ==0 or i == x_train.shape[1]-1:
                optimizer.step()
                optimizer.zero_grad()
        running_loss = running_loss/x_train.shape[1]
        running_mae = running_mae/x_train.shape[1]
        train_losses_tdnn.append(running_loss.detach())
        train_maes_tdnn.append(running_mae.detach())
        print(f'Training loss: {running_loss}')
        print(f'Training MAE: {running_mae}')
        print('Relative MAE: ', running_mae / y_train.abs().mean())
        # validation
        val_loss = 0
        val_mae = 0
        y_hats = torch.Tensor()
        for i in range( model.window_size -1, x_val.shape[1]):
            x_i = x_val[:, i -model.window_size +1 : i+1]
            with torch.no_grad():
                y_hat = model(x_i)
                y_hats = torch.cat((y_hats, y_hat))
                val_loss += loss( y_hat, y_val[:, i]) 
                val_mae += mae( y_hat, y_val[:, i])
        val_loss = val_loss/x_val.shape[1]
        val_mae = val_mae/x_val.shape[1]
        val_losses_tdnn.append(val_loss.detach())
        val_maes_tdnn.append(val_mae.detach())
        print(f'Validation loss: {val_loss}')
        print(f'Validation MAE: {val_mae}')
        print('Relative MAE: ', val_mae / y_val.abs().mean())
        print('-----------------------------------')
    return val_losses_tdnn, val_maes_tdnn, y_hats


def load_narma(path_to_ds ='datasets/NARMA10.csv', format='torch'):
    df = pd.read_csv(path_to_ds, header=None)
    x = torch.Tensor(df.iloc[0,:])
    y = torch.Tensor(df.iloc[1,:])
    x =x.unsqueeze(0)
    y =y.unsqueeze(0)
    x.shape, y.shape
    x_train = x[:, :4000]
    x_val = x[:, 4000:5000]
    x_test = x[:, 5000:]
    y_train = y[:, :4000]
    y_val = y[:, 4000:5000]
    y_test = y[:, 5000:]
    if format =='np':
        x_train=x_train.numpy()
        x_val=x_val.numpy()
        x_test=x_test.numpy()
        y_train=y_train.numpy()
        y_val=y_val.numpy()
        y_test = y_test.numpy()

    return x_train.T, y_train.T, x_val.T, y_val.T, x_test.T, y_test.T

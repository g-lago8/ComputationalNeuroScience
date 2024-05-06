import torch
import torch.nn as nn

class TDNN(nn.Module):
    def __init__(self, input_size, hidden_size, window_size, output_size) -> None:
        super().__init__()
        self.td = nn.Conv1d(in_channels= input_size, out_channels = hidden_size, kernel_size =window_size, stride=1, padding='valid', dilation=1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.relu =nn.ReLU()

    def forward(self, x):
        h = self.relu(self.td(x))
        print(h.shape)
        return None
    

if __name__=='__main__':
    tdnn=TDNN(input_size=1, hidden_size=2, window_size=2, output_size=1)
    input = torch.ones(1,100)
    tdnn(input)
        
        

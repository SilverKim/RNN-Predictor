import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size


        self.input_to_hidden = nn.Linear(self.input_size + self.hidden_size, self.hidden_size, bias=False)
        self.hidden_to_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x, hidden_state):
        # x is expected to be of shape [batch_size, input_size]
        # hidden_state is expected to be of shape [batch_size, hidden_size]

        # Concatenate input and hidden state on the last dimension
        combined = torch.cat((x, hidden_state), dim=1)
        hidden_state = torch.tanh(self.input_to_hidden(combined))
        hidden_state = self.hidden_to_hidden(hidden_state)
        output = self.hidden_to_output(hidden_state)

        return output, hidden_state
    
    def init_hidden(self,batch_size):
        # Adjust the initial hidden state to the batch size
        return torch.zeros(batch_size, self.hidden_size)


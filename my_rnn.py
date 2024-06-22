import torch
import torch.nn as nn

class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomRNN, self).__init__()

        self.hidden_size = hidden_size

        # Combine input and hidden state into one linear layer
        self.input_to_hidden = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden_state):
        # Concatenate input and hidden state
        combined = torch.cat((x, hidden_state), dim=1)
        
        # Compute the new hidden state
        hidden_state = torch.tanh(self.input_to_hidden(combined))
        
        # Compute the output
        output = self.hidden_to_output(hidden_state)

        return output, hidden_state
    
    def init_hidden(self, batch_size):
        # Initialize the hidden state to zeros
        return torch.zeros(batch_size, self.hidden_size)

# Example usage:
# rnn = CustomRNN(input_size=10, hidden_size=20, output_size=5)
# x = torch.randn(batch_size, input_size)
# hidden_state = rnn.init_hidden(batch_size)
# output, hidden_state = rnn(x, hidden_state)

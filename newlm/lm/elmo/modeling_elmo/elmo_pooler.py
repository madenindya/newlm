from torch import nn

class ELMOBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size + config.hidden_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, last_token_hidden_states):
        # expected input already last_token_hidden_state from l2r and r2l
        pooled_output = self.dense(last_token_hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output
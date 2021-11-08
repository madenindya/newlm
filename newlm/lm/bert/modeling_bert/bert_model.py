from torch import nn
from transformers import BertModel, BertForSequenceClassification
from newlm.lm.elmo.modeling_elmo.elmo_utils import get_sequence_lengths


class BertCausalPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, batch_size, sequence_lengths):
        # We "pool" LAST model by simply taking the hidden state corresponding
        # to the LAST token for each sequence
        last_token_tensor = hidden_states[range(batch_size), sequence_lengths]
        pooled_output = self.dense(last_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertCausalModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(
            config, add_pooling_layer=False
        )  # no need to pool from original model

        self.pooler_causal = BertCausalPooler(config) if add_pooling_layer else None

    def forward(self, **bert_input):
        bert_output = super().forward(**bert_input)

        # Pooled here
        if self.pooler_causal is not None:
            input_ids = bert_input["input_ids"]
            sequence_output = bert_output.last_hidden_state
            bs, seqlen = get_sequence_lengths(
                pad_token_id=self.config.pad_token_id, input_ids=input_ids
            )
            pooled_output = self.pooler_causal(
                hidden_states=sequence_output, batch_size=bs, sequence_lengths=seqlen
            )
            bert_output.pooler_output = pooled_output

        return bert_output

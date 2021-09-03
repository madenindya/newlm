from transformers import GPT2PreTrainedModel
import torch
from torch import nn
from .elmo_config import ELMOConfig
from .elmo_model import ELMOGPTHeadModel
from transformers.modeling_outputs import SequenceClassifierOutput


class ELMOForClassification(GPT2PreTrainedModel):
    def __init__(self, config: ELMOConfig):
        super().__init__(config)

        self.elmo = ELMOGPTHeadModel(config)

        # add classification layer
        self.num_labels = config.num_labels
        self.score = nn.Linear(
            config.l2r_gpt_config.hidden_size + config.r2l_gpt_config.hidden_size,
            self.num_labels,
            bias=False,
        )

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        return_dict=None,
    ):

        # get outputs
        elmo_out = self.elmo(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=None,  # no need to pass the classification labels to the LM
            use_cache=use_cache,
            output_attentions=False,  # no need
            output_hidden_states=True,  # set to True to get the hidden state
            return_dict=return_dict,
        )

        # get hidden states
        l2r_hidden_states = elmo_out.l2r_last_hidden_state
        r2l_hidden_states = elmo_out.r2l_last_hidden_state

        (batch_size, sequence_lengths) = self.elmo.get_sequence_lengths(
            input_ids=input_ids, inputs_embeds=inputs_embeds
        )

        l2r_last_hidden_states = l2r_hidden_states[range(batch_size), sequence_lengths]
        r2l_last_hidden_states = r2l_hidden_states[range(batch_size), sequence_lengths]

        # combine hidden states
        combined_hidden_states = torch.cat(
            [l2r_last_hidden_states, r2l_last_hidden_states], dim=1
        )

        # get logits
        logits = self.score(combined_hidden_states)

        # get loss
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.to(self.dtype).view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=combined_hidden_states,
            attentions=None,
        )

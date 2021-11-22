from torch import nn
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from newlm.lm.elmo.modeling_elmo.elmo_utils import get_sequence_lengths

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

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
        # print("Forward Bert Causal")
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

class BertModelCausalForSequenceClassification(BertPreTrainedModel):
    """
    Copied BertForSequenceClassification
    Use the original BertModel as base bert model
    Modified forward and replace pooling with BertCausalPooling
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.pooler_causal = BertCausalPooler(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # print("Forward bert causal seqclass ")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Start modification: causal pooling
        # pooled_output = outputs[1]
        sequence_output = outputs.last_hidden_state
        bs, seqlen = get_sequence_lengths(
            pad_token_id=self.config.pad_token_id, input_ids=input_ids
        )
        pooled_output = self.pooler_causal(
            hidden_states=sequence_output, batch_size=bs, sequence_lengths=seqlen
        )
        # End modification

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

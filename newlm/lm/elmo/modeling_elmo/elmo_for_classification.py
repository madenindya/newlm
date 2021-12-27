from transformers import (
    GPT2Config,
    GPT2PreTrainedModel,
    BertConfig,
    BertPreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from .elmo_model import ELMOGPTModel, ELMOBertModel
from .elmo_utils import get_sequence_lengths
from .elmo_pooler import ELMOBertPooler

from newlm.lm.bert.modeling_bert.bert_model import BertCausalPooler


class ELMOGPTForSequenceClassification(GPT2PreTrainedModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)

        self.transformer = ELMOGPTModel(config)

        # add classification layer
        self.num_labels = config.num_labels
        self.score = nn.Linear(
            config.hidden_size + config.hidden_size,
            self.num_labels,
            bias=False,
        )

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

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
        gpt_args = locals()
        gpt_args.pop("self")
        labels = gpt_args.pop("labels")

        # get outputs
        elmo_out = self.transformer(**gpt_args)

        # hidden_state left to right is in index 0 of transformer_outputs
        # hidden_state right to left is in index 1 of transformer_outputs

        # Get elmo out
        l2r_last_hidden_state = elmo_out.last_hidden_state[0]
        r2l_last_hidden_state = elmo_out.last_hidden_state[1]

        (batch_size, sequence_lengths) = get_sequence_lengths(
            pad_token_id=self.config.pad_token_id,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )

        l2r_last_hidden_state = l2r_last_hidden_state[
            range(batch_size), sequence_lengths
        ]
        r2l_last_hidden_state = r2l_last_hidden_state[
            range(batch_size), sequence_lengths
        ]

        # combine hidden states
        combined_hidden_states = torch.cat(
            [l2r_last_hidden_state, r2l_last_hidden_state], dim=1
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


class ELMOBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)

        self.transformer = ELMOBertModel(config)

        self.pooler = ELMOBertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # add classification layer
        self.num_labels = config.num_labels
        self.score = nn.Linear(
            config.hidden_size + config.hidden_size,
            self.num_labels,
        )

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

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
        gpt_args = locals()
        gpt_args.pop("self")
        labels = gpt_args.pop("labels")

        # get outputs
        elmo_out = self.transformer(**gpt_args)

        # hidden_state left to right is in index 0 of transformer_outputs
        # hidden_state right to left is in index 1 of transformer_outputs

        # Get elmo out
        l2r_last_hidden_state = elmo_out.last_hidden_state[0]
        r2l_last_hidden_state = elmo_out.last_hidden_state[1]

        (batch_size, sequence_lengths) = get_sequence_lengths(
            pad_token_id=self.config.pad_token_id,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )

        l2r_last_hidden_state = l2r_last_hidden_state[
            range(batch_size), sequence_lengths
        ]
        r2l_last_hidden_state = r2l_last_hidden_state[
            range(batch_size), sequence_lengths
        ]

        # combine hidden states
        combined_hidden_states = torch.cat(
            [l2r_last_hidden_state, r2l_last_hidden_state], dim=1
        )

        # Add pooler and dropout before classification
        pooled_output = self.pooler(combined_hidden_states)
        pooled_output = self.dropout(pooled_output)

        # get logits
        logits = self.score(pooled_output)

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


class ELMOBertL2RForSequenceClassification(BertPreTrainedModel):
    """
    Copied BertModelCausalForSequenceClassification
    Use the original BertModel as base bert model
    Modified forward and replace pooling with BertCausalPooling
    """

    def __init__(self, config):
        super().__init__(config)

        # modified here
        self.transformer = ELMOBertModel(config)
        self.bert = self.transformer.l2r_gpt
        ## end

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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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

        sequence_output = outputs.last_hidden_state
        bs, seqlen = get_sequence_lengths(
            pad_token_id=self.config.pad_token_id, input_ids=input_ids
        )
        pooled_output = self.pooler_causal(
            hidden_states=sequence_output, batch_size=bs, sequence_lengths=seqlen
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
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

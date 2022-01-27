from transformers import (
    GPT2Config,
    GPT2PreTrainedModel,
    BertConfig,
    BertPreTrainedModel,
)
import torch
from torch import nn
from .elmo_model import ELMOGPTModel, ELMOBertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from .elmo_utils import get_sequence_lengths
from .elmo_pooler import ELMOBertPooler
from newlm.lm.bert.modeling_bert.bert_model import (
    BertModelCausalForSequenceClassification,
    BertModelCausalR2LForSequenceClassification,
    BertCausalPooler
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

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

class ELMOBertForSequenceClassificationV2(BertPreTrainedModel):

    def __init__(self, config: BertConfig):
        super().__init__(config)

        self.l2r_cls = BertModelCausalForSequenceClassification(config)
        self.r2l_cls = BertModelCausalR2LForSequenceClassification(config)

        self.num_labels = config.num_labels

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

        l2r_output = self.l2r_cls(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=return_dict
        )
        r2l_output = self.r2l_cls(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=return_dict
        )

        logits = l2r_output.logits + r2l_output.logits # make sure this is element wise!

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

        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )



class ELMOBertForSequenceClassificationV3(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)

        self.transformer = ELMOBertModel(config)

        self.l2r_pooler = BertCausalPooler(config)
        self.r2l_pooler = BertCausalPooler(config)

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

        # Get elmo out
        l2r_last_hidden_state = elmo_out.last_hidden_state[0]
        r2l_last_hidden_state = elmo_out.last_hidden_state[1]

        (batch_size, sequence_lengths) = get_sequence_lengths(
            pad_token_id=self.config.pad_token_id,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )

        # Add pooler and dropout before classification
        pooled_output_l2r = self.l2r_pooler(
            hidden_states=l2r_last_hidden_state, batch_size=batch_size, sequence_lengths=sequence_lengths
        )
        pooled_output_r2l = self.r2l_pooler(
            hidden_states=r2l_last_hidden_state, batch_size=batch_size, sequence_lengths=sequence_lengths
        )
        # combine hidden states
        combined_hidden_states = torch.cat(
            [pooled_output_l2r, pooled_output_r2l], dim=1
        )
        pooled_output = self.dropout(combined_hidden_states)

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
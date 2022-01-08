from transformers import (
    GPT2Config,
    GPT2PreTrainedModel,
    GPT2Model,
    BertConfig,
    BertPreTrainedModel,
    BertModel,
)
import torch
from torch import nn
from transformers.utils import logging
from .elmo_dataclass import ElmoGPTCausalLMOutput
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from .elmo_utils import flip_tensor_by_length, get_sequence_lengths


logger = logging.get_logger(__name__)


class ELMOGPTModel(GPT2PreTrainedModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)

        self.l2r_gpt: GPT2Model = GPT2Model(config)
        self.r2l_gpt: GPT2Model = GPT2Model(config)

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
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        gpt_args = locals()
        gpt_args.pop("self")

        # l2r outs
        l2r_outs = self.l2r_gpt(**gpt_args)

        # Flip inputs
        (batch_size, sequence_lengths) = get_sequence_lengths(
            pad_token_id=self.config.pad_token_id,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )

        flip_input_ids = (
            flip_tensor_by_length(input_ids, batch_size, sequence_lengths)
            if input_ids is not None
            else None
        )

        flip_input_embeds = (
            flip_tensor_by_length(inputs_embeds, batch_size, sequence_lengths)
            if inputs_embeds is not None
            else None
        )

        r2l_input = gpt_args.copy()
        r2l_input["input_ids"] = flip_input_ids
        r2l_input["inputs_embeds"] = flip_input_embeds

        # r2l_outs
        r2l_outs = self.r2l_gpt(**r2l_input)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=(l2r_outs.last_hidden_state, r2l_outs.last_hidden_state),
            past_key_values=(l2r_outs.past_key_values, r2l_outs.past_key_values),
            hidden_states=(l2r_outs.hidden_states, r2l_outs.hidden_states),
            attentions=(l2r_outs.attentions, r2l_outs.attentions),
            cross_attentions=(l2r_outs.cross_attentions, r2l_outs.cross_attentions),
        )


class ELMOBertModel(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)

        self.l2r_gpt: BertModel = BertModel(config)
        self.r2l_gpt: BertModel = BertModel(config)

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
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        gpt_args = locals()
        gpt_args.pop("self")

        print("L2R Input")
        print(gpt_args)

        # l2r outs
        l2r_outs = self.l2r_gpt(**gpt_args)

        # Flip inputs
        (batch_size, sequence_lengths) = get_sequence_lengths(
            pad_token_id=self.config.pad_token_id,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )

        flip_input_ids = (
            flip_tensor_by_length(input_ids, batch_size, sequence_lengths)
            if input_ids is not None
            else None
        )

        flip_input_embeds = (
            flip_tensor_by_length(inputs_embeds, batch_size, sequence_lengths)
            if inputs_embeds is not None
            else None
        )

        flip_token_type_ids = (
            flip_tensor_by_length(token_type_ids, batch_size, sequence_lengths)
            if token_type_ids is not None
            else None
        )

        r2l_input = gpt_args.copy()
        r2l_input["input_ids"] = flip_input_ids
        r2l_input["inputs_embeds"] = flip_input_embeds
        r2l_input["token_type_ids"] = flip_token_type_ids

        print("R2L Input")
        print(r2l_input)


        # r2l_outs
        r2l_outs = self.r2l_gpt(**r2l_input)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=(l2r_outs.last_hidden_state, r2l_outs.last_hidden_state),
            past_key_values=(l2r_outs.past_key_values, r2l_outs.past_key_values),
            hidden_states=(l2r_outs.hidden_states, r2l_outs.hidden_states),
            attentions=(l2r_outs.attentions, r2l_outs.attentions),
            cross_attentions=(l2r_outs.cross_attentions, r2l_outs.cross_attentions),
        )

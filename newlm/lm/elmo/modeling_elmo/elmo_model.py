from transformers import GPT2PreTrainedModel, GPT2LMHeadModel
import torch
from torch import nn
from transformers.utils import logging
from .elmo_dataclass import ElmoGPTCausalLMOutput
from .elmo_config import ELMOConfig

logger = logging.get_logger(__name__)


class ELMOGPTHeadModel(GPT2PreTrainedModel):
    def __init__(self, config: ELMOConfig):
        super().__init__(config)

        self.l2r_gpt: GPT2LMHeadModel = GPT2LMHeadModel(config.l2r_gpt_config)
        self.r2l_gpt: GPT2LMHeadModel = GPT2LMHeadModel(config.r2l_gpt_config)

    def get_sequence_lengths(self, input_ids=None, inputs_embeds=None, **kwargs):
        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
                )
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        return (batch_size, sequence_lengths)

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
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> ElmoGPTCausalLMOutput:
        # NOTE input_embeds is not tested!
        l2r_outs = self.l2r_gpt(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Flip inputs
        (batch_size, sequence_lengths) = self.get_sequence_lengths(
            input_ids=input_ids, inputs_embeds=inputs_embeds
        )

        flip_input_ids = (
            self._flip_tensor_by_length(input_ids, batch_size, sequence_lengths)
            if input_ids is not None
            else None
        )

        flip_input_embeds = (
            self._flip_tensor_by_length(inputs_embeds, batch_size, sequence_lengths)
            if inputs_embeds is not None
            else None
        )

        flip_labels = (
            self._flip_tensor_by_length(labels, batch_size, sequence_lengths)
            if labels is not None
            else None
        )

        r2l_outs = self.r2l_gpt(
            flip_input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=flip_input_embeds,
            labels=flip_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # LM loss
        loss = None
        if r2l_outs.loss is not None:
            loss = l2r_outs.loss + r2l_outs.loss

        return ElmoGPTCausalLMOutput(
            loss=loss,
            l2r_last_hidden_state=l2r_outs.last_hidden_state,
            r2l_last_hidden_state=r2l_outs.last_hidden_state,
            l2r_logits=l2r_outs.logits,
            r2l_logits=l2r_outs.logits,
        )

    def _flip_tensor_by_length(tensor, batch_size, sequence_lengths):
        """
        Flip inp_ids
        """
        if tensor is None:
            return None
        flip_t = tensor.clone()
        for i in range(batch_size):
            n = sequence_lengths[i] + 1
            flip_t[i, :n] = flip_t[i, :n].flip(dims=[0])
        return flip_t

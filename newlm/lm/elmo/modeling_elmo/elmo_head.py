from transformers import (
    GPT2Config,
    GPT2PreTrainedModel,
    BertConfig,
    BertPreTrainedModel,
)
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
import torch
import copy
from torch import nn
from transformers.utils import logging
from .elmo_dataclass import ElmoGPTCausalLMOutput, Bert2TowerLMOutput
from .elmo_model import ELMOGPTModel, ELMOBertModel
from typing import Tuple
from .elmo_utils import flip_tensor_by_length, get_sequence_lengths
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

logger = logging.get_logger(__name__)


class ELMOGPTLMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)

        self.transformer = ELMOGPTModel(config)
        self.lm_head_l2r = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head_r2l = nn.Linear(config.n_embd, config.vocab_size, bias=False)

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
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> ElmoGPTCausalLMOutput:
        gpt_args = locals()
        gpt_args.pop("self")
        labels = gpt_args.pop("labels")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        (batch_size, sequence_lengths) = get_sequence_lengths(
            pad_token_id=self.config.pad_token_id,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )

        transformer_outputs = self.transformer(**gpt_args)
        last_hidden_states = transformer_outputs.last_hidden_state

        # hidden_state left to right is in index 0 of transformer_outputs
        # hidden_state right to left is in index 1 of transformer_outputs

        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            for i in range(len(hidden_states)):
                last_hidden_states[i] = last_hidden_states[i].to(
                    self.lm_head.weight.device
                )

        flip_labels = (
            flip_tensor_by_length(labels, batch_size, sequence_lengths)
            if labels is not None
            else None
        )

        l2r_loss, l2r_lm_logits = self._calculate_out_head(
            last_hidden_states, labels, r2l=False
        )

        r2l_loss, r2l_lm_logits = self._calculate_out_head(
            last_hidden_states, flip_labels, r2l=True
        )

        total_loss = l2r_loss + r2l_loss if labels is not None else None

        return ElmoGPTCausalLMOutput(
            loss=total_loss,
            l2r_hidden_states=transformer_outputs.hidden_states[0],
            r2l_hidden_states=transformer_outputs.hidden_states[1],
            l2r_logits=l2r_lm_logits,
            r2l_logits=r2l_lm_logits,
        )

    def _calculate_out_head(
        self,
        hidden_states,
        labels,
        r2l,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_state = hidden_states[1] if r2l else hidden_states[0]
        lm_head = self.lm_head_r2l if r2l else self.lm_head_l2r
        lm_logits = lm_head(hidden_state)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return loss, lm_logits


class ELMOBertLMHeadModel(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)

        self.transformer = ELMOBertModel(config)
        self.lm_head_l2r = BertOnlyMLMHead(config)  # Elmo FIX
        self.lm_head_r2l = BertOnlyMLMHead(config)

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
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> ElmoGPTCausalLMOutput:
        gpt_args = locals()
        gpt_args.pop("self")
        labels = gpt_args.pop("labels")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        (batch_size, sequence_lengths) = get_sequence_lengths(
            pad_token_id=self.config.pad_token_id,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )

        transformer_outputs = self.transformer(**gpt_args)
        last_hidden_states = transformer_outputs.last_hidden_state

        # hidden_state left to right is in index 0 of transformer_outputs
        # hidden_state right to left is in index 1 of transformer_outputs

        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            for i in range(len(hidden_states)):
                last_hidden_states[i] = last_hidden_states[i].to(
                    self.lm_head.weight.device
                )

        flip_labels = (
            flip_tensor_by_length(labels, batch_size, sequence_lengths)
            if labels is not None
            else None
        )

        l2r_loss, l2r_lm_logits = self._calculate_out_head(
            last_hidden_states, labels, r2l=False
        )

        r2l_loss, r2l_lm_logits = self._calculate_out_head(
            last_hidden_states, flip_labels, r2l=True
        )

        total_loss = l2r_loss + r2l_loss if labels is not None else None

        return ElmoGPTCausalLMOutput(
            loss=total_loss,
            l2r_hidden_states=transformer_outputs.hidden_states[0],
            r2l_hidden_states=transformer_outputs.hidden_states[1],
            l2r_logits=l2r_lm_logits,
            r2l_logits=r2l_lm_logits,
        )

    def _calculate_out_head(
        self,
        hidden_states,
        labels,
        r2l,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_state = hidden_states[1] if r2l else hidden_states[0]
        lm_head = self.lm_head_r2l if r2l else self.lm_head_l2r
        lm_logits = lm_head(hidden_state)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return loss, lm_logits


class Bert2TowerModel(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)

        self.transformer = ELMOBertModel(config)

        config2 = copy.deepcopy(config)
        config2.hidden_size = config.hidden_size * 2
        self.lm_head = BertOnlyMLMHead(config2)

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
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> ElmoGPTCausalLMOutput:
        gpt_args = locals()
        gpt_args.pop("self")
        labels = gpt_args.pop("labels")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        (batch_size, sequence_lengths) = get_sequence_lengths(
            pad_token_id=self.config.pad_token_id,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )

        transformer_outputs = self.transformer(**gpt_args)
        last_hidden_states = transformer_outputs.last_hidden_state

        # hidden_state left to right is in index 0 of transformer_outputs
        # hidden_state right to left is in index 1 of transformer_outputs

        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            for i in range(len(hidden_states)):
                last_hidden_states[i] = last_hidden_states[i].to(
                    self.lm_head.weight.device
                )

        loss, lm_logits, hidden_state = self._calculate_out_head(
            last_hidden_states, labels, batch_size, sequence_lengths
        )

        total_loss = loss if labels is not None else None

        return Bert2TowerLMOutput(
            loss=total_loss,
            hidden_states=hidden_state,
            l2r_hidden_states=transformer_outputs.hidden_states[0],
            r2l_hidden_states=transformer_outputs.hidden_states[1],
            logits=lm_logits,
        )

    def _calculate_out_head(
        self,
        hidden_states,
        labels,
        batch_size,
        sequence_lengths,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        l2r_hidden_state = hidden_states[0]
        r2l_hidden_state = hidden_states[1]
        r2l_hidden_state_flip = flip_tensor_by_length(
            r2l_hidden_state, batch_size, sequence_lengths
        )

        _, seqlen, hidden_size = l2r_hidden_state.shape

        hidden_state = torch.zeros((batch_size, seqlen, hidden_size * 2)).to(
            l2r_hidden_state.device
        )

        # L2R => [0, L2R[:-1]]
        l2r_state = torch.cat(
            [
                torch.zeros((batch_size, 1, hidden_size)).to(l2r_hidden_state.device),
                l2r_hidden_state[:, :-1, :],
            ],
            dim=1,
        )
        # R2L => [R2L[1:], 0]
        r2l_state = torch.cat(
            [
                r2l_hidden_state_flip[:, 1:, :],
                torch.zeros((batch_size, 1, hidden_size)).to(l2r_hidden_state.device),
            ],
            dim=1,
        )
        # L2R + R2L
        hidden_state = torch.cat([l2r_state, r2l_state], dim=-1)

        ## [Previous Slow Implementation]
        # # concat l2r_hidden_state and r2l_hidden_state_flip
        # for b in range(batch_size):
        #     for i in range(sequence_lengths[b]):
        #         if i == 0:
        #             hidden_state[b][i] = torch.cat(
        #                 [
        #                     torch.zeros(hidden_size).to(l2r_hidden_state.device),
        #                     r2l_hidden_state_flip[b][i + 1],
        #                 ]
        #             )
        #         elif i == sequence_lengths[b] - 1:
        #             hidden_state[b][i] = torch.cat(
        #                 [
        #                     l2r_hidden_state[b][i - 1],
        #                     torch.zeros(hidden_size).to(l2r_hidden_state.device),
        #                 ]
        #             )
        #         else:
        #             hidden_state[b][i] = torch.cat(
        #                 [l2r_hidden_state[b][i - 1], r2l_hidden_state_flip[b][i + 1]]
        #             )

        lm_logits = self.lm_head(hidden_state)

        loss = None
        if labels is not None:
            ### No need to shift anything!
            # shift_logits = lm_logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)
            )

        return loss, lm_logits, hidden_state

from transformers import BertLMHeadModel
from newlm.lm.elmo.modeling_elmo.elmo_utils import flip_tensor_by_length, get_sequence_lengths


class BertLMHeadR2LModel(BertLMHeadModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        **bert_args,
    ):

        input_ids = bert_args["input_ids"]
        inputs_embeds = bert_args["inputs_embeds"]
        labels = bert_args["labels"]

        # Flip inputs
        (batch_size, sequence_lengths) = get_sequence_lengths(
            pad_token_id=self.config.pad_token_id,
            input_ids=input_ids,
        )

        flip_input_ids = (
            flip_tensor_by_length(input_ids, batch_size, sequence_lengths)
            if input_ids is not None
            else None
        )

        flip_inputs_embeds = (
            flip_tensor_by_length(inputs_embeds, batch_size, sequence_lengths)
            if inputs_embeds is not None
            else None
        )

        flip_labels = (
            flip_tensor_by_length(labels, batch_size, sequence_lengths)
            if labels is not None
            else None
        )

        bert_args["input_ids"] = flip_input_ids
        bert_args["inputs_embeds"] = flip_inputs_embeds
        bert_args["labels"] = flip_labels

        return super().forward(**bert_args)

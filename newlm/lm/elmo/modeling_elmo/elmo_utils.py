import torch


def flip_tensor_by_length(tensor, batch_size, sequence_lengths):
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


def get_sequence_lengths(pad_token_id, input_ids=None, inputs_embeds=None, **kwargs):
    if input_ids is not None:
        batch_size, sequence_length = input_ids.shape[:2]
    else:
        batch_size, sequence_length = inputs_embeds.shape[:2]

    assert (
        pad_token_id is not None or batch_size == 1
    ), "Cannot handle batch sizes > 1 if no padding token is defined."
    if pad_token_id is None:
        sequence_lengths = -1
    else:
        if input_ids is not None:
            sequence_lengths = torch.ne(input_ids, pad_token_id).sum(-1) - 1
        else:
            sequence_lengths = -1
            logger.warning(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

    return (batch_size, sequence_lengths)

from transformers.configuration_utils import PretrainedConfig
from transformers import GPT2Config


class ELMOConfig(PretrainedConfig):

    model_type = "gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        l2r_gpt_config: GPT2Config
        r2l_gpt_config: GPT2Config

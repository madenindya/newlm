from transformers.configuration_utils import PretrainedConfig
from transformers import GPT2Config


class ELMOConfig(PretrainedConfig):

    model_type = "gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self, lr2_gpt_config: GPT2Config, r2l_gpt_config: GPT2Config, **kwargs
    ):
        super().__init__(
            lr2_gpt_config=lr2_gpt_config, r2l_gpt_config=r2l_gpt_config, **kwargs
        )
        self.pad_token_id: int = kwargs.get("pad_token_id", None)
        self.l2r_gpt_config: GPT2Config = lr2_gpt_config
        self.r2l_gpt_config: GPT2Config = r2l_gpt_config

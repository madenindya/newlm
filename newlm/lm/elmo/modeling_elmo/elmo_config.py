from dataclasses import dataclass
from transformers import GPT2Config

class ELMOConfig(GPT2Config):

    shared: bool = False

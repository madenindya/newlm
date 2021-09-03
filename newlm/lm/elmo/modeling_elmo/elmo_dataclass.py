from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


@dataclass
class ElmoGPTCausalLMOutput(CausalLMOutputWithCrossAttentions):

    l2r_last_hidden_state: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    r2l_last_hidden_state: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

    l2r_logits: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    r2l_logits: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

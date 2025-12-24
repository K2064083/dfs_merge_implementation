from transformers.models.llama.configuration_llama import LlamaConfig


class EvoLlamaConfig(LlamaConfig):
    model_type = "evollama"
    
    def __init__(self, num_hops: int = 64, **kwargs):
        self.num_hops = num_hops
        super().__init__(**kwargs)

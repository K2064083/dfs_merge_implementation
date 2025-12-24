from transformers.models.mistral.configuration_mistral import MistralConfig


class EvoMistralAffineConfig(MistralConfig):
    model_type = "evomistralaffine"
    
    def __init__(self, num_hops: int = 64, **kwargs):
        self.num_hops = num_hops
        super().__init__(**kwargs)

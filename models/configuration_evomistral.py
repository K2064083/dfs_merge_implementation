from transformers.models.mistral.configuration_mistral import MistralConfig


class EvoMistralConfig(MistralConfig):
    model_type = "evomistral"
    
    def __init__(self, num_hops: int = 64, **kwargs):
        self.num_hops = num_hops
        super().__init__(**kwargs)

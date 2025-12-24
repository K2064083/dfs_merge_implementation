# model_registry.py
from .modeling_evomistral import EvoMistralForCausalLM, EvoMistralForSequenceClassification
from .configuration_evomistral import EvoMistralConfig
from .modeling_evollama import EvoLlamaForCausalLM, EvoLlamaForSequenceClassification
from .configuration_evollama import EvoLlamaConfig

# for affine transform model
from .modeling_evomistralaffine import EvoMistralAffineForCausalLM
from .configuration_evomistralaffine import EvoMistralAffineConfig
from .modeling_evollamaaffine import EvoLlamaAffineForCausalLM, EvoLlamaAffineForSequenceClassification
from .configuration_evollamaaffine import EvoLlamaAffineConfig

MODEL_REGISTRY = {
    "LlamaForCausalLM": {
        "model_class": EvoLlamaAffineForCausalLM,
        "config_class": EvoLlamaAffineConfig,
    },
    "MistralForCausalLM": {
        "model_class": EvoMistralAffineForCausalLM,
        "config_class": EvoMistralAffineConfig,
        "auto_map": {
            "AutoConfig": "SakanaAI/EvoLLM-v1-JP-10B--configuration_evomistral.EvoMistralConfig",
            "AutoModelForCausalLM": "SakanaAI/EvoLLM-v1-JP-10B--modeling_evomistral.EvoMistralForCausalLM"
        }
    },
    "LlamaForSequenceClassification": {
        "model_class": EvoLlamaForSequenceClassification,
        "config_class": EvoLlamaConfig,
    },
    "MistralForSequenceClassification": {
        "model_class": EvoMistralForSequenceClassification,
        "config_class": EvoMistralConfig,
    },
}

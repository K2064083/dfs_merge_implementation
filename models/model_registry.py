# model_registry.py
from .modeling_evomistral import EvoMistralForCausalLM, EvoMistralForSequenceClassification
from .configuration_evomistral import EvoMistralConfig
from .modeling_evollama import EvoLlamaForCausalLM, EvoLlamaForSequenceClassification
from .configuration_evollama import EvoLlamaConfig
from .modeling_evomistralaffine import EvoMistralAffineForCausalLM
from .configuration_evomistralaffine import EvoMistralAffineConfig

MODEL_REGISTRY = {
    "LlamaForCausalLM": {
        "model_class": EvoLlamaForCausalLM,
        "config_class": EvoLlamaConfig,
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

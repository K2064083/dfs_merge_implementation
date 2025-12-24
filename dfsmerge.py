"""
DataFlowSpaceMerge Module

This module provides an implementation of the DataFlowSpaceMerge algorithm.
Each models must have the same architecture.
This implementation only supports Mistral architectures.
Support for other architectures requires adding a model wrapper to `modeling.py`.

Version History:
v1.1.0: Added support for revision IDs.
v1.2.0: Added support for scaling factors.
v1.2.2: Modify load_state_dict args "assign=True"
v1.2.3: Modify by hirose
v1.4.0: Add executor from config
v2.0.0: Output with modeling_evomistral.py(Sakana AI)
v2.0.1: Add last_layer_scale
v2.0.2: Add save_only_config
v3.0.0: Compatible with Llama models
v3.0.1: Changed to use input model config instead of base model
v3.0.2: Support for SequenceClassification
v3.1.0: Seve with bfloat16, add Autoconfig, use_cache:False
v3.1.1: Add args 'add_last_layer'

References:
Akiba, Takuya, et al. "Evolutionary optimization of model merging recipes."
arXiv preprint arXiv:2403.13187 (2024).
https://arxiv.org/abs/2403.13187

Author: kudo
"""

__version__ = "3.0.2"

import re

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .models.model_registry import MODEL_REGISTRY

@torch.no_grad()
def data_flow_space_layer_merge(
    model_paths: list[tuple[str, str]],
    num_r: int,
    layer_mask: list[bool],
    scaling_factors: list[float],
    bias_factors: list[float],
    add_last_layer: bool | None=True,
    last_layer_scale: float | None = 1.0,
    input_layer_model: int | None = 0,
    output_layer_model: int | None= 0,
    output_path: str | None = None,
    save_only_config: bool | None= False,
    device: str | None = "cpu",
    cache_dir: str | None = None,
    classification: bool | None = False,
    debug: bool | None = False,
) -> torch.nn.Module:
    """
    Merges multiple models according to the layer mask.

    Args:
        model_paths (list): A list of (model_name, revision_id) tuples.
        num_r (int): Number of times to repeat the layer cycle
        layer_mask (list): Layer mask, a list of shape (num_r, 2, num_layers)
        scaling_factors (list): Scaling W same shape with layer_mask
        add_last_layer (bool): If true, the final decoder layer of the output_layer_model is added to the end.
        last_layer_scale (float): last_layer scale variable
        input_model (int): Model index to use for the input layer. Defaults to the first model.
        output_model(int): Model index to use for the output layer. Defaults to the first model.
        output_path (bool, optional): Path to save the output model. If None, the model is not saved.
        device (str, optional):  Device to use for merging. Specify a CUDA device (e.g., "cuda:0") for GPU merging.
        cache_dir (str, optional):  Directory to cache models.  Defaults to the standard Hugging Face cache directory.
        classification (bool, optional): For classification model. Add score.weight.
        debug (bool, optional): Enable more verbose logging. Defaults to False.

    Returns:
        nn.Module: Merged model
    """
    # Load models
    if debug:
        print("loading models...")
    models = []
    model_configs = []
    model_architecture = ""
    for model_path in model_paths:
        config = AutoConfig.from_pretrained(
            model_path[0], revision=model_path[1], cache_dir=cache_dir
        )
        model_configs.append(config)
        AutoModel_class = AutoModelForSequenceClassification if classification else AutoModelForCausalLM
        model = AutoModel_class.from_pretrained(
            model_path[0],
            revision=model_path[1],
            cache_dir=cache_dir,
            device_map=device,
        )
            
        models.append(model)

    # Check Archtecture
    model_architecture = model_configs[input_layer_model].architectures[0]
    if model_architecture not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model architecture: {model_architecture}")

    # Check input
    assert (
        np.array(layer_mask).shape
        == (num_r, len(model_paths), model_configs[0].num_hidden_layers)
    ), f"Invalid input shapes.  Layer_mask must have shape ({num_r}, {len(models)}, num_layers)"
    # assert (
    #     np.array(scaling_factors).shape == np.array(layer_mask).shape
    # ), f"Invalid input shapes. Scaling_factors must have same shape as Layer_mask"
    assert (
        input_layer_model == 0 or input_layer_model == 1
    ), f"input_layer_model must be 0 or 1, but got {input_layer_model}."
    assert (
        output_layer_model == 0 or output_layer_model == 1
    ), f"output_layer_model must be 0 or 1, but got {output_layer_model}."

    # make merge recipes
    used_layers=[]
    used_indices=[]
    for n in range(num_r):
        for m in range(len(models)):
            for l in range(model_configs[0].num_hidden_layers):
                if layer_mask[n][m][l]:
                    if (m,l) not in used_layers:
                        used_layers.append((m,l))
                    used_indices.append(used_layers.index((m,l)))
    # append last layers
    if add_last_layer:
        m = output_layer_model
        l = model_configs[0].num_hidden_layers - 1 
        if (m,l) not in used_layers:
            used_layers.append((m,l))
        used_indices.append(used_layers.index((m,l)))
        if debug:
            print("use layers (model_num, layer_num)",used_layers)
            print("use indices", used_indices)

    # Get param names
    param_names = {"embedding_layer_params": [], "output_layer_params": []}
    for i in range(model_configs[0].num_hidden_layers):
        param_names[f"layer_{i}_params"] = []
    num_layer = -1
    for i, [param_name, _] in enumerate(models[0].named_parameters()):
        assert (
            param_name == list(models[1].named_parameters())[i][0]
        ), f"The model uses different parameter names, such as {param_name}and{list(models[1].named_parameters())[i][0]}"
        if num_layer == -1:
            if "model.layers" not in param_name:
                param_names["embedding_layer_params"].append(param_name)
            else:
                num_layer += 1
        if num_layer >= 0:
            if "model.layers" in param_name:
                num_layer = int(re.search(r"layers\.(\d+)", param_name).group(1))
                param_names[f"layer_{num_layer}_params"].append(param_name)
            else:
                param_names["output_layer_params"].append(param_name)

    # For llama. In particular, lm_head cannot be read with llama.
    if classification==False and "lm_head.weight" not in param_names["output_layer_params"]:
       param_names["output_layer_params"].append("lm_head.weight")

    if debug:
        for k, v in param_names.items():
            print(f"{k}:\n{v}")

    # convert to state_dict
    if debug:
        print("convert to state_dict")
    models = [model.state_dict() for model in models]
    merged_state_dict = {}

    # Copy embedding layer
    for name in param_names["embedding_layer_params"]:
        merged_state_dict[name] = models[input_layer_model][name]
        if debug:
            print(f"Copying {name} from model{input_layer_model} size:{merged_state_dict[name].shape}")

    # Copy decoder layer
    for i, layer in enumerate(used_layers):
        layer_prefix = f"model.layers.{layer[1]}."
        for name in param_names[f"layer_{layer[1]}_params"]:
            new_name = name.replace(str(layer[1]), str(i))
            merged_state_dict[new_name] = models[layer[0]][name].to(device)
            if debug:
                print(f"Copying {name} from model{layer[0]} size:{merged_state_dict[name].shape}")

    # Copy output layers (head)
    for name in param_names["output_layer_params"]:
        merged_state_dict[name] = models[output_layer_model][name].to(device)
        if debug:
            print(f"Copying {name} from model{output_layer_model} size:{merged_state_dict[name].shape}")

    # scaling factors
    merged_state_dict["model.input_layers"] = torch.Tensor(used_indices) 
    if add_last_layer:# 要修正last_layer_scaleに未対応
        input_scales = np.append(np.array(scaling_factors)[np.array(layer_mask)], last_layer_scale)
        input_bias = np.append(np.array(bias_factors)[np.array(layer_mask)], last_layer_scale)
    else:
        input_scales = np.array(scaling_factors)[np.array(layer_mask)]
        input_bias = np.array(bias_factors)[np.array(layer_mask)]
    merged_state_dict["model.input_scales"] = torch.Tensor(input_scales)
    merged_state_dict["model.input_bias"] = torch.Tensor(input_bias)

    # load evo-merge model
    model_info = MODEL_REGISTRY[model_architecture]
    config_class = model_info["config_class"]
    model_class = model_info["model_class"]
    architecture = model_class.__name__
    
    model_config = config_class(
        num_hops=len(used_indices),
        num_hidden_layers = len(used_layers),
        architectures = [architecture],
        sliding_window = 2048,
    )
    base_config_dict = model_configs[0].to_dict()
    base_config_dict.pop('architectures')
    base_config_dict.pop('num_hidden_layers')
    model_config.update(base_config_dict)
    model_config.update({'use_cache':False})

    if 'auto_map' in model_info.keys():
        model_config.update({'auto_map':model_info['auto_map']})

    output_model = model_class(model_config)

    if debug:
        print(model_config)
        print(f"num_hiddenn_layers:{model_config.num_hidden_layers}")
        print("Creating merged model...")

    # Create the merged model
    if debug:print("loading state dict...")
    output_model.load_state_dict(merged_state_dict)
    output_model.to(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(
            model_paths[input_layer_model][0],
            revision=model_paths[input_layer_model][1],
        )

    # Save the model
    if output_path:
        if save_only_config:
            if debug:print("Savinge config...")
            model_config.save_pretrained(output_path)
        else:
            if debug:print("Saving model...")
            output_model.save_pretrained(output_path)
            if debug:print("copying tokenizer")
            tokenizer.save_pretrained(output_path)

    return output_model, tokenizer

def dfs_excute_from_config(merge_config, **kwargs) -> torch.nn.Module:
    with open(merge_config, 'r') as f:
        config = yaml.load(f,Loader=yaml.Loader)

    output_model, tokenizer = data_flow_space_layer_merge(
        model_paths = config['model_paths'],
        num_r = config['num_r'],
        layer_mask = config['layer_mask'],
        scaling_factors = config['scaling_factors'],
        bias_factors = config['bias_factors'],
        input_layer_model = config.get('input_layer_model', 0),
        output_layer_model = config.get('output_layer_model', 0),
        **kwargs
    )

    return output_model, tokenizer

def main():
    import time
    import os
    model_path = ".cache/super_model/"

    start = time.time()
    
    os.makedirs(model_path, exist_ok=True)
    D = 4096
    base_config = {
        'model_paths': [
            ('SakanaAI/EvoLLM-JP-v1-7B', None),
            ('augmxnt/shisa-gamma-7b-v1', None)
        ],
        'input_layer_model': 0,
        'output_layer_model': 0,
        'num_r': 1,
        'layer_mask': [[[True] * 32, [True] * 32],],
        'scaling_factors': [[[np.eye(D).tolist()] * 32, [np.eye(D).tolist()] * 32],],
        'bias_factors':[[[[0.0]*D, [0.0]*D]]]
    }

    merge_config_path = os.path.join(model_path, 'merge_config.yml')
    with open(merge_config_path, 'w') as f:
        yaml.dump(base_config, f, indent=2, default_flow_style=False)
        
    dfs_excute_from_config(
        merge_config=merge_config_path,
        output_path=model_path,
        add_last_layer=False,
    )

    end = time.time()
    time_diff = end - start
    print("実行時間", time_diff)

    prompt = "my_list = [1, 2, 3, 4, 5]\nprint(my_list)\n >>>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids,max_length=200,pad_token_id=tokenizer.eos_token_id)
    generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated:\n", generated_code)


if __name__ == "__main__":
    main()

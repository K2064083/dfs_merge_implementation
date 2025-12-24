# DataFlowSpaceMerge Implementation

This repository provides an implementation of the **DataFlowSpaceMerge** algorithm, designed to merge multiple Large Language Models (LLMs) with the same architecture into a single, more capable model. This implementation is inspired by the evolutionary optimization techniques for model merging developed by Sakana AI.

## Features

* **Multi-Architecture Support**: Compatible with **Mistral** and **Llama** architectures for both `CausalLM` and `SequenceClassification` tasks.
* **Layer-Wise Merging**: Allows precise control over model merging through layer masks, scaling factors, and bias factors.
* **Flexible Configuration**: Supports execution via YAML configuration files for reproducible merge recipes.
* **Evolutionary Merge Ready**: Designed to work with the output of evolutionary optimization processes.
* **Bfloat16 Precision**: Automatically saves merged models in `bfloat16` for efficient inference.

## Requirements

The project requires a Python 3.10 environment. You can set up the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate llm2

```

Key dependencies include:

* `torch` (>= 2.1.2)
* `transformers` (>= 4.46.2)
* `evomerge`
* `pyyaml`
* `numpy`

## Usage

### 1. Prepare a Merge Configuration

Create a YAML file (e.g., `merge_config.yml`) defining the models to merge and the specific merge parameters. Below is a simplified example based on the project's structure:

```yaml
model_paths:
  - ["SakanaAI/EvoLLM-JP-v1-7B", null]
  - ["augmxnt/shisa-gamma-7b-v1", null]
num_r: 1
layer_mask:
  - - [true, true, ..., true] # Layers from model 0
    - [false, false, ..., false] # Layers from model 1
scaling_factors: [...]
bias_factors: [...]
input_layer_model: 0
output_layer_model: 0

```

### 2. Execute the Merge

You can run the merge process by calling the `dfs_excute_from_config` function within `dfsmerge.py`:

```python
from dfsmerge import dfs_excute_from_config

model, tokenizer = dfs_excute_from_config(
    merge_config='path/to/merge_config.yml',
    output_path='./merged_model',
    device='cuda' # or 'cpu'
)

```

## Supported Architectures

The implementation uses a `MODEL_REGISTRY` to manage different model classes:

* `LlamaForCausalLM`
* `MistralForCausalLM`
* `LlamaForSequenceClassification`
* `MistralForSequenceClassification`

Custom model wrappers for evolutionary merging (e.g., `EvoLlamaAffineForCausalLM`) are used to handle the specific requirements of the DataFlowSpace algorithm.

## References

If you use this implementation, please cite the original research by Sakana AI:

* Akiba, Takuya, et al. "Evolutionary optimization of model merging recipes." arXiv preprint arXiv:2403.13187 (2024). [https://arxiv.org/abs/2403.13187](https://arxiv.org/abs/2403.13187).

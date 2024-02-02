# Sigma MoE 🤗
This is a huggingface-like implementation of the Sigma-MoE architecture proposed by [Csordas et al.](https://aclanthology.org/2023.findings-emnlp.49/).
All of the features of huggingface are supported and we have pre-trained models available in the HF organization of the Analog In-Memory Computing Group of IBM Research -- Zurich, which can be found [here](https://huggingface.co/ibm-aimc).

## Requirements
The Sigma-MoE layer also has a CPU implementation. If you are running it on a GPU, you need to have one which is at least Volta (V100, A100, H100) since this package leverages triton.
## Getting started 🚀
You can create a clean environment using the following
```
conda create -n torch-nightly python=3.10 -y
conda activate torch-nightly
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
pip install triton transformers datasets
```

Then, install this package using `pip install -e .`

## Usage ⚒️
Analogous to Huggingface, we have various models capable of, for example, language modelling. The models are:
```
SigmaMoEConfiguration
SigmaMoEModel
SigmaMoEPreTrainedModel
SigmaMoEForCausalLM
SigmaMoEForSequenceClassification
SigmaMoEForTokenClassification
```
and can be imported using
```
from sigma_moe import SigmaMoEPreTrainedModel
```
For an example, see `example.py`.

## Note on `torch.compile`
This layer supports `torch.compile`.

## License
```
Copyright 2023/2024 IBM. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
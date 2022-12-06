#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import deepspeed
import logging
import math
import os
import sys
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

torch.manual_seed(1234)
from language_modeling_via_stochastic_processes.src.models import language
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
import datasets
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    #default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
import tokenizers
from tokenizers import Tokenizer
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers import AutoModelForCausalLM, GPT2Tokenizer, AutoConfig, PreTrainedTokenizerFast

from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping
InputDataClass = NewType("InputDataClass", Any)
# Data collator for batching diffusion related stuff
def diffusion_torch_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    import torch

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                if k == 'chunk_input_ids':
                    #import pdb; pdb.set_trace()
                    max_num_chunks = max([len(f[k]) for f in features])
                    blank = [69] * len(features[0][k][0])
                    batch[k] = torch.tensor([f[k] + [blank,]*(max_num_chunks-len(f[k])) for f in features])
                elif k == 'chunk_attn_mask':
                    #import pdb; pdb.set_trace()
                    max_num_chunks = max([len(f[k]) for f in features])
                    blank = [1] * len(features[0][k][0])
                    batch[k] = torch.tensor([f[k] + [blank,]*(max_num_chunks-len(f[k])) for f in features])
                else:
                    batch[k] = torch.tensor([f[k] for f in features])

    return batch


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.23.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    cl_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    latent_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    latent_dim=32
    proj_model = nn.Linear(latent_dim, 8192)
    tokenizer_path = '/lus/eagle/projects/CVD-Mol-AI/for_yuntian/codon_wordlevel_100vocab_added.json'
    print (f'Loading tokenizer from {tokenizer_path}')
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(tokenizer_path))
    cl_model_name_or_path = '/lus/eagle/projects/CVD-Mol-AI/yuntian/genomenewnaive/encoder_93810/run_l0.001_b32/checkpoints'
    cl_model = language.GPT2OUEncoder(
         hidden_dim=128,
         latent_dim=latent_dim,
         finetune_gpt2=False, single_layer=False, genome=True, tokenizer=tokenizer)
    if not os.path.isfile(cl_model_name_or_path):
        import glob
        filename = glob.glob(os.path.join(cl_model_name_or_path, '*pt'))[0]
        cl_model_name_or_path = filename
    print (f'Loading CL Model from {cl_model_name_or_path}')
    
    state_dict = torch.load(cl_model_name_or_path)
    new_dict = {}
    for k, v in state_dict['state_dict'].items():
        if any([i in k for i in ['model.model.g_ar', 'model.model.W_k']]):
            new_dict[k[6:]] = v
        elif any([i in k for i in ['model.g_ar', 'model.W_k', 'time_model']]):
            continue
        elif "model." in k:
            new_dict[k[6:]] = v
        else:
            if 'opt_mlp' in k or k == 'sigma' or k == 'log_sigma':
                continue
            new_dict[k] = v
    if any(['g_ar' in k for k in new_dict.keys()]):
        cl_model.g_ar = nn.GRU(input_size=latent_dim,
                           hidden_size=2400, # default number in infoNCE for langauge
                           num_layers=3,
                           batch_first=True
                           )
        cl_model.W_k = nn.Linear(2400, latent_dim)
    elif any(['time_model' in k for k in state_dict['state_dict'].keys()]):
        cl_model.fc_mu = nn.Linear(latent_dim, latent_dim)
        cl_model.fc_var = nn.Linear(latent_dim, latent_dim)
    
    #import pdb; pdb.set_trace()
    cl_model.load_state_dict(new_dict)
    for p in cl_model.parameters():
        p.requires_grad = False
    cl_model.eval()
    print ('*'*10)
    for p in cl_model.parameters():
        print (p.device)
        break
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    #send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    #if model_args.tokenizer_name:
    #    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    #elif model_args.model_name_or_path:
    #    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    #else:
    #    raise ValueError(
    #        "You are instantiating a new tokenizer from scratch. This is not supported by this script."
    #        "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    #    )
    ###tokenizer_path = 'gene_transformer/gene_transformer/tokenizer_files/codon_wordlevel_100vocab.json'
    tokenizer_path = '/lus/eagle/projects/CVD-Mol-AI/for_yuntian/codon_wordlevel_100vocab_added.json'
    print (f'Loading tokenizer from {tokenizer_path}')

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(tokenizer_path))
    #### Add special tokens <|endoftext|>
    ###special_tokens = ['<|endoftext|>']
    ###vocab = ['A', 'T', 'C', 'G']
    ###for c in vocab:
    ###    special_tokens.append(c)
    ###for c1 in vocab:
    ###    for c2 in vocab:
    ###        special_tokens.append(c1+c2)

    ###print (f'Adding special tokens {special_tokens}')
    ###print (f'Old vocab size: {len(tokenizer)}')
    ####import pdb; pdb.set_trace()
    ###special_tokens_added = []
    ###for token in special_tokens:
    ###    token = tokenizers.AddedToken(token, single_word=True)
    ###    special_tokens_added.append(token)
    ####import pdb; pdb.set_trace()
    ###tokenizer.add_tokens(special_tokens_added)
    ###tokenizer._tokenizer.save('codon_wordlevel_100vocab_added.json')
    print (f'New vocab size: {len(tokenizer)}')
    #if model_args.model_name_or_path:
    #    model = AutoModelForCausalLM.from_pretrained(
    #        model_args.model_name_or_path,
    #        from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #        config=config,
    #        cache_dir=model_args.cache_dir,
    #        revision=model_args.model_revision,
    #        use_auth_token=True if model_args.use_auth_token else None,
    #    )
    #else:
    #    model = AutoModelForCausalLM.from_config(config)
    #    n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    #    logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    #base_config = AutoConfig.from_pretrained('/lus/eagle/projects/CVD-Mol-AI/for_yuntian/patric_2.5B_pretraining/config/neox_2,533,931,008.json')
    #base_config = AutoConfig.from_pretrained('/lus/eagle/projects/CVD-Mol-AI/for_yuntian/patric_25B_pretraining/neox_25B.json')
    base_config = AutoConfig.from_pretrained('/lus/eagle/projects/CVD-Mol-AI/yuntian/sarscov2_filtered/sarscov2_filtered/transformers_deepspeed/neox_25B.json')
    #model = AutoModelForCausalLM.from_config(base_config)
    #print (model)
    #model = AutoModelForCausalLM.from_pretrained('/lus/eagle/projects/CVD-Mol-AI/yuntian/sarscov2_filtered/sarscov2_filtered/transformers_diffusion_deepspeed/model-epoch00-val_loss0.70-v2.pt', config=base_config)
    model = AutoModelForCausalLM.from_pretrained('/lus/eagle/projects/CVD-Mol-AI/yuntian/sarscov2_filtered/sarscov2_filtered/transformers_deepspeed/foundation_25B_sharded/', config=base_config)
    #import pdb; pdb.set_trace()
    #model.cuda()
    n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    device = None
    print ('*'*10)
    for p in model.parameters():
        device = p.device
        print (p.device)
        break
    cl_model = cl_model.half().to(device)
    #a = torch.load('../for_yuntian/patric_25M_pretraining/checkpoints/pytorch_model.bin')
    #a = torch.load('/lus/grand/projects/RL-fold/for_yuntian/genome_finetuning_25m/model-epoch88-val_loss0.01.pt')['state_dict']
    #a = torch.load('/lus/eagle/projects/CVD-Mol-AI/for_yuntian/genome_finetuning_25m/model-epoch88-val_loss0.01.pt')['state_dict']
    #a = torch.load('/lus/eagle/projects/CVD-Mol-AI/for_yuntian/patric_25B_pretraining/model-epoch00-val_loss0.70-v2.pt')['state_dict']
    #for i in range(1, 100):
    #    key = f"model.gpt_neox.layers.{i}.attention.bias"
    #    if key in a:
    #        del a[key]
    #    #    if key in ["buffer_names"]:
    #    #        buffer_ind = cp["buffer_names"].index(key)
    #    #        del cp["buffer_names"][buffer_ind]
    #    #else:
    #    #   break
    #b = {}
    #for k in a:
    #    #b[k.replace('module.model.', '')] = a[k]
    #    b[k.replace('model.', '')] = a[k]
    ##import pdb; pdb.set_trace()
    #model.load_state_dict(b)

    model.resize_token_embeddings(len(tokenizer))
    ### Load cl model
    #Dec6 from language_modeling_via_stochastic_processes.src.models import language
    #Dec6 cl_model = language.GPT2OUEncoder(
    #Dec6      hidden_dim=128,
    #Dec6      latent_dim=model_args.latent_dim,
    #Dec6      finetune_gpt2=False, single_layer=False, genome=True, tokenizer=tokenizer)
    #Dec6 if not os.path.isfile(model_args.cl_model_name_or_path):
    #Dec6     import glob
    #Dec6     filename = glob.glob(os.path.join(model_args.cl_model_name_or_path, '*pt'))[0]
    #Dec6     model_args.cl_model_name_or_path = filename
    #Dec6 print (f'Loading CL Model from {model_args.cl_model_name_or_path}')

    #Dec6 state_dict = torch.load(model_args.cl_model_name_or_path)
    #Dec6 new_dict = {}
    #Dec6 for k, v in state_dict['state_dict'].items():
    #Dec6     if any([i in k for i in ['model.model.g_ar', 'model.model.W_k']]):
    #Dec6         new_dict[k[6:]] = v
    #Dec6     elif any([i in k for i in ['model.g_ar', 'model.W_k', 'time_model']]):
    #Dec6         continue
    #Dec6     elif "model." in k:
    #Dec6         new_dict[k[6:]] = v
    #Dec6     else:
    #Dec6         if 'opt_mlp' in k or k == 'sigma' or k == 'log_sigma':
    #Dec6             continue
    #Dec6         new_dict[k] = v
    #Dec6 if any(['g_ar' in k for k in new_dict.keys()]):
    #Dec6     cl_model.g_ar = nn.GRU(input_size=latent_dim,
    #Dec6                        hidden_size=2400, # default number in infoNCE for langauge
    #Dec6                        num_layers=3,
    #Dec6                        batch_first=True
    #Dec6                        )
    #Dec6     cl_model.W_k = nn.Linear(2400, latent_dim)
    #Dec6 elif any(['time_model' in k for k in state_dict['state_dict'].keys()]):
    #Dec6     cl_model.fc_mu = nn.Linear(latent_dim, latent_dim)
    #Dec6     cl_model.fc_var = nn.Linear(latent_dim, latent_dim)

    #Dec6 #import pdb; pdb.set_trace()
    #Dec6 cl_model.load_state_dict(new_dict)
    #Dec6 for p in cl_model.parameters():
    #Dec6     p.requires_grad = False
    #Dec6 cl_model.eval()
    #Dec6 model.cl_model = cl_model
    #Dec6 for n, p in model.named_parameters():
    #Dec6     if 'cl_model' in n:
    #Dec6         p.requires_grad = False
    #model.proj_model = proj_model.half()
    model.cl_model = cl_model
    #with deepspeed.zero.Init():
    #proj_model = nn.Linear(model_args.latent_dim, model.gpt_neox.embed_in.weight.shape[1]).half().to(device)
    proj_model = nn.Linear(model_args.latent_dim, 8192).half().to(device)
    model.proj_model = proj_model

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            #import pdb; pdb.set_trace()
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return output
    #import pdb; pdb.set_trace()

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    #import pdb; pdb.set_trace()

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    print ('-----')
    print (block_size)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        SEP = 2
        # Concatenate all texts.
        input_ids_list = examples['input_ids']
        #import pdb; pdb.set_trace()
        offset = 0
        global_chunks = []
        chunk_ids_list = []
        for input_ids in input_ids_list:
            input_ids = torch.LongTensor(input_ids)
            #input_ids = torch.LongTensor([0,] + input_ids[:-1])
            sep_mask = input_ids.eq(SEP)
            chunk_ids = torch.cumsum(sep_mask.int(), 0) + offset
            chunk_ids_list.append(chunk_ids.tolist())
            sep_ids = sep_mask.nonzero().view(-1)
            sep_ids = sep_ids.tolist()
            prev_sep_id = 0
            for sep_id in sep_ids + [len(input_ids)-1,]:
                global_chunks.append(input_ids[prev_sep_id:sep_id+1].tolist())
                prev_sep_id = sep_id + 1
            offset = chunk_ids[-1].item() + 1
            assert offset == len(global_chunks)
        examples['chunk_ids'] = chunk_ids_list

            
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        #import pdb; pdb.set_trace()
        pad_id = 69
        pad_size = 514
        #max_num_chunks = 5
        blank_chunk = [pad_id] * pad_size
        chunk_input_ids_list = []
        chunk_ids_list = []
        chunk_attn_mask_list = []
        for i in range(len(result['input_ids'])):
            #import pdb; pdb.set_trace()
            chunk_ids = result['chunk_ids'][i]
            chunk_ids_set = sorted(list(set(chunk_ids)))
            max_num_chunks = len(chunk_ids_set)
            #if len(chunk_ids_set) > max_num_chunks:
            #    import pdb; pdb.set_trace()
            assert len(chunk_ids_set) <= max_num_chunks, len(chunk_ids_set)
            chunk_input_ids = []
            chunk_attn_masks = []
            #mapping = {}
            #for e, c_id in enumerate(chunk_ids_set):
            #    mapping[c_id] = e
            base = chunk_ids[0]
            chunk_ids_new = [item-base for item in chunk_ids]
            chunk_ids_list.append(chunk_ids_new)
            for j in range(max_num_chunks):
                if j < len(chunk_ids_set):
                    chunk_id = chunk_ids_set[j]
                    chunk = global_chunks[chunk_id]
                    chunk_attn_mask = [1] * len(chunk) + [0] * (pad_size - len(chunk))
                    chunk = chunk + [pad_id] * (pad_size - len(chunk))
                else:
                    assert False
                    chunk = blank_chunk
                    chunk_attn_mask = [0] * len(chunk)
                chunk_input_ids.append(chunk)
                chunk_attn_masks.append(chunk_attn_mask)
            chunk_input_ids_list.append(chunk_input_ids)
            chunk_attn_mask_list.append(chunk_attn_masks)
        result['chunk_ids'] = chunk_ids_list
        result['chunk_input_ids'] = chunk_input_ids_list
        result['chunk_attn_mask'] = chunk_attn_mask_list
        #import pdb; pdb.set_trace()
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            #load_from_cache_file=False,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = load_metric("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=diffusion_torch_default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
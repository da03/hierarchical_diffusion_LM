import os
import sys
import json
import argparse
import tqdm
import random
import numpy as np

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, GPT2Tokenizer, AutoConfig, PreTrainedTokenizerFast
from tokenizers import Tokenizer

from language_modeling_via_stochastic_processes.src.models import language
 
parser = argparse.ArgumentParser(description='Sample from the language models.')
parser.add_argument('--tokenizer_file', type=str, default='/lus/eagle/projects/CVD-Mol-AI/for_yuntian/codon_wordlevel_100vocab_added.json',
                    help='Tokenizer file, note that this should contain the added words of single- and double-letters, and start-of-sequence/end-of-sequence symbol <|endoftext|>.')
parser.add_argument('--high_level_sample_file', type=str, default='/lus/eagle/projects/CVD-Mol-AI/yuntian/sarscov2_filtered/sarscov2_filtered/transformers_diffusion/samples.pt',
                    help='High level sample file. Can be either oracle or diffusion model generations.')
parser.add_argument('--model_config', type=str, default='/lus/eagle/projects/CVD-Mol-AI/for_yuntian/patric_2.5B_pretraining/config/neox_2,533,931,008.json',
                    help='Model architecture config.')
parser.add_argument('--model_checkpoint', type=str, default='/lus/eagle/projects/CVD-Mol-AI/yuntian/sarscov2_filtered/sarscov2_filtered/transformers_diffusion_deepspeed/2.5B_10nodes_deepspeed_diffusion_sep_checkpoints_1e-4/checkpoint-13365/',
                    help='Language model checkpoint. Should be a folder containing pytorch_model.bin and other configuration files.')
parser.add_argument('--latent_dim', type=int, default=32,
                    help='Latent dimension of high level variables.')
parser.add_argument('--max_length', type=int, default=15000,
                    help='Maximum possible generation length. Generation will stop after generating this many of tokens, if no eos token is generated.')
parser.add_argument('--min_length', type=int, default=0,
                    help='Maximum possible generation length. Generation will stop after generating this many of tokens, if no eos token is generated.')
parser.add_argument('--block_size', type=int, default=1024,
                    help='Block size. Should be smaller or equal to the value during training.')
parser.add_argument('--sliding_stride', type=int, default=512,
                    help='Stride during generation.')
parser.add_argument('--seed', type=int, default=1234,
                    help='Random seed.')
parser.add_argument('--batch_size', type=int, default=10,
                    help='Batch_size.')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='Device.')
args = parser.parse_args()


class SequenceGenerationModule:
    def __init__(self, config):
        # Load diffusion samples
        diffusion_samples = torch.load(config.high_level_sample_file)
        #import pdb; pdb.set_trace()
        if isinstance(diffusion_samples, dict):
            diffusion_samples = diffusion_samples['test']
            samples = []
            examples = []
            for sample in diffusion_samples:
                samples.append(sample['padded_sentence_embeddings'])
                examples.append(sample['raw_text'])
            self.samples = torch.stack(samples, dim=0) # bsz, num_chunks, latent_dim
            self.examples = examples
        else:
            self.samples = diffusion_samples
            self.examples = None
        #import pdb; pdb.set_trace()
        # Do any expensive initialization up front
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Load tokenizer
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(config.tokenizer_file))
        bos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"
        unk_token = "[UNK]"
        bos_id = self.tokenizer.encode(bos_token, add_special_tokens=False, return_tensors='pt').item()
        eos_id = self.tokenizer.encode(eos_token, add_special_tokens=False, return_tensors='pt').item()
        unk_id = self.tokenizer.encode(unk_token, add_special_tokens=False, return_tensors='pt').item()
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id

        # Create model
        base_config = AutoConfig.from_pretrained(config.model_config)
        model = AutoModelForCausalLM.from_config(base_config)
        model.resize_token_embeddings(len(self.tokenizer))
        device = torch.device(config.device)
        self.device = device

        ## Load model
        ### Load cl model
        cl_model = language.GPT2OUEncoder(
             hidden_dim=128,
             latent_dim=config.latent_dim,
             finetune_gpt2=False, single_layer=False, genome=True, tokenizer=self.tokenizer)
        model.cl_model = cl_model
        proj_model = nn.Linear(config.latent_dim, model.gpt_neox.embed_in.weight.shape[1])
        model.proj_model = proj_model
        # Load model
        print (f'Loading model {os.path.join(config.model_checkpoint, "pytorch_model.bin")}')
        model.load_state_dict(torch.load(os.path.join(config.model_checkpoint, 'pytorch_model.bin')))

        model.to(device)
        model.eval()

        self.model = model

        # Default settings
        self.block_size = config.block_size
        self.sliding_stride = config.sliding_stride
        self.max_length = config.max_length
        self.min_length = config.min_length
        self.batch_examples = None

    def run(self, offset, prefix_codons='', top_p=1.0, batch_size=1,  max_length=None, min_length=None, sliding_stride=None, block_size=None, device=None):
        assert prefix_codons == '', 'to enable passing in prefix_codons we need to implement diffusion infilling'
        # high level latents
        cl_feats = self.samples[offset:offset+batch_size].to(self.device) # 22, latent_dim
        cl_feats = self.model.proj_model(cl_feats)
        if self.examples is not None:
            examples = self.examples[offset:offset+batch_size]
            self.batch_examples = examples
        #import pdb; pdb.set_trace()
        max_length= self.max_length if max_length is None else max_length 
        min_length = self.min_length if min_length is None else min_length 
        sliding_stride = self.sliding_stride if sliding_stride is None else sliding_stride
        block_size = self.block_size if block_size is None else block_size
        device = self.device if device is None else device

        prefix = f"[SEP] {self.eos_token}\n{self.bos_token} [SEP]"
        #cl_feats = torch.cat((cl_feats[:, -1:], cl_feats[:, -1:], cl_feats[:, 0:1], cl_feats[:, 1:]), dim=1)
        past_seq_cl_feats = torch.cat((cl_feats[:, -1:], cl_feats[:, -1:], cl_feats[:, 0:1]), dim=1)
        input_ids = self.tokenizer.encode(prefix, add_special_tokens=False, return_tensors='pt').to(device)
        prefix_length = input_ids.shape[-1]
        input_ids = self.tokenizer.encode(prefix, add_special_tokens=False, return_tensors='pt').to(device)
        input_ids = input_ids.view(1, -1).expand(batch_size, -1)

        m = min(block_size, max_length)
        do_sample = True
        if top_p == 0:
            do_sample = False
            top_p = 1
        self.model.chunk_ids = [0] * batch_size # pointing to current cl_feats
        self.model.past_seq_cl_feats = None
        self.model.chunk_id = 0
        self.model.chunk_offset = 0
        #import pdb; pdb.set_trace()
        sample_output = self.model.generate(input_ids, max_length=m, min_length=min(min_length, m), \
                cl_feats=cl_feats,
                past_seq_cl_feats=past_seq_cl_feats,
                do_sample=do_sample, top_p=top_p, eos_token_id=self.eos_id, \
                bad_words_ids=[[self.unk_id]])
        #import pdb; pdb.set_trace()
        is_final = torch.LongTensor(batch_size).fill_(0).eq(1).to(device)
        step_output = sample_output
        num_left = max_length - sample_output.shape[1]
        while num_left > 0 and not is_final.all():
            context = step_output[:, sliding_stride:]
            #import pdb; pdb.set_trace()
            past_seq_cl_feats = self.model.past_seq_cl_feats[:, sliding_stride:]
            self.model.past_seq_cl_feats = None
            input_ids = context
            context_length = context.shape[1]
            m = min(block_size, num_left+context_length)
            step_output = self.model.generate(input_ids, max_length=m, min_length=min(min_length-sample_output.shape[1]+context_length, m), \
                    cl_feats=cl_feats,
                    past_seq_cl_feats=past_seq_cl_feats,
                    do_sample=do_sample, top_p=top_p, eos_token_id=self.eos_id, \
                    bad_words_ids=[[self.unk_id]])
            is_final = is_final | step_output[:, context_length:].eq(self.eos_id).any(dim=-1)
            sample_output = torch.cat((sample_output, step_output[:, context_length:]), dim=-1)
            num_left = max_length - sample_output.shape[1]
        sequences = []
        #import pdb; pdb.set_trace()
        for sample in sample_output:
            sample = sample[prefix_length:]
            eos_pos = sample.eq(self.eos_id).nonzero().view(-1)
            if eos_pos.shape[0] > 0:
                eos_pos = eos_pos[0].item()
                sample = sample[:eos_pos+1]

            text = self.tokenizer.decode(sample, skip_special_tokens=False)
            text = text.replace('<|endoftext|>', '').strip()
            sequences.append(text)
        return sequences
    
def post_process(sequence):
    sequence = sequence.replace('<|endoftext|>', '').strip()
    words = sequence.strip().split()
    words = [word for word in words if word != '[SEP]']
    return ' '.join(words)

if __name__ == '__main__':
    batch_size = args.batch_size
    sequence_generator = SequenceGenerationModule(args)
    #for top_p in [1, 0.9, 0.8, 0.7, .6, .5, .4, .3, .2, .1, .0]:
    #for top_p in [1, 0.9, 0.8]:
    #for top_p in [0.7, 0.6, 0.5]:
    #for top_p in [.4, .3, .2]:
    #for top_p in [.1]:
    for top_p in [0.9]:
    #for top_p in [1]:
        print (f'Generating {top_p}')
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        #with open(f'hierarchical_generations_oracle/top_p{top_p}.txt', 'w') as fout:
        #    with open(f'hierarchical_generations_oracle_gt/gt_top_p{top_p}.txt', 'w') as fgt:
        with open(f'2.5B_hierarchical_generations_diffusion_vanilla2_p09/top_p{top_p}', 'w') as fout:
            with open(f'2.5B_hierarchical_generations_diffusion_gt_vanilla2_p09/gt_top_p{top_p}', 'w') as fgt:
                sys.stdout.flush()
                sequences = sequence_generator.run(offset=i*batch_size, prefix_codons='', top_p=top_p, batch_size=batch_size)
                batch_examples = sequence_generator.batch_examples
                for i, sequence in enumerate(sequences):
                    fout.write(post_process(sequence) + '\n')
                    fout.flush()
                    if batch_examples is not None:
                        fgt.write(post_process(batch_examples[i][0]) + '\n')
                        fgt.flush()

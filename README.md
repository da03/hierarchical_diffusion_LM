# Diffusion-based hierarchical modeling.

## Dependencies

Please follow the instructions in [genslm](https://github.com/ramanathanlab/genslm/blob/main/docs/INSTALL.md) to setup environment. This is particularly important if you plan to use DeepSpeed for distributed training.

Next, install this directory by

```
pip install -e .
```

## Training with DeepSpeed Zero Stage 2

For foundation models with fewer than (including) 2.5B parameters, we can train the model using Zero Stage 2:

```
export NODES=10
export GPUS_PER_NODE=4
export MASTER_ADDR=x3006c0s13b1n0.hsn.cm.polaris.alcf.anl.gov
export LR=1e-4
export EPOCHS=20
export TRAIN_BATCH_SIZE=2
export ACCUMULATION=1
export EVAL_BATCH_SIZE=1
export SAVE_TOTAL_LIMIT=5
export SAVE_FOLDER=2.5B_${NODES}nodes_deepspeed_diffusion_sep_checkpoints_${LR}
export TRAIN_FILE=data/sample_train.txt
export TEST_FILE=data/sample_val.txt
export CL_MODEL=/lus/eagle/projects/CVD-Mol-AI/yuntian/genomenewnaive/encoder_93810/run_l0.001_b32/checkpoints
export MODEL=EleutherAI/gpt-neox-20b # doesn't matter, will be ignored
deepspeed --num_gpus=${GPUS_PER_NODE} --num_nodes=${NODES} --master_addr=${MASTER_ADDR} --hostfile=hostfile --master_port=54321 examples/pytorch/language-modeling/run_clm_genslm_2.5B.py \
       --per_device_train_batch_size=${TRAIN_BATCH_SIZE} \
       --deepspeed=deepspeed_configs/zero2.json \
       --per_device_eval_batch_size=${EVAL_BATCH_SIZE} \
       --gradient_accumulation_steps=${ACCUMULATION} \
       --output_dir=${SAVE_FOLDER} \
       --model_type=${MODEL} \
       --model_name_or_path=${MODEL} \
       --do_train \
       --do_eval \
       --train_file=${TRAIN_FILE} \
       --validation_file=${TEST_FILE} --overwrite_output_dir --save_total_limit=${SAVE_TOTAL_LIMIT} \
       --learning_rate=${LR} --num_train_epochs=${EPOCHS} --load_best_model_at_end=True \
       --evaluation_strategy=epoch --save_strategy=epoch \
       --cl_model_name_or_path=${CL_MODEL} \
       --latent_dim=32 \
       --block_size 1024 --fp16 --prediction_loss_only

```

## Training with DeepSpeed Zero Stage 3

```
export NODES=10
export GPUS_PER_NODE=4
export MASTER_ADDR=x3006c0s13b1n0.hsn.cm.polaris.alcf.anl.gov
export LR=1e-4
export EPOCHS=20
export TRAIN_BATCH_SIZE=2
export ACCUMULATION=1
export EVAL_BATCH_SIZE=1
export SAVE_TOTAL_LIMIT=5
export SAVE_FOLDER=2.5B_${NODES}nodes_deepspeed_diffusion_sep_checkpoints_${LR}
export TRAIN_FILE=data/sample_train.txt
export TEST_FILE=data/sample_val.txt
export CL_MODEL=/lus/eagle/projects/CVD-Mol-AI/yuntian/genomenewnaive/encoder_93810/run_l0.001_b32/checkpoints
export MODEL=EleutherAI/gpt-neox-20b # doesn't matter, will be ignored
deepspeed --num_gpus=${GPUS_PER_NODE} --num_nodes=${NODES} --master_addr=${MASTER_ADDR} --hostfile=hostfile --master_port=54321 examples/pytorch/language-modeling/run_clm_genslm_25B.py \
       --per_device_train_batch_size=${TRAIN_BATCH_SIZE} \
       --deepspeed=deepspeed_configs/zero3.json \
       --per_device_eval_batch_size=${EVAL_BATCH_SIZE} \
       --gradient_accumulation_steps=${ACCUMULATION} \
       --output_dir=${SAVE_FOLDER} \
       --model_type=${MODEL} \
       --model_name_or_path=${MODEL} \
       --do_train \
       --do_eval \
       --train_file=${TRAIN_FILE} \
       --validation_file=${TEST_FILE} --overwrite_output_dir --save_total_limit=${SAVE_TOTAL_LIMIT} \
       --learning_rate=${LR} --num_train_epochs=${EPOCHS} --load_best_model_at_end=True \
       --evaluation_strategy=epoch --save_strategy=epoch \
       --cl_model_name_or_path=${CL_MODEL} \
       --latent_dim=32 \
       --block_size 1024 --fp16 --prediction_loss_only

```

## Generate

To generate, run

```
CUDA_VISIBLE_DEVICES=0 python examples/pytorch/language-modeling/generate_genslm_2.5B.py
```

## Citations
If you use our models in your research, please cite this paper:

```
@article{zvyagin2022genslms,
  title={GenSLMs: Genome-scale language models reveal SARS-CoV-2 evolutionary dynamics.},
  author={Zvyagin, Max T and Brace, Alexander and Hippe, Kyle and Deng, Yuntian and Zhang, Bin and Bohorquez, Cindy Orozco and Clyde, Austin and Kale, Bharat and Perez-Rivera, Danilo and Ma, Heng and others},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```

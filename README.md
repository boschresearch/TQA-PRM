# Exploring Generative Process Reward Modeling for Semi-Structured Data: A Case Study of Table Question Answering

Code for EACL submission [Exploring Generative Process Reward Modeling for Semi-Structured Data:
A Case Study of Table Question Answering](https://arxiv.org/abs/2510.20304)

## Overview
Process reward models (PRMs) improve complex reasoning in large language models (LLMs) by grading candidate solutions step-by-step and selecting answers via aggregated step scores. 
While effective in domains such as mathematics, their applicability to tasks involving semi-structured data, like table question answering (TQA) remains unexplored. 
TQA poses unique challenges for PRMs, including abundant irrelevant information, loosely connected reasoning steps, and domain-specific reasoning.
This work presents the first systematic study of PRMs for TQA. We evaluate state-of-the-art generative PRMs  on TQA from both answer and step perspectives. 
Results show that PRMs that combine textual and code verification can aid solution selection but struggle to generalize to out-of-domain data. 
Analysis reveals a weak correlation between performance in step-level verification and answer accuracy, possibly stemming from weak step dependencies and loose causal links. 
Our findings highlight limitations of current PRMs on TQA and offer valuable insights for building more robust, process-aware verifiers.


### Environment
```
conda create --name tqa_prm 

conda activate tqa_prm 

pip install -r requirements.txt
```

### Reasoning Path Generation

    bash policy_inference/inference.sh \
        --LM <model_path> \
        --round 0 \
        --bs 32 \
        --mt 6000 \
        --n_gpus 1 \
        --task scitab.json \
        --loop 1

### Rationale Generation

    python rationale_generation/process.py \
        --model_path <model_path> \
        --data_path <data_path> \
        --save_path <save_path> \
        --num_gpu_per 1 \
        --majority_of_N 1

### Consensus Filtering
    python -m consensus_filtering.dataset_preparation \
        --data_path <data_path> \
        --save_path <save_path> \
        --epsilon 0.8 
## Model Training
    deepspeed --num_gpus 1 /LLaMA-Factory/src/train.py \
        --deepspeed /home/tal1rng/LLaMA-Factory/examples/deepspeed/ds_z3_offload_config.json \
        --stage sft \
        --do_train True \
        --model_name_or_path <model_pth> \
        --preprocessing_num_workers 8 \
        --finetuning_type full \
        --template llama3 \
        --bf16 True \
        --flash_attn auto \
        --dataset_dir /LLaMA-Factory/data \
        --dataset E1 \
        --cutoff_len 2048 \
        --learning_rate 2e-05 \
        --num_train_epochs 1.0 \
        --max_samples 100000 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        --logging_steps 5 \
        --save_steps 0 \
        --warmup_steps 0 \
        --packing False \
        --enable_thinking False \
        --report_to none \
        --output_dir <output_pth> \
        --plot_loss True \
        --trust_remote_code True \
        --ddp_timeout 180000000 \
        --include_num_input_tokens_seen True \
        --optim adamw_torch \
## Evaluation
    python -m eval.eval_prm \
            --data_path <data_pth> \
            --data_name <data_name> \
            --size 8

## Acknowledgement
The code framework is mainly based on [GenPRM](https://github.com/RyanLiu112/GenPRM). The model training is based on [LlamaFactoryllama](https://github.com/hiyouga/LLaMA-Factory/tree/main/src/llamafactory).

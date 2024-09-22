
## 1. Data Preparation

This repository contains scripts for preparing two datasets: Toxigen and HateExplain. 

Install the necessary packages:

```bash
pip install -r requirements.txt
```

### Toxigen

To generate train, validation and test datasets used in the study do the following:

```bash
python -m data_prep.toxigen_prep --train_samples_per_group 800\
    --test_samples_per_group 50 \
    --path_to_save ../data/toxigen/ \
    --seed 42
```

### HateExplain

## 2. Fine-tune pre-trained BERT model

Fine-tune pre-trained BERT model on the training subset of the ToxiGen dataset by running: 

```bash
python -m exp_bert.train --checkpoint_dir ../output_bert/toxigen/base/ \
    --seed 42 \
    --data_dir ../data/toxigen/
```

## 3. Run 2 modules of D3M

1) Compute scores

```bash
python -m exp_bert.compute_firstmod_scores --checkpoint_dir ../output_bert/toxigen/base/best_checkpoint \
    --data_dir ../data/toxigen/ \
    --path_to_save ../output_bert/toxigen/


srun --gpus=1 --partition=a100-galvani --time=2:00:00 --output=firstmod_job_output --error=firstmod_job_errors sh -c "source ~/.bashrc && conda activate ../env && python -m exp_bert.compute_firstmod_scores --checkpoint_dir ../output_bert/toxigen/base/best_checkpoint \
    --data_dir ../data/toxigen/ \
    --path_to_save ../output_bert/toxigen/
    "&    
```

2) Run counterfactual

```bash
python -m exp_bert.run_counterfactual --checkpoint_dir ../output_bert/toxigen/base/best_checkpoint \
        --data_dir ../data/toxigen/ \
        --output_dir ../output_bert/toxigen/ \
        --method IF
```

```bash
srun --gpus=1 --partition=2080-galvani --time=14:00:00 --output=if_job_output --error=if_job_errors sh -c "source ~/.bashrc && conda activate ../env && python -m exp_bert.run_counterfactual --checkpoint_dir ../output_bert/toxigen/base/best_checkpoint \
    --data_dir ../data/toxigen/ \
    --output_dir ../output_bert/toxigen/\
    --method IF \
    --num_runs 3
    "&

srun --gpus=1 --partition=2080-galvani --time=14:00:00 --output=trak_job_output --error=trak_job_errors sh -c "source ~/.bashrc && conda activate ../env && python -m exp_bert.run_counterfactual --checkpoint_dir ../output_bert/toxigen/base/best_checkpoint \
    --data_dir ../data/toxigen/ \
    --output_dir ../output_bert/toxigen/\
    --method TRAK \
    --num_runs 3
    "&

srun --gpus=1 --partition=2080-galvani --time=14:00:00 --output=random_job_output --error=random_job_errors sh -c "source ~/.bashrc && conda activate ../env && python -m exp_bert.run_counterfactual --checkpoint_dir ../output_bert/toxigen/base/best_checkpoint \
    --data_dir ../data/toxigen/ \
    --output_dir ../output_bert/toxigen/\
    --method random \
    --num_runs 3
    "&
```

and plot the results

```bash
python -m exp_bert.vis_counterfactual --checkpoint_dir ../output_bert/toxigen/base/best_checkpoint \
        --data_dir ../data/toxigen/ \
        --results_dir ../output_bert/toxigen/ \
        --path_to_save vis/vis_bert_toxigen/
```

```bash
python -m exp_bert.vis_scores --checkpoint_dir ../output_bert/toxigen/base/best_checkpoint \
        --data_dir ../data/toxigen/ \
        --results_dir ../output_bert/toxigen/ \
        --path_to_save vis/vis_bert_toxigen/
```
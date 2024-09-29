
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
```bash
python -m data_prep.hatexplain_prep 
    --path_to_save ../data/hatexplain/ \
    --path_to_save_vis vis/vis_bert_hatexplain
```    

## 2. Fine-tune pre-trained BERT model

Fine-tune pre-trained BERT model on the training subset of the ToxiGen dataset by running: 

```bash
python -m exp_bert.train --checkpoint_dir ../output_bert/toxigen/base/ \
    --seed 42 \
    --data_dir ../data/toxigen/
```


```bash
srun --gpus=1 --partition=a100-galvani --time=2:00:00 --output=if_job_output --error=if_job_errors sh -c "source ~/.bashrc && conda activate ../env && python -m exp_bert.train --checkpoint_dir ../output_bert/hatexplain/base/ \
    --seed 42 \
    --data_dir ../data/hatexplain/
    "&
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

srun --gpus=1 --partition=a100-galvani --time=4:00:00 --output=firstmod_job_output --error=firstmod_job_errors sh -c "source ~/.bashrc && conda activate ../env && python -m exp_bert.compute_firstmod_scores --checkpoint_dir ../output_bert/hatexplain/base/best_checkpoint \
    --data_dir ../data/hatexplain/ \
    --path_to_save ../output_bert/hatexplain/
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




srun --gpus=1 --partition=2080-galvani --time=4:00:00 --output=if_job_output --error=if_job_errors sh -c "source ~/.bashrc && conda activate ../env && python -m exp_bert.run_counterfactual --checkpoint_dir ../output_bert/hatexplain/base/best_checkpoint \
    --data_dir ../data/hatexplain/ \
    --output_dir ../output_bert/hatexplain/\
    --method IF \
    --num_runs 3
    "&

srun --gpus=1 --partition=2080-galvani --time=4:00:00 --output=trak_job_output --error=trak_job_errors sh -c "source ~/.bashrc && conda activate ../env && python -m exp_bert.run_counterfactual --checkpoint_dir ../output_bert/hatexplain/base/best_checkpoint \
    --data_dir ../data/hatexplain/ \
    --output_dir ../output_bert/hatexplain/\
    --method TRAK \
    --num_runs 3
    "&

srun --gpus=1 --partition=2080-galvani --time=4:00:00 --output=random_job_output --error=random_job_errors sh -c "source ~/.bashrc && conda activate ../env && python -m exp_bert.run_counterfactual --checkpoint_dir ../output_bert/hatexplain/base/best_checkpoint \
    --data_dir ../data/hatexplain/ \
    --output_dir ../output_bert/hatexplain/\
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
        --output_dir ../output_bert/toxigen/ \
        --path_to_save vis/vis_bert_toxigen/
```

```bash
python -m exp_bert.vis_counterfactual --checkpoint_dir ../output_bert/hatexplain/base/best_checkpoint \
        --data_dir ../data/hatexplain/ \
        --results_dir ../output_bert/hatexplain/ \
        --path_to_save vis/vis_bert_hatexplain/
```

```bash
python -m exp_bert.compute_metrics --checkpoint_dir ../output_bert/hatexplain/base/best_checkpoint \
        --data_dir ../data/hatexplain/ \
        --output_dir ../output_bert/hatexplain/ \
        --path_to_save vis/vis_bert_hatexplain/
```

3) Compare the results to baseline balancing

```bash

srun --gpus=1 --partition=2080-galvani --time=2:00:00 --output=balance_job_output --error=balance_job_errors sh -c "source ~/.bashrc && conda activate ../env && python -m exp_bert.run_baseline_balancing --checkpoint_dir ../output_bert/toxigen/base/best_checkpoint \
        --data_dir ../data/toxigen/ \
        --path_to_save res/toxigen/ \
        --path_to_save_model ../output_bert/toxigen/ \
        --output_dir ../output_bert/toxigen/
    "&

srun --gpus=1 --partition=2080-galvani --time=2:00:00 --output=balance_job_output --error=balance_job_errors sh -c "source ~/.bashrc && conda activate ../env && python -m exp_bert.run_baseline_balancing --checkpoint_dir ../output_bert/hatexplain/base/best_checkpoint \
        --data_dir ../data/hatexplain/ \
        --path_to_save res/hatexplain/ \
        --path_to_save_model ../output_bert/hatexplain/ \
        --output_dir ../output_bert/hatexplain/
    "&    
```

```bash
python -m exp_bert.vis_scores_matrix --data_dir ../data/toxigen/ \
        --output_dir ../output_bert/toxigen \
        --path_to_save vis/vis_bert_toxigen/

python -m exp_bert.vis_scores_matrix --data_dir ../data/hatexplain/ \
        --output_dir ../output_bert/hatexplain \
        --path_to_save vis/vis_bert_hatexplain/
```

```bash
python -m exp_bert.get_examples --checkpoint_dir ../output_bert/toxigen/base/best_checkpoint \
        --data_dir ../data/toxigen/ \
        --output_dir ../output_bert/toxigen/ \
        --path_to_save res/toxigen/


python -m exp_bert.get_examples --checkpoint_dir ../output_bert/hatexplain/base/best_checkpoint \
        --data_dir ../data/hatexplain/ \
        --output_dir ../output_bert/hatexplain/ \
        --path_to_save res/hatexplain/
```


```bash
python -m exp_bert.vis_scores_matrix --data_dir ../data/toxigen/ \
        --output_dir ../output_bert/toxigen/ \
        --path_to_save res/toxigen/


python -m exp_bert.vis_scores_matrix --data_dir ../data/hatexplain/ \
        --output_dir ../output_bert/hatexplain/ \
        --path_to_save res/hatexplain/
```


```bash
python -m exp_bert.vis_scores_label --checkpoint_dir ../output_bert/toxigen/base/best_checkpoint \
        --data_dir ../data/toxigen/ \
        --output_dir ../output_bert/toxigen/ \
        --path_to_save res/toxigen/


python -m exp_bert.vis_scores_label --checkpoint_dir ../output_bert/hatexplain/base/best_checkpoint \
        --data_dir ../data/hatexplain/ \
        --output_dir ../output_bert/hatexplain/ \
        --path_to_save res/hatexplain/
```
# Data Debiasing for Language Models via Training Data Attribution: Improving Worst-Group Accuracy

This is my master's thesis project. We aim to determine whether selected training data attribution explainability methods (TRAK and influence functions) can debias training datasets, and therefore improve the fairness of models trained on datasets debiased in this way.

We extend the "D3M" method described in [Data Debiasing with Datamodels (D3M): Improving Subgroup Robustness via Data Selection (Jain et al., 2024)](https://arxiv.org/abs/2406.16846). The method consists of 2 modules: 

- TRAK scores computation, 

- identifying the most influential training data samples that contribute negatively to worst-group accuracy by computing group alignment scores.

 In the project, we examine whether the alternative, both more simple and more complex methods, provide similar results, improvement of worst-group accuracy in particular. We modify the first module and compute the following scores in addition to TRAK:

- Influence functions (kronfluence)

- TRAK scores

Then, we feed the scores obtained in the first module to the second one to compute group alignment scores for each training sample. 

We apply the method to a hate speech classification as it is one of the most well-studied NLP tasks with corresponding benchmarks. 

Datasets: [ToxiGen](https://github.com/microsoft/TOXIGEN), [HateExplain](https://github.com/hate-alert/HateXplain)

Models that are to be finetuned for the hate-speech classification: [Bert base model (uncased)](https://huggingface.co/google-bert/bert-base-uncased).



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
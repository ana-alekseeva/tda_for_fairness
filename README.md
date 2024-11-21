# Improving Group Fairness in Language Models via Training Data Attribution

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

The CLI commands provided below are designed for the ToxiGen dataset. To apply them to the HateXplain dataset, simply replace instances of 'toxigen' with 'hatexplain'.

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

```

 2) Run counterfactual

```bash
python -m exp_bert.run_counterfactual --checkpoint_dir ../output_bert/toxigen/base/best_checkpoint \
        --data_dir ../data/toxigen/ \
        --output_dir ../output_bert/toxigen/ \
        --method IF
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


3) Compare the results to baseline balancing

```bash

python -m exp_bert.run_baseline_balancing --checkpoint_dir ../output_bert/toxigen/base/best_checkpoint \
        --data_dir ../data/toxigen/ \
        --path_to_save res/toxigen/ \
        --path_to_save_model ../output_bert/toxigen/ \
        --output_dir ../output_bert/toxigen/
    "&    
```

```bash
python -m exp_bert.vis_scores_matrix --data_dir ../data/toxigen/ \
        --output_dir ../output_bert/toxigen \
        --path_to_save vis/vis_bert_toxigen/
```

```bash
python -m exp_bert.get_examples --checkpoint_dir ../output_bert/toxigen/base/best_checkpoint \
        --data_dir ../data/toxigen/ \
        --output_dir ../output_bert/toxigen/ \
        --path_to_save res/toxigen/
```


```bash
python -m exp_bert.vis_scores_matrix --data_dir ../data/toxigen/ \
        --output_dir ../output_bert/toxigen/ \
        --path_to_save res/toxigen/
```


```bash
python -m exp_bert.vis_scores_label --checkpoint_dir ../output_bert/toxigen/base/best_checkpoint \
        --data_dir ../data/toxigen/ \
        --output_dir ../output_bert/toxigen/ \
        --path_to_save res/toxigen/
```
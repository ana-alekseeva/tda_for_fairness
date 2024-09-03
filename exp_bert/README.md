
## 1. Data Preparation

This repository contains scripts for preparing two datasets: Toxigen and HateExplain. 

Install the necessary packages:

```bash
pip install -r requirements.txt
```

### Toxigen

To generate train, validation and test datasets used in the study do the following:

```bash
python toxigen_prep.py --train_samples_per_group 2000\
    --test_samples_per_group 100 \
    --path_to_save ../../data/toxigen/ \
    --seed 42
```

### HateExplain

## 2. Fine-tune pre-trained BERT model

Fine-tune pre-trained BERT model on the training subset of the ToxiGen dataset by running: 

```bash
python train.py --checkpoint_dir ../../output_bert/toxigen/base/ \
    --seed 42 \
    --data_dir ../../data/toxigen/
```

## 3. Run 2 modules of D3M

1) Compute scores

```bash
python compute_firstmod_scores.py --checkpoint_dir ../../output_bert/toxigen/base/best_checkpoint \
    --data_dir ../../data/toxigen/
    --path_to_save ../../output_bert/toxigen/
```

2) Run counterfactual

```bash
python run_counterfactual.py --checkpoint_dir ../../output_bert/toxigen/base/best_checkpoint \
        --data_dir ../../data/toxigen/
        --output_dir ../../output_bert/toxigen/
```
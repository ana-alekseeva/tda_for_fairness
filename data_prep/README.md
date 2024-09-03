# Data Preparation

This repository contains scripts for preparing two datasets: Toxigen and HateExplain. 

Install the necessary packages:

```bash
pip install -r requirements.txt
```

## Toxigen

To generate train, validation and test datasets used in the study do the following:

```bash
python toxigen_prep.py --test_samples_per_group 100 \
    --path_to_save ../../data/toxigen/ \
    --seed 42
```
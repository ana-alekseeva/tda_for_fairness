
1) Fine-tune pre-trained BERT model on the training subset of the ToxiGen dataset by running: 

```bash
python train.py --checkpoint_dir ../../output_bert/toxigen/base/ \
    --seed 42 \
    --data_dir ../../data/toxigen/
```

2) Compute scores

```bash
python compute_scores.py --checkpoint_dir ../../output_bert/toxigen/base/best_checkpoint \
    --data_dir ../../data/toxigen/
    --path_to_save ../../output_bert/toxigen/
```

3) Run counterfactual

```bash
python train.py 
```
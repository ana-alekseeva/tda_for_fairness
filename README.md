# Data Debiasing for Language Models via Training Data Attribution: Improving Worst-Group Accuracy

This is my master's thesis project. We aim to determine whether selected training data attribution explainability methods (TRAK and influence functions) can debias training datasets, and therefore improve the fairness of models trained on datasets debiased in this way.

We extend the "D3M" method described in [Data Debiasing with Datamodels (D3M): Improving Subgroup Robustness via Data Selection (Jain et al., 2024)](https://arxiv.org/abs/2406.16846). The method consists of 2 modules: 

- TRAK scores computation, 

- identifying the most influential training data samples that contribute negatively to worst-group accuracy by computing group alignment scores.

 In the project, we examine whether the alternative, both more simple and more complex methods, provide similar results, improvement of worst-group accuracy in particular. We modify the first module and compute the following scores in addition to TRAK:

- Bm25 scores

- FAISS scores
 
- Influence functions (kronfluence)

- TRAK scores

Then, we feed the scores obtained in the first module to the second one to compute group alignment scores for each training sample. 

We apply the method to a hate speech classification as it is one of the most well-studied NLP tasks with corresponding benchmarks. 

Datasets: [ToxiGen](https://github.com/microsoft/TOXIGEN), [HateExplain](https://github.com/hate-alert/HateXplain)

Models that are to be finetuned for the hate-speech classification: [Bert base model (uncased)](https://huggingface.co/google-bert/bert-base-uncased), [RoBERTa base model](https://huggingface.co/FacebookAI/roberta-base), and a large language model.
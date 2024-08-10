# Data Debiasing for Language Models via Training Data Attribution: Improving Worst-Group Accuracy

This is my amazing master's thesis project!


We extend the "D3M" method develop by the Madry's Lab and examine whether the alternative, both more simple and more complex methods, provide similar results, improvement of worst-group accuracy in particular. We modify the first module and compute:

- Bm25 scores

- FAISS scores
 
- Influence functions (kronfluence)

- TRAK scores

Then, we feed the scores obtained in the first module to the second one to compute group alignment scores for each training sample. 
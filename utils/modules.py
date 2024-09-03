import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from torch import nn
from transformers import default_data_collator

from rank_bm25 import BM25Okapi
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import faiss
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.arguments import FactorArguments, ScoreArguments
from trak import TRAKer

from typing import List, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from tda_for_fairness.utils import get_dataloader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FirstModuleBaseline():
    def __init__(self,train_texts, test_texts,model, tokenizer, path_to_save):
        self.train_texts = train_texts
        self.test_texts = test_texts
        self.model = model
        self.tokenizer = tokenizer
        self.path_to_save = path_to_save
    
    def get_embeddings(self, texts):
        self.model.eval()

        embeddings = []
        for i in range(0,len(texts),config.BATCH_SIZE):
            try:
                batch = texts[i:i+config.BATCH_SIZE]
            except:
                batch = texts[i:]
            with torch.no_grad():
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=config.MAX_LENGTH)
                inputs.to(DEVICE)
                outputs = self.model(**inputs, output_hidden_states=True)
            # Extract the hidden states 
            hidden_states = outputs.hidden_states
            embeddings.append(hidden_states[-1][:, 0, :])

        return torch.vstack(embeddings)


    def get_Bm25_scores(self):

        def preprocessing(texts):
            nltk.download('punkt')
            nltk.download('stopwords')
            stemmer = PorterStemmer()
            stop_words = set(stopwords.words('english'))
            
            preprocessed_texts = []
            
            for text in texts:
                # Convert to lowercase
                text = text.lower()
                # Remove punctuation
                text = text.translate(str.maketrans('', '', string.punctuation))
                # Tokenize
                tokens = word_tokenize(text)
                # Remove stopwords and stem
                processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
                # Join tokens back into a string
                preprocessed_text = ' '.join(processed_tokens)
                preprocessed_texts.append(preprocessed_text)
            return preprocessed_texts
        
        # Preprocessing
        train_texts_preprocessed = preprocessing(self.train_texts)
        test_texts_preprocessed = preprocessing(self.test_texts)

        bm25 = BM25Okapi(train_texts_preprocessed)
        scores = torch.from_numpy(
                np.vstack(
                    [bm25.get_scores(query) for query in test_texts_preprocessed],
                ))
        
        # normalize scores
        scores = scores / scores.sum(axis=1,keepdims = True)
        torch.save(scores, self.path_to_save + 'BM25_scores.pt')

    def get_FAISS_scores(self):
        """
        Compute 2 types of similarity scores using the FAISS library:
        1) Cosine similarity
        2) Inverse of L2 distance
        """

        test_embeddings = self.get_embeddings(self.test_texts)
        train_embeddings = self.get_embeddings(self.train_texts)
   
        train_embeddings = train_embeddings.cpu().numpy().astype(np.float32)
        test_embeddings = test_embeddings.cpu().numpy().astype(np.float32)
 
        n_train, d = train_embeddings.shape

        def sim_matrix_from_index(index, train_embeddings=train_embeddings, test_embeddings=test_embeddings):  
            index.add(train_embeddings)
            D,I = index.search(test_embeddings, n_train)
            reordered_D = np.zeros_like(D)
            for row in range(I.shape[0]):
                reordered_D[row, I[row]] = D[row]
            return reordered_D

        # 1.Cosine similarity
        index = faiss.IndexFlatIP(d) 
        train_norm_embeddings = F.normalize(train_embeddings, p=2, dim=1)
        test_norm_embeddings = F.normalize(test_embeddings, p=2, dim=1)
        cosine_scores = sim_matrix_from_index(index,train_norm_embeddings, test_norm_embeddings) 
        torch.save(cosine_scores, self.path_to_save + 'cosine_scores.pt')

        # 2. Inverse of L2 distance
        index = faiss.IndexFlatL2(d)
        l2_scores = sim_matrix_from_index(index) 
        l2_scores = 1/(1e-3 + l2_scores)
        torch.save(l2_scores, self.path_to_save + 'l2_scores.pt')



class FirstModuleTDA():
    def __init__(self,train_dataset,test_dataset,model, path_to_save):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model.to(DEVICE)
        self.path_to_save = path_to_save     

    def get_IF_scores(self):

        class TextClassificationTask(Task):
            def compute_train_loss(
                self,
                batch,
                model: nn.Module,
                sample: bool = False,
            ) -> torch.Tensor:
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                ).logits
                if not sample:
                    return F.cross_entropy(logits, batch["labels"], reduction="sum")
                with torch.no_grad():
                    probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
                    sampled_labels = torch.multinomial(
                        probs,
                        num_samples=1,
                    ).flatten()
                return F.cross_entropy(logits, sampled_labels, reduction="sum")

            def compute_measurement(
                self,
                batch,
                model: nn.Module,
            ) -> torch.Tensor:
                # Copied from: https://github.com/MadryLab/trak/blob/main/trak/modelout_functions.py.
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                ).logits

                labels = batch["labels"]
                bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
                logits_correct = logits[bindex, labels]

                cloned_logits = logits.clone()
                cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

                margins = logits_correct - cloned_logits.logsumexp(dim=-1)
                return -margins.sum()

            def get_attention_mask(self, batch) -> torch.Tensor:
                return batch["attention_mask"]
            

            def tracked_modules(self) -> Optional[List[str]]:
                # These are the module names we will use to compute influence functions.
                return ["classifier"]
               
        task = TextClassificationTask()
        model = prepare_model(self.model, task)


        analyzer = Analyzer(
            analysis_name="toxigen_bert",
            model=model,
            task=task,
            profile=False,
        )
        
        # Configure parameters for DataLoader.
        dataloader_kwargs = DataLoaderKwargs(collate_fn=default_data_collator)
        analyzer.set_dataloader_kwargs(dataloader_kwargs)

        analyzer.fit_all_factors(
                    factors_name="ekfac",
                    dataset=self.train_dataset,
                    per_device_batch_size=config.BATCH_SIZE,
                    overwrite_output_dir=True,
                )


        # Compute influence factors.
        factors_name = "ekfac"
        factor_args = FactorArguments(strategy="ekfac")

        # Compute pairwise scores.
        score_args = ScoreArguments()
        scores_name = factor_args.strategy

        analyzer.compute_pairwise_scores(
            score_args=score_args,
            scores_name=scores_name,
            factors_name=factors_name,
            query_dataset=self.test_dataset,
            #query_indices=list(range(min([len(self.val_dataset), 2000]))),
            train_dataset=self.train_dataset,
            per_device_query_batch_size=8,
            per_device_train_batch_size=8,
            overwrite_output_dir=False,
        )
        scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]


        torch.save(scores, self.path_to_save + 'IF_scores.pt')

    def get_TracIn_scores(self):
        pass

    def get_TRAK_scores(self):
        class SequenceClassificationModel(nn.Module):
            """
            Wrapper for HuggingFace sequence classification models.
            """
            def __init__(self,model):
                super().__init__()
                self.model = model
                self.model.eval().to(DEVICE)

            def forward(self, input_ids, token_type_ids, attention_mask):
                return self.model(input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask).logits


        model = SequenceClassificationModel(self.model)
        

        def process_batch(batch):
            return batch['input_ids'], batch['token_type_ids'], batch['attention_mask'], batch['labels']

        train_dataloader = get_dataloader(self.train_dataset, 8,shuffle=False)
        test_dataloader = get_dataloader(self.test_dataset, 8, shuffle=False)

        traker = TRAKer(model=model,
                        task="text_classification",
                        train_set_size=self.train_dataset.num_rows,
                        save_dir=self.path_to_save,
                        device=DEVICE,
                        proj_dim= 1024) 

        traker.load_checkpoint(model.state_dict(), model_id=0)
        for batch in tqdm(train_dataloader, desc='Featurizing..'):
            # process batch into compatible form for TRAKer TextClassificationModelOutput
            batch = process_batch(batch)
            batch = [x.to(DEVICE) for x in batch]
            traker.featurize(batch=batch, num_samples=batch[0].shape[0])

        traker.finalize_features()

        traker.start_scoring_checkpoint(exp_name='trak_exp',
                                        checkpoint=model.state_dict(),
                                        model_id=0,
                                        num_targets=self.test_dataset.num_rows)
        
        for batch in tqdm(test_dataloader, desc='Scoring..'):
            batch = process_batch(batch)
            batch = [x.to(DEVICE) for x in batch]
            traker.score(batch=batch, num_samples=batch[0].shape[0])

        scores = traker.finalize_scores(exp_name='trak_exp')
        torch.save(scores.T, self.path_to_save + 'TRAK_scores.pt')



class D3M:
    """
    Data Debiasing with Datamodels
    """

    def __init__(
        self,
        model,
        checkpoints,
        train_dataloader,
        val_dataloader,
        group_indices_train,
        group_indices_val,
        scores,
        train_set_size=None,
        val_set_size=None
    ) -> None:
        """
        Args:
            model:
                The model to be debiased.
            checkpoints:
                A list of model checkpoints (state dictionaries) for debiasing
                (used to compute TRAK scores).
            train_dataloader:
                DataLoader for the training dataset.
            val_dataloader:
                DataLoader for the validation dataset.
            group_indices:
                A list indicating the group each sample in the validation
                dataset belongs to.
            scores:
                Precomputed scores.
            train_set_size (optional):
                The size of the training dataset. Required if the dataloader
                does not have a dataset attribute.
            val_set_size (optional):
                The size of the validation dataset. Required if the dataloader
                does not have a dataset attribute.
            trak_kwargs (optional):
                Additional keyword arguments to be passed to
                `attrib.get_trak_matrix`.
            device (optional):
                pytorch device
        """
        self.model = model.to(DEVICE)
        self.checkpoints = checkpoints
        self.dataloaders = {"train": train_dataloader, "val": val_dataloader}
        self.group_indices_train = group_indices_train
        self.group_indices_val = group_indices_val
        self.device = DEVICE
        self.scores = scores

    def get_group_losses(self, model, val_dl, group_indices) -> list:
        """Returns a list of losses for each group in the validation set."""
        losses = []
        model.eval()
        with torch.no_grad():
            #for inputs, labels in val_dl:
            for batch in val_dl:
                batch = {k:batch[k].to(self.device) for k in batch.keys()}
                outputs = model(**batch)["logits"]
                loss = F.cross_entropy(
                    outputs, batch["labels"], reduction="none"
                )
                losses.append(loss)
        losses = torch.cat(losses)

        n_groups = len(set(group_indices))
        group_losses = [losses[np.array(group_indices) == i].mean() for i in range(n_groups)]
        return group_losses

    def compute_group_alignment_scores(self, scores, group_indices, group_losses):
        """
        Computes group alignment scores (check Section 3.2 in our paper for
        details).

        Args:
            scores:
                result of get_trak_matrix
            group_indices:
                a list of the form [group_index(x) for x in train_dataset]

        Returns:
            a list of group alignment scores for each training example
        """
        n_groups = len(set(group_indices))
        S = np.array(scores)
        g = [
            group_losses[i].cpu().numpy() * np.nanmean(S[:, np.array(group_indices) == i],axis=1)
            for i in range(n_groups)
        ]
        g = np.stack(g)
        
        nan_perc = np.isnan(g).mean()
        if nan_perc > 0.2:
            raise ValueError(f"The percentage of NaNs in g is {nan_perc}.")
        group_alignment_scores = np.nanmean(g,axis=0) # I changed here as some BM25 scores are too small
        return group_alignment_scores

    def get_debiased_train_indices(
        self, group_alignment_scores, use_heuristic=True, num_to_discard=None
    ):
        """
        If use_heuristic is True, training examples with negative score will be discarded,
        and the parameter num_to_discard will be ignored
        Otherwise, the num_to_discard training examples with lowest scores will be discarded.
        """
        if use_heuristic:
            return [i for i, score in enumerate(group_alignment_scores) if score >= 0]

        if num_to_discard is None:
            raise ValueError("num_to_discard must be specified if not using heuristic.")

        sorted_indices = sorted(
            range(len(group_alignment_scores)),
            key=lambda i: group_alignment_scores[i],
        )
        return sorted_indices[num_to_discard:]

    def debias(self, use_heuristic=True, num_to_discard=None):
        """
        Debiases the training process by constructing a new training set that
        excludes examples which harm worst-group accuracy.

        Args:
            use_heuristic:
                If True, examples with negative group alignment scores are
                discarded.  If False, the `num_to_discard` examples with the
                lowest scores are discarded.
            num_to_discard:
                The number of training examples to discard based on their group
                alignment scores.  This parameter is ignored if `use_heuristic`
                is True.

        Returns:
            debiased_train_inds (list):
                A list of indices for the training examples that should be
                included in the debiased training set.
        """

        # Step 2 (Step 1 is to compute TRAK scores):
        # compute group alignment scores
        group_losses = self.get_group_losses(
            model=self.model,
            val_dl=self.dataloaders["val"],
            group_indices=self.group_indices_val,
        )

        group_alignment_scores = self.compute_group_alignment_scores(
            self.scores, self.group_indices_val, group_losses  # I changed group_indices_train to group_indices_val
        )

        # Step 3:
        # construct new training set
        debiased_train_inds = self.get_debiased_train_indices(
            group_alignment_scores,
            use_heuristic=use_heuristic,
            num_to_discard=num_to_discard,
        )

        return debiased_train_inds
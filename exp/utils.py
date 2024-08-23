import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import torch
from torch.nn import functional as F
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
from transformers import default_data_collator
from typing import List, Optional, Tuple
from torch import nn
import math
import config
import datasets_prep as dp
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from trak import TRAKer


def plot_distr_by_group(df, title):
    """
    Plot the distribution of samples by group.
    """
    # Create a count of samples for each combination of label and target group
    grouped_data = df.groupby(['target_group', 'label']).size().unstack()

    # Set up the plot
    plt.figure(figsize=(12, 6))

    # Create the grouped bar plot
    grouped_data.plot(kind='bar', ax=plt.gca())

    # Customize the plot
    plt.title('Distribution of Labels by Group')
    plt.xlabel('Group')
    plt.ylabel('Number of Samples')
    plt.legend(['Neutral (0)', 'Hate (1)'])

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add value labels on top of each bar
    for i in range(len(grouped_data)):
        for j in range(len(grouped_data.columns)):
            value = grouped_data.iloc[i, j]
            plt.text(i, value, str(value), ha='center', va='bottom')

    # Adjust layout to prevent cutoff
    plt.tight_layout()
    plt.savefig(f'../vis/distr_by_group_{title}.png', dpi=300, bbox_inches='tight')


class FirstModuleBaseline():
    def __init__(self,train_texts, test_texts,model, tokenizer):
        self.train_texts = train_texts
        self.test_texts = test_texts
        self.model = model
        self.tokenizer = tokenizer
    
    def get_embeddings(self, texts):
        # Set model to evaluation mode
        self.model.eval()

        embeddings = []
        for i in range(0,len(texts),config.BATCH_SIZE):
            try:
                batch = texts[i:i+config.BATCH_SIZE]
            except:
                batch = texts[i:]
            # Get the model's output
            with torch.no_grad():
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=config.MAX_LENGTH).to("cuda")
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
        torch.save(scores, '../../output/BM25_scores.pt')

    def get_FAISS_scores(self):

        test_embeddings = self.get_embeddings(self.test_texts)
        train_embeddings = self.get_embeddings(self.train_texts)
   
        train_embeddings = train_embeddings.cpu().numpy().astype(np.float32)
        test_embeddings = test_embeddings.cpu().numpy().astype(np.float32)
 
        n_train, d = train_embeddings.shape
   
        index = faiss.IndexFlatIP(d)  # Inner product similarity
        # Add the training embeddings to the index
        index.add(train_embeddings)
        # Compute similarities
        scores, _ = index.search(test_embeddings, n_train)
        
        torch.save(scores, '../../output/FAISS_scores.pt')

    def get_gradient_scores(self,val_labels):
        
        # Set the model to evaluation mode and enable gradient computation

        # Prepare the training sample
        train_encoding = self.tokenizer(self.train_texts["text"].to_list(), return_tensors='pt', padding=True, truncation=True, max_length=config.MAX_LENGTH)
        train_labels = self.train_texts['label'].to_list()
        #train_input_ids = train_encoding['input_ids']
        #train_attention_mask = train_encoding['attention_mask']

        # Prepare the test sample
        #test_encoding = self.tokenizer(self.test_texts, return_tensors='pt', padding=True, truncation=True, max_length=config.MAX_LENGTH)
        #test_input_ids = test_encoding['input_ids']
        #test_attention_mask = test_encoding['attention_mask']


        train_dataset = TensorDataset(train_encoding['input_ids'], train_encoding['attention_mask'], torch.tensor(train_labels))
        train_loader = DataLoader(train_dataset, batch_size=len(self.train_texts))  # Use all training samples in one batch

        # Function to get sentence embeddings
        def get_sentence_embedding(model, input_ids, attention_mask):
            with torch.no_grad():
                outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding

        # Compute sentence embeddings for all training samples
        self.model.eval()
        for batch in train_loader:
            train_input_ids, train_attention_mask, _ = batch
            train_embeddings = get_sentence_embedding(self.model, train_input_ids, train_attention_mask)

        # Enable gradient computation for train embeddings
        train_embeddings.requires_grad_()

        for test_sample, test_label in zip(self.test_texts["text"], self.test_texts["label"]):
        # Compute loss for test sample
            test_encoding = self.tokenizer(test_sample, return_tensors='pt', padding=True, truncation=True, max_length=config.MAX_LENGTH)
            test_input_ids = test_encoding['input_ids']
            test_attention_mask = test_encoding['attention_mask']

            # Forward pass
            outputs = self.model(input_ids=test_input_ids, attention_mask=test_attention_mask, labels=torch.tensor([test_label]))
            loss = outputs.loss

            # Compute gradients
            loss.backward()

            # The gradients are now stored in train_embeddings.grad
            gradients = train_embeddings.grad
            break
        return gradients

        #scores = ...
        #torch.save(scores, '../../output/gradients_scores.pt')

class FirstModuleTDA():
    def __init__(self,train_dataset,test_dataset,model):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model.to("cuda")      

    def get_IF_scores(self,out):

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


        torch.save(scores, '../../output/IF_scores.pt')

    def get_TRAK_scores(self,out):

        def process_batch(batch):
            return batch['input_ids'], batch['token_type_ids'], batch['attention_mask'], batch['labels']

        device = 'cuda'
        model = self.model.to(device)
        train_dataloader = dp.get_dataloader(self.train_dataset, 8)
        test_dataloader = dp.get_dataloader(self.test_dataset, 8)

        traker = TRAKer(model=self.model,
                        task="text_classification",
                        train_set_size=self.train_dataset.num_rows,
                        save_dir=out,
                        device=device,
                        proj_dim= 1024) 

        traker.load_checkpoint(model.state_dict(), model_id=0)
        for batch in tqdm(train_dataloader, desc='Featurizing..'):
            # process batch into compatible form for TRAKer TextClassificationModelOutput
            batch = process_batch(batch)
            batch = [x.to(device) for x in batch]
            traker.featurize(batch=batch, num_samples=batch[0].shape[0])

        traker.finalize_features()

        traker.start_scoring_checkpoint(exp_name='toxigen_bert',
                                        checkpoint=model.state_dict(),
                                        model_id=0,
                                        num_targets=self.val_dataset.num_rows)
        
        for batch in tqdm(test_dataloader, desc='Scoring..'):
            batch = process_batch(batch)
            batch = [x.cuda() for x in batch]
            traker.score(batch=batch, num_samples=batch[0].shape[0])

        scores = traker.finalize_scores(exp_name='toxigen_bert')
        torch.save(scores, '../../output/TRAK_scores.pt')



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
        val_set_size=None,
        device="cuda"
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
        self.model = model.to(device)
        self.checkpoints = checkpoints
        self.dataloaders = {"train": train_dataloader, "val": val_dataloader}
        self.group_indices_train = group_indices_train
        self.group_indices_val = group_indices_val
        self.device = device
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
            group_losses[i].cpu().numpy() * S[:, np.array(group_indices) == i].mean(axis=1)
            for i in range(n_groups)
        ]
        g = np.stack(g)
        group_alignment_scores = g.mean(axis=0)
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
            self.scores, self.group_indices_train, group_losses
        )

        # Step 3:
        # construct new training set
        debiased_train_inds = self.get_debiased_train_indices(
            group_alignment_scores,
            use_heuristic=use_heuristic,
            num_to_discard=num_to_discard,
        )

        return debiased_train_inds

def compute_accuracy(model, dataloader, device="cuda"):
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k:batch[k].to(device) for k in batch.keys()}
            outputs = model(**batch)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(pred)
            true_labels.extend(batch['labels'].cpu().numpy())

    return sum(np.array(true_labels) == np.array(predictions) ) / len(predictions)

def compute_predictions(model, dataloader, device="cuda"):
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k:batch[k].to(device) for k in batch.keys()}
            outputs = model(**batch)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(pred)

    return predictions
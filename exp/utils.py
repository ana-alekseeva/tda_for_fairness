import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
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
from typing import List, Optional, Tuple
from torch import nn

from tqdm import tqdm
from trak import TRAKer




class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_datasets(path_to_data="../../data/toxigen/"):
    annotated_test = pd.read_csv(path_to_data+"annotated_test.csv")
    annotated_train = pd.read_csv(path_to_data+"annotated_train.csv")

    annotated_test["target_group"] = annotated_test["target_group"].replace('black folks / african-americans', 'black/african-american folks')
    annotated_test['label'] = 1*(annotated_test['toxicity_human'] > 2)
    annotated_train['label'] = 1*(annotated_train['toxicity_human'] > 2)

    annotated_train["text"] = [i[2:] for i in annotated_train["text"]]
    annotated_train = annotated_train.dropna(subset=["text","label"])
    annotated_test = annotated_test.dropna(subset=["text","label"])
    annotated_train = annotated_train.astype(str)
    annotated_test = annotated_test.astype(str)

    annotated_train["label"] = annotated_train["label"].replace({"hate":1,"neutral":0}).astype(int)
    annotated_test["label"] = annotated_test["label"].astype(int)

    return annotated_train, annotated_test

def get_dataloader(data, tokenizer, max_length, batch_size):

    texts = data["text"].tolist()
    labels = data['label'].to_list()
    dataset = TextClassificationDataset(texts, labels, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


class FirstModuleBaseline():
    def __init__(self,train_texts, test_texts,model, tokenizer):
        self.train_texts = train_texts
        self.test_texts = test_texts
        self.model = model
        self.tokenizer = tokenizer
    
    def get_embeddings(self, texts):
        # Set model to evaluation mode
        self.model.eval()

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        # Get the model's output
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        # Extract the hidden states 
        hidden_states = outputs.hidden_states
        return hidden_states[-1][:, 0, :]


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

        return torch.from_numpy(
                np.vstack(
                    [bm25.get_scores(query) for query in test_texts_preprocessed],
                ))

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
        
        return scores

    def get_gradient_scores(self,val_labels):
        
        # Set the model to evaluation mode and enable gradient computation
        self.model.eval()
        self.model.zero_grad()

        # Prepare the training sample
        train_encoding = self.tokenizer(self.train_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        train_input_ids = train_encoding['input_ids']
        train_attention_mask = train_encoding['attention_mask']

        # Prepare the test sample
        test_encoding = self.tokenizer(self.test_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        test_input_ids = test_encoding['input_ids']
        test_attention_mask = test_encoding['attention_mask']

        # Enable gradient computation for the training sample
        train_input_ids.requires_grad_()

        # Forward pass for the training sample
        train_outputs = self.model(input_ids=train_input_ids, attention_mask=train_attention_mask)
        train_logits = train_outputs.logits

        # Forward pass for the test sample
        test_outputs = self.model(input_ids=test_input_ids, attention_mask=test_attention_mask)
        test_logits = test_outputs.logits

        # Compute the loss for the test sample
        loss_fn = torch.nn.CrossEntropyLoss()
        test_loss = loss_fn(test_logits, val_labels)

        # Compute gradients of the test loss with respect to the training logits
        test_loss.backward()

        # The gradients are now stored in train_input_ids.grad
        gradients = train_input_ids.grad
        # You can also compute the gradient norm if needed
        #gradient_norm = torch.norm(gradients)


class FirstModuleTDA():
    def __init__(self,train_texts, test_texts,model,tokenizer):
        self.train_dataloader = train_texts
        self.test_dataloader = test_texts
        self.model = model
        self.tokenizer = tokenizer        

    def get_IF_scores(self):
        class ClassificationTask(Task):
            def compute_train_loss(
                self,
                batch: Tuple[torch.Tensor, torch.Tensor],
                model: nn.Module,
                sample: bool = False,
            ) -> torch.Tensor:
                inputs, targets = batch
                outputs = model(inputs)
                if not sample:
                    return F.cross_entropy(outputs, targets)
                # Sample the outputs from the model's prediction for true Fisher.
                with torch.no_grad():
                    sampled_targets = torch.normal(outputs, std=math.sqrt(0.5))
                return F.cross_entropy(outputs, sampled_targets.detach())

            def compute_measurement(
                self,
                batch: Tuple[torch.Tensor, torch.Tensor],
                model: nn.Module,
            ) -> torch.Tensor:
                # The measurement function is set as a training loss.
                return self.compute_train_loss(batch, model, sample=False)

            def tracked_modules(self) -> Optional[List[str]]:
                # These are the module names we will use to compute influence functions.
                return ["0", "2", "4", "6"]
        task = ClassificationTask()
        model = prepare_model(model, task)
        analyzer = Analyzer(
                    analysis_name="classification",
                    model=model,
                    task=task,
                    cpu=True,
                )
        
        analyzer.fit_all_factors(
                    factors_name="classification_factor",
                    dataset=self.train_texts,
                    per_device_batch_size=None,
                    overwrite_output_dir=True,
                )
        
        analyzer.compute_pairwise_scores(
                    scores_name="classification_score",
                    factors_name="classification_factor",
                    query_dataset=self.test_texts,
                    train_dataset=self.train_texts,
                    per_device_query_batch_size=len(self.test_texts),
                    overwrite_output_dir=True,
                )
        scores = analyzer.load_pairwise_scores(scores_name="classification_score")['all_modules']

        return scores

    def get_TRAK_scores(self):

        def process_batch(batch):
            return batch['input_ids'], batch['token_type_ids'], batch['attention_mask'], batch['labels']

        out = PATH + "/output"

        device = 'cuda'
        model = self.model.to(device)

        traker = TRAKer(model=self.model,
                        task='text_classification',
                        train_set_size=TRAIN_SET_SIZE,
                        save_dir=out,
                        device=device,
                        proj_dim=1024)

        traker.load_checkpoint(self.model.state_dict(), model_id=0)
        for batch in tqdm(self.train_dataloader, desc='Featurizing..'):
            # process batch into compatible form for TRAKer TextClassificationModelOutput
            batch = process_batch(batch)
            batch = [x.cuda() for x in batch]
            traker.featurize(batch=batch, num_samples=batch[0].shape[0])

        traker.finalize_features()

        traker.start_scoring_checkpoint(exp_name='text_classification',
                                        checkpoint=model.state_dict(),
                                        model_id=0,
                                        num_targets=VAL_SET_SIZE)
        
        for batch in tqdm(self.test_dataloader, desc='Scoring..'):
            batch = process_batch(batch)
            batch = [x.cuda() for x in batch]
            traker.score(batch=batch, num_samples=batch[0].shape[0])

        scores = traker.finalize_scores(exp_name='text_classification')

        return scores



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
        self.model = model
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
                labels = batch["labels"]

                outputs = model(**batch)["logits"]
                loss = F.cross_entropy(
                    outputs, labels.to(self.device), reduction="none"
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


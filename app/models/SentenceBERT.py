# import packages
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import datasets
from sklearn.metrics import classification_report
import time
import os
import re

class CustomTokenizer:
    def __init__(self, word2id):
        self.word2id = word2id
        # Define special token IDs
        self.pad_id = word2id.get("[PAD]", 0)
        self.cls_id = word2id.get("[CLS]", 1)
        self.sep_id = word2id.get("[SEP]", 2)
        self.mask_id = word2id.get("[MASK]", 3)
        # If '[UNK]' exists in vocabulary, use it; otherwise fallback to pad_id
        self.unk_id = word2id.get("[UNK]", self.pad_id)

    def tokenize(self, text):
        sent = text.lower()
        sent = re.sub(r"[.,!?\\-]", " ", sent)  # clean sentence
        tks = sent.split()  # split the sentence to tokens
        return tks

    def encode(self, text, max_length=128, padding=True, truncation=True):
        """Convert text to a list of token IDs with [CLS] and [SEP]."""
        tokens = self.tokenize(text)
        if truncation:
            tokens = tokens[: max_length - 2]  # leave room for [CLS] and [SEP]
        ids = (
            [self.cls_id]
            + [self.word2id.get(tok, self.unk_id) for tok in tokens]
            + [self.sep_id]
        )
        if padding:
            pad_len = max_length - len(ids)
            if pad_len > 0:
                ids += [self.pad_id] * pad_len
        return ids

    def __call__(
        self, texts, max_length=128, padding=True, truncation=True, return_tensors="pt"
    ):
        """Batch encode texts. Returns a dict with 'input_ids' and 'attention_mask'."""
        batch_ids = []
        for text in texts:
            ids = self.encode(
                text, max_length=max_length, padding=padding, truncation=truncation
            )
            batch_ids.append(ids)

        input_ids = torch.tensor(batch_ids, dtype=torch.long)
        attention_mask = (input_ids != self.pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
def mean_pooling(token_embeddings, attention_mask):
    """Mean pooling for sentence embeddings"""
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class SentenceBERT(nn.Module):
    """Siamese network structure for Sentence-BERT"""

    def __init__(self, bert_model, hidden_dim=768, num_classes=3):
        super(SentenceBERT, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(hidden_dim * 3, num_classes)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        # Get embeddings from both sentences
        emb_a = self.get_sentence_embedding(input_ids_a, attention_mask_a)
        emb_b = self.get_sentence_embedding(input_ids_b, attention_mask_b)

        # Configuration
        # Concatenate features: u, v, |u-v|
        diff = torch.abs(emb_a - emb_b)
        features = torch.cat([emb_a, emb_b, diff], dim=-1)

        # Classification
        logits = self.classifier(features)
        return logits

    def get_sentence_embedding(self, input_ids, attention_mask):
        """Extract sentence embedding from BERT"""
        # Get BERT outputs
        outputs, _ = self.bert(input_ids, torch.zeros_like(input_ids))

        # Mean pooling
        sentence_embedding = mean_pooling(outputs, attention_mask)
        return sentence_embedding

    def encode(self, input_ids, attention_mask):
        """Encode sentence for similarity computation"""
        with torch.no_grad():
            return self.get_sentence_embedding(input_ids, attention_mask)
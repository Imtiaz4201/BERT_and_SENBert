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


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model=768, max_len=512, n_segments=2):
        super(Embedding, self).__init__()
        # embedding matrix: maps token ids -> vectors
        self.tok_embed = nn.Embedding(
            vocab_size, d_model
        )  # (V, D) ; lookup on input (B, L) -> (B, L, D)
        # positional embedding matrix: maps position idx -> vectors
        self.pos_embed = nn.Embedding(
            max_len, d_model
        )  # (M, D) ; pos embedding for positions 0..L-1 -> (B, L, D)

        # segment (token type) embedding: maps segment id -> vectors
        self.seg_embed = nn.Embedding(
            n_segments, d_model
        )  # (S, D) ; seg lookup on seg ids (B, L) -> (B, L, D)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        x: input ids shape (B, L)
        seg: segment ids shape (B, L)
        """
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
        # K.transpose(-1, -2): (B, H, d_k, L_k)
        # Q:                   (B, H, L_q, d_k)
        # matmul result:       (B, H, L_q, L_k)
        # scores:              (B, H, L_q, L_k)
        scores.masked_fill_(attn_mask, -1e9)
        # attn_mask:           (B, H, L_q, L_k)
        # scores unchanged shape: (B, H, L_q, L_k)
        attn = nn.Softmax(dim=-1)(scores)  # (B, H, L_q, L_k)
        context = torch.matmul(attn, V)  # (B, H, L_q, d_v)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=12, d_k=64, d_v=64):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads  # H
        self.d_k = d_k  # d_k
        self.d_v = d_v  # d_v

        # linear projections for Q, K, V
        # W_Q: projects (B, L, D) -> (B, L, H * d_k)
        # weight shape: (D, H*d_k) ; bias shape: (H*d_k,)
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # Q, K, V: (B, L, D)
        residual = Q
        batch_size = Q.size(0)  # B

        q_s = (
            self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        )  # q_s: (B, H, L, d_k)
        k_s = (
            self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        )  # k_s: (B, H, L, d_k)
        v_s = (
            self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        )  # v_s: (B, H, L, d_v)

        attn_mask = attn_mask.unsqueeze(1).repeat(
            1, self.n_heads, 1, 1
        )  # (B, H, L_q, L_k)

        context, attn = ScaledDotProductAttention()(
            q_s, k_s, v_s, attn_mask
        )  # context: (B, H, L, d_v) ; attn: (B, H, L, L)

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_v)
        )  # (B, L, H * d_v)

        output = self.linear(context)

        return (
            self.norm(output + residual),
            attn,
        )  # output: (B, L, D), attn: (B, H, L, L)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model=768, d_ff=3072):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model=768, n_heads=12, d_ff=3072):
        super(EncoderLayer, self).__init__()
        # self-attention sublayer
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        # feed-forward sublayer
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask
        )
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class BERT(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12, n_heads=12, max_len=512):
        super(BERT, self).__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.embedding = Embedding(vocab_size, d_model, max_len)  #  (B, L, D)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads) for _ in range(n_layers)]
        )

        # MLM head
        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        # NSP head
        self.nsp_classifier = nn.Linear(d_model, 2)

        # MLM decoder (tied to embedding)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.decoder.weight = self.embedding.tok_embed.weight
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, input_ids, segment_ids, masked_pos=None):
        # Embedding
        output = self.embedding(input_ids, segment_ids)  # (B, L, D)

        # Attention mask
        attn_mask = get_attn_pad_mask(input_ids, input_ids)  # (B, L, L)

        # Encoder layers
        for layer in self.layers:
            output, _ = layer(output, attn_mask)

        # NSP prediction (using [CLS] token)
        cls_output = output[:, 0, :]  # (B, D)
        nsp_logits = self.nsp_classifier(cls_output)  # (B, 2)

        # MLM prediction (if masked_pos provided)
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(
                -1, -1, self.d_model
            )  # (B, N_mask, D)
            h_masked = torch.gather(output, 1, masked_pos)  # (B, N_mask, D)
            h_masked = self.norm(
                torch.nn.functional.gelu(self.linear(h_masked))
            )  # (B, N_mask, D)
            mlm_logits = self.decoder(h_masked) + self.decoder_bias  # (B, N_mask, V)
            return mlm_logits, nsp_logits

        return output, nsp_logits

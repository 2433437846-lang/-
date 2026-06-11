#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bi-LSTM模型定义
- Bidirectional LSTM with Embedding + Dropout + FC
- 支持torchsummary模型结构打印
- Embedding（dim=128）+ Dropout
- 2 层双向 LSTM（hidden_dim=256）
- FC 分类头：Dropout → Linear(512,256) → ReLU → Dropout → Linear(256,2)
- 自定义正则分词器，词表大小 10,038，最大序列长度 256
"""

import torch
import torch.nn as nn
from config import LSTM_CONFIG


class BiLSTMClassifier(nn.Module):
    """双向LSTM文本分类器"""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, num_classes=2, dropout=0.5, bidirectional=True):
        super(BiLSTMClassifier, self).__init__()

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        # LSTM层
        lstm_dropout = dropout if num_layers > 1 else 0
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout,
        )

        # 全连接分类头
        fc_input_dim = hidden_dim * self.num_directions
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, seq_len) token indices
            lengths: (batch,) 实际序列长度（用于pack_padded_sequence）
        Returns:
            logits: (batch, num_classes)
        """
        # Embedding
        embedded = self.embed_dropout(self.embedding(x))  # (batch, seq_len, embed_dim)

        # LSTM
        if lengths is not None:
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            # 按长度排序（pack_padded_sequence要求）
            sorted_lengths, sorted_idx = lengths.sort(descending=True)
            embedded = embedded[sorted_idx]
            packed = pack_padded_sequence(embedded, sorted_lengths.cpu(), batch_first=True)
            output, (hidden, cell) = self.lstm(packed)
            output, _ = pad_packed_sequence(output, batch_first=True)
            # 恢复原始顺序
            _, unsorted_idx = sorted_idx.sort()
            output = output[unsorted_idx]
            hidden = hidden[:, unsorted_idx, :]
        else:
            output, (hidden, cell) = self.lstm(embedded)

        # 取最后一层的双向隐藏状态拼接
        if self.bidirectional:
            # hidden shape: (num_layers*2, batch, hidden_dim)
            last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch, 2*hidden_dim)
        else:
            last_hidden = hidden[-1]  # (batch, hidden_dim)

        # 分类
        logits = self.fc(last_hidden)  # (batch, num_classes)
        return logits


def build_lstm_model(vocab_size, config=None):
    """构建Bi-LSTM模型并打印结构"""
    if config is None:
        config = LSTM_CONFIG

    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional'],
    )

    # 打印模型结构
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[模型] Bi-LSTM 结构")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

    return model


if __name__ == '__main__':
    # 测试模型
    model = build_lstm_model(vocab_size=20000)
    print(model)

    # 测试前向传播
    x = torch.randint(0, 20000, (4, 256))
    lengths = torch.tensor([256, 200, 180, 150])
    out = model(x, lengths)
    print(f"\n  输入: {x.shape}")
    print(f"  输出: {out.shape}")

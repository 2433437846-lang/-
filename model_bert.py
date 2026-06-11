#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BERT Transformer模型定义
- 基于HuggingFace bert-base-uncased预训练模型
- 用于文本分类微调
- `bert-base-uncased`，12 层 Transformer，12 个注意力头，hidden_size=768
- HuggingFace BertTokenizer，最大序列长度 128
- Warmup + Cosine Decay 学习率调度（10% warmup）
"""

import torch
import torch.nn as nn
from config import BERT_CONFIG, BERT_MODEL_NAME


def build_bert_model(config=None):
    """构建BERT分类模型并打印结构"""
    from transformers import BertForSequenceClassification

    if config is None:
        config = BERT_CONFIG

    print(f"\n[模型] 加载 BERT: {BERT_MODEL_NAME}")
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=config['num_labels'],
    )

    # 打印模型结构
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    print(f"  隐藏层维度: {model.config.hidden_size}")
    print(f"  注意力头数: {model.config.num_attention_heads}")
    print(f"  Transformer层数: {model.config.num_hidden_layers}")

    return model


def get_bert_attention(model, input_ids, attention_mask):
    """获取BERT注意力权重（用于可视化）"""
    model.eval()
    # 强制使用eager attention以支持output_attentions
    original_attn = model.config.attn_implementation
    model.config.attn_implementation = 'eager'
    with torch.no_grad():
        outputs = model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
    model.config.attn_implementation = original_attn
    # attentions: tuple of (batch, num_heads, seq_len, seq_len) per layer
    return outputs.attentions


if __name__ == '__main__':
    model = build_bert_model()

    # 测试前向传播
    input_ids = torch.randint(0, 30522, (2, 128))
    attention_mask = torch.ones(2, 128, dtype=torch.long)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    print(f"\n  输入: {input_ids.shape}")
    print(f"  输出logits: {outputs.logits.shape}")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据工程模块：
- 数据加载与清洗
- 标签构建（score → 二分类）
- 数据划分（Train/Val/Test）
- 分词与词表构建
- DataLoader封装
- 类别分布可视化
"""

import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch
from config import (
    DATA_PATH, TEXT_COL, LABEL_COL, SCORE_COL, POSITIVE_THRESHOLD,
    TRAIN_RATIO, VAL_RATIO, SEED, LSTM_CONFIG, BERT_CONFIG, BERT_MODEL_NAME
)
from config import setup_chinese_font

plt = setup_chinese_font()


# ==================== 数据加载与清洗 ====================
def load_and_clean_data(path=DATA_PATH):
    """加载CSV数据并清洗"""
    print("=" * 60)
    print("模块1：数据加载与清洗")
    print("=" * 60)

    # 加载
    df = pd.read_csv(path)
    print(f"\n[1.1] 原始数据: {len(df)} 条")
    print(f"  列名: {list(df.columns)}")

    # 去除overview为空的行
    df = df.dropna(subset=[TEXT_COL]).reset_index(drop=True)
    print(f"  去除空overview后: {len(df)} 条")

    # 文本清洗：去除HTML标签、多余空格（保留标点和特殊字符供BERT使用）
    def clean_text(text):
        text = str(text)
        text = re.sub(r'<[^>]+>', '', text)         # HTML标签
        text = re.sub(r'\s+', ' ', text).strip()     # 多余空格
        return text

    df[TEXT_COL] = df[TEXT_COL].apply(clean_text)
    # 去除清洗后为空的行
    df = df[df[TEXT_COL].str.len() > 0].reset_index(drop=True)
    print(f"  文本清洗后: {len(df)} 条")

    # 构建二分类标签
    df[LABEL_COL] = (df[SCORE_COL] >= POSITIVE_THRESHOLD).astype(int)

    # 统计类别分布
    pos_count = (df[LABEL_COL] == 1).sum()
    neg_count = (df[LABEL_COL] == 0).sum()
    print(f"\n[1.2] 标签构建 (score >= {POSITIVE_THRESHOLD} → 正面)")
    print(f"  正面样本: {pos_count} ({pos_count/len(df)*100:.1f}%)")
    print(f"  负面样本: {neg_count} ({neg_count/len(df)*100:.1f}%)")

    # 文本长度统计
    text_lengths = df[TEXT_COL].str.split().str.len()
    print(f"\n[1.3] 文本长度统计（词数）")
    print(f"  平均: {text_lengths.mean():.1f}, 中位数: {text_lengths.median():.0f}")
    print(f"  最短: {text_lengths.min()}, 最长: {text_lengths.max()}")

    return df


# ==================== 数据划分 ====================
def split_data(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """分层抽样划分 Train / Val / Test"""
    from sklearn.model_selection import train_test_split

    # 先分出train和temp
    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio), random_state=SEED,
        stratify=df[LABEL_COL]
    )
    # 再从temp中分出val和test
    val_ratio_in_temp = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio_in_temp), random_state=SEED,
        stratify=temp_df[LABEL_COL]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"\n[1.4] 数据划分 (Train/Val/Test)")
    print(f"  训练集: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  验证集: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  测试集: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    # 各子集类别分布
    for name, subset in [('训练集', train_df), ('验证集', val_df), ('测试集', test_df)]:
        p = (subset[LABEL_COL] == 1).sum()
        n = (subset[LABEL_COL] == 0).sum()
        print(f"  {name}: 正面={p}, 负面={n}")

    return train_df, val_df, test_df


# ==================== 类别分布可视化 ====================
def plot_class_distribution(df, train_df, val_df, test_df, save_dir='outputs'):
    """绘制类别分布图"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    datasets = [('全量数据', df), ('训练集', train_df),
                ('验证集', val_df), ('测试集', test_df)]

    for ax, (name, data) in zip(axes, datasets):
        counts = data[LABEL_COL].value_counts().sort_index()
        bars = ax.bar(['负面', '正面'], [counts.get(0, 0), counts.get(1, 0)],
                      color=['#e74c3c', '#2ecc71'])
        ax.set_title(name)
        ax.set_ylabel('样本数')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('数据集类别分布', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'class_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  类别分布图已保存: {save_path}")
    return save_path


# ==================== LSTM专用：自定义Tokenizer ====================
class TextTokenizer:
    """基于正则的英文分词器（参考deepstudy10）"""

    def __init__(self, max_vocab_size=20000, min_freq=3):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq

    def tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)
        return tokens

    def build_vocab(self, texts):
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)

        print(f"  原始词汇总数: {len(word_counts)}")
        sorted_words = [(w, c) for w, c in word_counts.items() if c >= self.min_freq]
        sorted_words.sort(key=lambda x: x[1], reverse=True)

        for word, _ in sorted_words[:self.max_vocab_size - 2]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"  最终词表大小: {len(self.word2idx)}")
        return len(self.word2idx)

    def encode(self, text, max_len=256):
        tokens = self.tokenize(text)
        indices = [self.word2idx.get(t, self.word2idx['<UNK>']) for t in tokens]
        if len(indices) > max_len:
            indices = indices[:max_len]
        return indices


# ==================== LSTM专用：Dataset ====================
class LSTMDataset(Dataset):
    """LSTM数据集：返回(token_indices, length, label)"""

    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        indices = self.tokenizer.encode(self.texts[idx], self.max_len)
        length = min(len(self.tokenizer.tokenize(self.texts[idx])), self.max_len)
        return (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


def lstm_collate_fn(batch):
    """LSTM的collate_fn：动态padding"""
    indices, lengths, labels = zip(*batch)
    # 按最长序列padding
    max_len = max(idx.size(0) for idx in indices)
    padded = torch.zeros(len(indices), max_len, dtype=torch.long)
    for i, idx in enumerate(indices):
        padded[i, :idx.size(0)] = idx
    lengths = torch.stack(lengths)
    labels = torch.stack(labels)
    return padded, lengths, labels


# ==================== BERT专用：Dataset ====================
class BERTDataset(Dataset):
    """BERT数据集：返回tokenizer编码结果"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ==================== 类别权重计算 ====================
def compute_class_weights(labels):
    """计算类别权重，用于处理类别不平衡"""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"\n  类别权重: 正面={class_weights[1]:.4f}, 负面={class_weights[0]:.4f}")
    return class_weights


# ==================== 构建DataLoader ====================
def build_lstm_dataloaders(train_df, val_df, test_df, config=None):
    """构建LSTM的DataLoader"""
    if config is None:
        config = LSTM_CONFIG

    print("\n" + "=" * 60)
    print("模块2A：LSTM数据准备")
    print("=" * 60)

    # 构建词表
    print("\n[2A.1] 构建词表")
    tokenizer = TextTokenizer(
        max_vocab_size=config['max_vocab_size'],
        min_freq=config['min_freq']
    )
    vocab_size = tokenizer.build_vocab(train_df[TEXT_COL].tolist())

    # 构建Dataset
    print(f"\n[2A.2] 构建Dataset (max_len={config['max_len']})")
    train_ds = LSTMDataset(train_df[TEXT_COL].tolist(), train_df[LABEL_COL].tolist(),
                           tokenizer, config['max_len'])
    val_ds = LSTMDataset(val_df[TEXT_COL].tolist(), val_df[LABEL_COL].tolist(),
                         tokenizer, config['max_len'])
    test_ds = LSTMDataset(test_df[TEXT_COL].tolist(), test_df[LABEL_COL].tolist(),
                          tokenizer, config['max_len'])

    # 构建DataLoader
    batch_size = config.get('batch_size', 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=lstm_collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=lstm_collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=lstm_collate_fn, num_workers=0)

    # 验证一个batch
    batch = next(iter(train_loader))
    print(f"\n[2A.3] DataLoader验证")
    print(f"  input shape: {batch[0].shape}")
    print(f"  length shape: {batch[1].shape}")
    print(f"  label shape: {batch[2].shape}")

    return train_loader, val_loader, test_loader, tokenizer, vocab_size


def build_bert_dataloaders(train_df, val_df, test_df, config=None):
    """构建BERT的DataLoader"""
    from transformers import BertTokenizer

    if config is None:
        config = BERT_CONFIG

    print("\n" + "=" * 60)
    print("模块2B：BERT数据准备")
    print("=" * 60)

    # 加载Tokenizer
    print(f"\n[2B.1] 加载Tokenizer: {BERT_MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    print(f"  词表大小: {tokenizer.vocab_size}")

    # 构建Dataset
    max_length = config['max_length']
    print(f"\n[2B.2] 构建Dataset (max_length={max_length})")
    train_ds = BERTDataset(train_df[TEXT_COL].tolist(), train_df[LABEL_COL].tolist(),
                           tokenizer, max_length)
    val_ds = BERTDataset(val_df[TEXT_COL].tolist(), val_df[LABEL_COL].tolist(),
                         tokenizer, max_length)
    test_ds = BERTDataset(test_df[TEXT_COL].tolist(), test_df[LABEL_COL].tolist(),
                          tokenizer, max_length)

    # 构建DataLoader
    batch_size = config.get('batch_size', 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 验证一个batch
    batch = next(iter(train_loader))
    print(f"\n[2B.3] DataLoader验证")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  label shape: {batch['label'].shape}")

    return train_loader, val_loader, test_loader, tokenizer


# ==================== 主函数（独立测试）====================
if __name__ == '__main__':
    from config import set_seed
    set_seed()

    # 加载数据
    df = load_and_clean_data()
    train_df, val_df, test_df = split_data(df)
    plot_class_distribution(df, train_df, val_df, test_df)

    # 测试LSTM DataLoader
    train_loader, val_loader, test_loader, tokenizer, vocab_size = \
        build_lstm_dataloaders(train_df, val_df, test_df)

    # 测试BERT DataLoader
    train_loader_b, val_loader_b, test_loader_b, tokenizer_b = \
        build_bert_dataloaders(train_df, val_df, test_df)

    print("\n数据加载测试完成！")

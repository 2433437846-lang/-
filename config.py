#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置模块：超参数、路径、设备设置
"""

import os
import torch
import numpy as np
import random

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'imdb_movies.csv')
RUNS_DIR = os.path.join(BASE_DIR, 'runs')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# HuggingFace镜像（国内加速）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ==================== 设备配置 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

# ==================== 随机种子 ====================
SEED = 42

def set_seed(seed=SEED):
    """固定随机种子，确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== 数据配置 ====================
TEXT_COL = 'overview'       # 文本列
LABEL_COL = 'label'         # 标签列
SCORE_COL = 'score'         # 评分列（用于构建标签）
POSITIVE_THRESHOLD = 70     # score >= 70 为正面

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==================== Bi-LSTM 配置 ====================
LSTM_CONFIG = {
    'max_vocab_size': 20000,
    'min_freq': 3,
    'max_len': 256,
    'embed_dim': 128,
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.5,
    'bidirectional': True,
    'num_classes': 2,
}

LSTM_TRAIN_CONFIG = {
    'batch_size': 64,
    'epochs': 15,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'patience': 3,
    'optimizer': 'AdamW',
}

# ==================== BERT 配置 ====================
BERT_MODEL_NAME = 'bert-base-uncased'
BERT_CONFIG = {
    'max_length': 256,
    'num_labels': 2,
}

BERT_TRAIN_CONFIG = {
    'batch_size': 16,
    'epochs': 6,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'patience': 3,
}

# ==================== 超参数对比实验配置 ====================
LSTM_HYPERPARAM_EXPERIMENTS = [
    {'name': 'lr=1e-3_bs64', 'learning_rate': 1e-3, 'batch_size': 64, 'optimizer': 'AdamW'},
    {'name': 'lr=5e-4_bs64', 'learning_rate': 5e-4, 'batch_size': 64, 'optimizer': 'AdamW'},
    {'name': 'lr=1e-3_bs32_Adam', 'learning_rate': 1e-3, 'batch_size': 32, 'optimizer': 'Adam'},
]

BERT_HYPERPARAM_EXPERIMENTS = [
    {'name': 'lr=2e-5_bs16', 'learning_rate': 2e-5, 'batch_size': 16},
    {'name': 'lr=3e-5_bs16', 'learning_rate': 3e-5, 'batch_size': 16},
    {'name': 'lr=5e-5_bs16', 'learning_rate': 5e-5, 'batch_size': 16},
]

# ==================== 中文字体配置 ====================
def setup_chinese_font():
    """设置matplotlib中文字体"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return plt

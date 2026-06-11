#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
深度学习期末综合实践 - 方向二
融合LSTM与Transformer/BERT的RTX 4060加速序列系统
====================================================

任务：电影简介情感分析（二分类）
数据集：IMDB Movies CSV
模型：Bi-LSTM vs BERT (HuggingFace)

执行流程：
  阶段① 数据准备：加载、清洗、划分、可视化
  阶段② 模型构建：Bi-LSTM + BERT
  阶段③ 模型训练：FP32 + AMP对比
  阶段④ 超参数对比：≥3组实验
  阶段⑤ 评估对比：测试集指标、混淆矩阵、Attention热力图
  阶段⑥ 报告生成：GPU Profile表、实验报告
"""

import os
import sys
import time
import torch
import numpy as np

# 确保当前目录在path中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    set_seed, DEVICE, GPU_NAME, SEED,
    LSTM_CONFIG, LSTM_TRAIN_CONFIG,
    BERT_CONFIG, BERT_TRAIN_CONFIG,
    LSTM_HYPERPARAM_EXPERIMENTS, BERT_HYPERPARAM_EXPERIMENTS,
    CHECKPOINT_DIR, OUTPUT_DIR, RUNS_DIR,
)
from data_loader import (
    load_and_clean_data, split_data, plot_class_distribution,
    build_lstm_dataloaders, build_bert_dataloaders, compute_class_weights,
)
from model_lstm import build_lstm_model, BiLSTMClassifier
from model_bert import build_bert_model
from train import train_lstm, train_bert, run_amp_comparison
from evaluate import (
    evaluate_test, plot_confusion_matrix, plot_training_comparison,
    plot_bert_attention, plot_gpu_profile_table, plot_hyperparam_table,
    plot_final_comparison, generate_report,
)


def main():
    """主函数：完整实验流程"""
    print("=" * 70)
    print("  深度学习期末综合实践 - 方向二")
    print("  融合LSTM与Transformer/BERT的RTX 4060加速序列系统")
    print("=" * 70)

    # 固定随机种子
    set_seed(SEED)

    # 设备信息
    print(f"\n  设备: {DEVICE} ({GPU_NAME})")
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  PyTorch版本: {torch.__version__}")
        print(f"  GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 创建输出目录
    for d in [CHECKPOINT_DIR, OUTPUT_DIR, RUNS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ============================================================
    # 阶段① 数据准备
    # ============================================================
    print("\n" + "▶" * 30)
    print("  阶段①：数据准备")
    print("▶" * 30)

    df = load_and_clean_data()
    train_df, val_df, test_df = split_data(df)
    plot_class_distribution(df, train_df, val_df, test_df, save_dir=OUTPUT_DIR)

    # 构建LSTM DataLoader
    lstm_train_loader, lstm_val_loader, lstm_test_loader, lstm_tokenizer, vocab_size = \
        build_lstm_dataloaders(train_df, val_df, test_df)

    # 构建BERT DataLoader
    bert_train_loader, bert_val_loader, bert_test_loader, bert_tokenizer = \
        build_bert_dataloaders(train_df, val_df, test_df)

    # ============================================================
    # 阶段② 模型构建
    # ============================================================
    print("\n" + "▶" * 30)
    print("  阶段②：模型构建")
    print("▶" * 30)

    lstm_model = build_lstm_model(vocab_size)
    bert_model = build_bert_model()

    # ============================================================
    # 阶段③ 模型训练（FP32）
    # ============================================================
    print("\n" + "▶" * 30)
    print("  阶段③：模型训练（FP32）")
    print("▶" * 30)

    # Bi-LSTM 训练
    lstm_config = {**LSTM_TRAIN_CONFIG, 'batch_size': LSTM_TRAIN_CONFIG['batch_size']}
    lstm_results = train_lstm(lstm_model, lstm_train_loader, lstm_val_loader,
                               config=lstm_config, use_amp=False)
    lstm_results['total_params'] = sum(p.numel() for p in lstm_model.parameters())
    lstm_results['data_size'] = len(df)
    lstm_results['gpu'] = GPU_NAME

    # 计算类别权重（处理类别不平衡）
    class_weights = compute_class_weights(train_df['label'].tolist())

    # BERT 训练
    bert_config = {**BERT_TRAIN_CONFIG, 'batch_size': BERT_TRAIN_CONFIG['batch_size']}
    bert_results = train_bert(bert_model, bert_train_loader, bert_val_loader,
                               config=bert_config, use_amp=False, class_weights=class_weights)
    bert_results['total_params'] = sum(p.numel() for p in bert_model.parameters())
    bert_results['data_size'] = len(df)
    bert_results['gpu'] = GPU_NAME

    # ============================================================
    # 阶段③续 AMP混合精度对比
    # ============================================================
    print("\n" + "▶" * 30)
    print("  阶段③续：AMP混合精度训练对比")
    print("▶" * 30)

    # Bi-LSTM AMP对比
    print("\n--- Bi-LSTM AMP vs FP32 ---")
    lstm_amp_model = BiLSTMClassifier(vocab_size=vocab_size,
                                       embed_dim=LSTM_CONFIG['embed_dim'],
                                       hidden_dim=LSTM_CONFIG['hidden_dim'],
                                       num_layers=LSTM_CONFIG['num_layers'],
                                       dropout=LSTM_CONFIG['dropout'])
    lstm_amp_results = train_lstm(lstm_amp_model, lstm_train_loader, lstm_val_loader,
                                   config={**lstm_config, 'epochs': 5, 'patience': 5},
                                   use_amp=True)
    lstm_amp_results['total_params'] = sum(p.numel() for p in lstm_amp_model.parameters())
    lstm_results['fp32_mem'] = lstm_results['peak_memory_mb']
    lstm_results['amp_mem'] = lstm_amp_results['peak_memory_mb']

    # BERT AMP对比
    print("\n--- BERT AMP vs FP32 ---")
    bert_amp_model = build_bert_model()
    bert_amp_results = train_bert(bert_amp_model, bert_train_loader, bert_val_loader,
                                   config={**bert_config, 'epochs': 2, 'patience': 2},
                                   use_amp=True, class_weights=class_weights)
    bert_amp_results['total_params'] = sum(p.numel() for p in bert_amp_model.parameters())
    bert_results['fp32_mem'] = bert_results['peak_memory_mb']
    bert_results['amp_mem'] = bert_amp_results['peak_memory_mb']

    # GPU Profile表
    plot_gpu_profile_table(lstm_results, lstm_amp_results,
                           bert_results, bert_amp_results, save_dir=OUTPUT_DIR)

    # ============================================================
    # 阶段④ 超参数对比实验
    # ============================================================
    print("\n" + "▶" * 30)
    print("  阶段④：超参数对比实验（≥3组）")
    print("▶" * 30)

    # Bi-LSTM 超参数实验
    print("\n--- Bi-LSTM 超参数实验 ---")
    lstm_exp_results = []
    for exp_cfg in LSTM_HYPERPARAM_EXPERIMENTS:
        print(f"\n  实验: {exp_cfg['name']}")
        exp_model = BiLSTMClassifier(vocab_size=vocab_size,
                                      embed_dim=LSTM_CONFIG['embed_dim'],
                                      hidden_dim=LSTM_CONFIG['hidden_dim'],
                                      num_layers=LSTM_CONFIG['num_layers'],
                                      dropout=LSTM_CONFIG['dropout'])
        exp_config = {
            'batch_size': exp_cfg['batch_size'],
            'epochs': 8,
            'learning_rate': exp_cfg['learning_rate'],
            'weight_decay': 1e-4,
            'patience': 3,
            'optimizer': exp_cfg.get('optimizer', 'AdamW'),
        }
        # 重新构建DataLoader以适配不同batch_size
        from torch.utils.data import DataLoader
        from data_loader import LSTMDataset, lstm_collate_fn
        exp_train_ds = LSTMDataset(train_df['overview'].tolist(), train_df['label'].tolist(),
                                    lstm_tokenizer, LSTM_CONFIG['max_len'])
        exp_val_ds = LSTMDataset(val_df['overview'].tolist(), val_df['label'].tolist(),
                                  lstm_tokenizer, LSTM_CONFIG['max_len'])
        exp_train_loader = DataLoader(exp_train_ds, batch_size=exp_cfg['batch_size'],
                                       shuffle=True, collate_fn=lstm_collate_fn, num_workers=0)
        exp_val_loader = DataLoader(exp_val_ds, batch_size=exp_cfg['batch_size'],
                                     shuffle=False, collate_fn=lstm_collate_fn, num_workers=0)

        exp_result = train_lstm(exp_model, exp_train_loader, exp_val_loader,
                                 config=exp_config, use_amp=False)
        exp_result['name'] = exp_cfg['name']
        exp_result['learning_rate'] = exp_cfg['learning_rate']
        exp_result['batch_size'] = exp_cfg['batch_size']
        exp_result['optimizer'] = exp_cfg.get('optimizer', 'AdamW')
        lstm_exp_results.append(exp_result)

    plot_hyperparam_table(lstm_exp_results, model_name='LSTM', save_dir=OUTPUT_DIR)

    # BERT 超参数实验
    print("\n--- BERT 超参数实验 ---")
    bert_exp_results = []
    for exp_cfg in BERT_HYPERPARAM_EXPERIMENTS:
        print(f"\n  实验: {exp_cfg['name']}")
        exp_model = build_bert_model()
        exp_config = {
            'batch_size': exp_cfg['batch_size'],
            'epochs': 4,
            'learning_rate': exp_cfg['learning_rate'],
            'weight_decay': 0.01,
            'warmup_ratio': 0.1,
            'patience': 2,
        }
        from torch.utils.data import DataLoader
        from data_loader import BERTDataset
        exp_train_ds = BERTDataset(train_df['overview'].tolist(), train_df['label'].tolist(),
                                    bert_tokenizer, BERT_CONFIG['max_length'])
        exp_val_ds = BERTDataset(val_df['overview'].tolist(), val_df['label'].tolist(),
                                  bert_tokenizer, BERT_CONFIG['max_length'])
        exp_train_loader = DataLoader(exp_train_ds, batch_size=exp_cfg['batch_size'],
                                       shuffle=True, num_workers=0)
        exp_val_loader = DataLoader(exp_val_ds, batch_size=exp_cfg['batch_size'],
                                     shuffle=False, num_workers=0)

        exp_result = train_bert(exp_model, exp_train_loader, exp_val_loader,
                                 config=exp_config, use_amp=False, class_weights=class_weights)
        exp_result['name'] = exp_cfg['name']
        exp_result['learning_rate'] = exp_cfg['learning_rate']
        exp_result['batch_size'] = exp_cfg['batch_size']
        exp_result['optimizer'] = 'AdamW'
        bert_exp_results.append(exp_result)

    plot_hyperparam_table(bert_exp_results, model_name='BERT', save_dir=OUTPUT_DIR)

    # ============================================================
    # 阶段⑤ 测试集评估与可视化
    # ============================================================
    print("\n" + "▶" * 30)
    print("  阶段⑤：测试集评估与可视化")
    print("▶" * 30)

    # 加载最佳模型
    best_lstm = BiLSTMClassifier(vocab_size=vocab_size,
                                  embed_dim=LSTM_CONFIG['embed_dim'],
                                  hidden_dim=LSTM_CONFIG['hidden_dim'],
                                  num_layers=LSTM_CONFIG['num_layers'],
                                  dropout=LSTM_CONFIG['dropout'])
    best_lstm.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'best_lstm_model.pth'),
                                          map_location=DEVICE))
    best_lstm = best_lstm.to(DEVICE)

    best_bert = build_bert_model()
    best_bert.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'best_bert_model.pth'),
                                          map_location=DEVICE))
    best_bert = best_bert.to(DEVICE)

    # 测试集评估
    lstm_test = evaluate_test(best_lstm, lstm_test_loader, 'lstm', 'Bi-LSTM')
    bert_test = evaluate_test(best_bert, bert_test_loader, 'bert', 'BERT')

    # 更新结果
    lstm_results['test_acc'] = lstm_test['accuracy']
    lstm_results['test_f1'] = lstm_test['f1']
    bert_results['test_acc'] = bert_test['accuracy']
    bert_results['test_f1'] = bert_test['f1']

    # 混淆矩阵
    plot_confusion_matrix(lstm_test['labels'], lstm_test['preds'], 'Bi-LSTM', OUTPUT_DIR)
    plot_confusion_matrix(bert_test['labels'], bert_test['preds'], 'BERT', OUTPUT_DIR)

    # 训练曲线对比
    plot_training_comparison(lstm_results, bert_results, save_dir=OUTPUT_DIR)

    # BERT Attention热力图
    sample_texts = [
        "This movie was absolutely fantastic! Great acting and amazing plot.",
        "Terrible film. Waste of time. The acting was horrible.",
    ]
    for i, text in enumerate(sample_texts):
        plot_bert_attention(best_bert, bert_tokenizer, text, save_dir=OUTPUT_DIR)

    # 最终对比表
    plot_final_comparison(lstm_results, bert_results, save_dir=OUTPUT_DIR)

    # ============================================================
    # 阶段⑥ 报告生成
    # ============================================================
    print("\n" + "▶" * 30)
    print("  阶段⑥：实验报告生成")
    print("▶" * 30)

    generate_report(lstm_results, bert_results, lstm_test, bert_test, save_dir=OUTPUT_DIR)

    # ============================================================
    # 完成
    # ============================================================
    print("\n" + "=" * 70)
    print("  实验全部完成！")
    print("=" * 70)
    print(f"\n  输出目录: {OUTPUT_DIR}")
    print(f"  TensorBoard: tensorboard --logdir={RUNS_DIR}")
    print(f"  Checkpoint: {CHECKPOINT_DIR}")
    print(f"\n  生成的文件:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"    - {f}")


if __name__ == '__main__':
    main()

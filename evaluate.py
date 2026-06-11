#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估与可视化模块：
- 测试集指标（Accuracy, Precision, Recall, F1）
- 混淆矩阵
- 训练曲线
- BERT Attention热力图
- GPU Profile表格
- 超参数对比表
- 实验报告生成
"""

import os
import time
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from config import DEVICE, setup_chinese_font, OUTPUT_DIR

plt = setup_chinese_font()
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 测试集评估 ====================
def evaluate_test(model, test_loader, model_type='lstm', model_name='model'):
    """在测试集上评估模型"""
    print(f"\n[评估] 测试集评估 ({model_name})")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            if model_type == 'lstm':
                inputs, lengths, labels = batch
                inputs, lengths = inputs.to(DEVICE), lengths.to(DEVICE)
                outputs = model(inputs, lengths)
            else:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label']
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                outputs = outputs.logits

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            if model_type == 'lstm':
                all_labels.extend(labels.numpy())
            else:
                all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='binary')
    rec = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    print(f"  准确率: {acc*100:.2f}%")
    print(f"  精确率: {prec*100:.2f}%")
    print(f"  召回率: {rec*100:.2f}%")
    print(f"  F1值: {f1*100:.2f}%")

    target_names = ['负面', '正面']
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print(f"\n  分类报告:\n{report}")

    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'preds': all_preds, 'labels': all_labels
    }


# ==================== 混淆矩阵 ====================
def plot_confusion_matrix(labels, preds, model_name='model', save_dir=OUTPUT_DIR):
    """绘制混淆矩阵"""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负面', '正面'], yticklabels=['负面', '正面'], ax=ax)
    ax.set_title(f'混淆矩阵 - {model_name}')
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')

    save_path = os.path.join(save_dir, f'confusion_matrix_{model_name}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  混淆矩阵已保存: {save_path}")
    return save_path


# ==================== 训练曲线对比 ====================
def plot_training_comparison(lstm_results, bert_results, save_dir=OUTPUT_DIR):
    """绘制LSTM vs BERT训练曲线对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Loss曲线
    ax = axes[0, 0]
    ax.plot(lstm_results['train_losses'], 'b-o', label='LSTM Train', markersize=4)
    ax.plot(lstm_results['val_losses'], 'b--s', label='LSTM Val', markersize=4)
    ax.plot(bert_results['train_losses'], 'r-o', label='BERT Train', markersize=4)
    ax.plot(bert_results['val_losses'], 'r--s', label='BERT Val', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('训练/验证 Loss 对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) Accuracy曲线
    ax = axes[0, 1]
    ax.plot(lstm_results['train_accs'], 'b-o', label='LSTM Train', markersize=4)
    ax.plot(lstm_results['val_accs'], 'b--s', label='LSTM Val', markersize=4)
    ax.plot(bert_results['train_accs'], 'r-o', label='BERT Train', markersize=4)
    ax.plot(bert_results['val_accs'], 'r--s', label='BERT Val', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('训练/验证 Accuracy 对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (3) 每epoch训练耗时
    ax = axes[1, 0]
    epochs_lstm = range(1, len(lstm_results['epoch_times']) + 1)
    epochs_bert = range(1, len(bert_results['epoch_times']) + 1)
    ax.bar([e - 0.2 for e in epochs_lstm], lstm_results['epoch_times'],
           width=0.4, label='LSTM', color='steelblue')
    ax.bar([e + 0.2 for e in epochs_bert], bert_results['epoch_times'],
           width=0.4, label='BERT', color='coral')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('时间 (s)')
    ax.set_title('每Epoch训练耗时对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (4) 模型参数量对比（柱状图）
    ax = axes[1, 1]
    models = ['Bi-LSTM', 'BERT']
    # 从结果中获取参数量（如果有的话）
    lstm_params = lstm_results.get('total_params', 0)
    bert_params = bert_results.get('total_params', 0)
    if lstm_params and bert_params:
        params = [lstm_params / 1e6, bert_params / 1e6]
        bars = ax.bar(models, params, color=['steelblue', 'coral'])
        ax.set_ylabel('参数量 (M)')
        ax.set_title('模型参数量对比')
        for bar, p in zip(bars, params):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{p:.1f}M', ha='center', va='bottom')
    else:
        ax.text(0.5, 0.5, '参数量数据未提供', ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    plt.suptitle('LSTM vs BERT 训练对比', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  训练对比图已保存: {save_path}")
    return save_path


# ==================== BERT Attention热力图 ====================
def plot_bert_attention(model, tokenizer, text, save_dir=OUTPUT_DIR):
    """绘制BERT注意力热力图"""
    from model_bert import get_bert_attention

    model.eval()
    encoding = tokenizer(text, padding=True, truncation=True,
                         max_length=64, return_tensors='pt')
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    try:
        attentions = get_bert_attention(model, input_ids, attention_mask)
    except Exception as e:
        print(f"  注意力热力图跳过: {e}")
        return None

    if not attentions:
        print("  注意力热力图跳过: attentions为空")
        return None

    # 取最后一层的平均注意力
    last_layer_attn = attentions[-1]  # (batch, num_heads, seq_len, seq_len)
    avg_attn = last_layer_attn.mean(dim=1).squeeze(0).cpu().numpy()  # (seq_len, seq_len)

    # 获取token
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())
    # 只取前20个token避免图太大
    n = min(20, len(tokens))
    avg_attn = avg_attn[:n, :n]
    tokens = tokens[:n]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(avg_attn, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_title('BERT最后一层平均注意力权重')
    plt.colorbar(im, ax=ax)

    save_path = os.path.join(save_dir, 'bert_attention_heatmap.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  注意力热力图已保存: {save_path}")
    return save_path


# ==================== GPU Profile对比表 ====================
def plot_gpu_profile_table(lstm_fp32, lstm_amp, bert_fp32, bert_amp, save_dir=OUTPUT_DIR):
    """绘制GPU Profile对比表"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    columns = ['模型', '训练模式', '总训练时间(s)', '平均每Epoch(s)',
               '显存峰值(MB)', '最佳Val Acc(%)']

    data = [
        ['Bi-LSTM', 'FP32', f'{lstm_fp32["total_time"]:.1f}',
         f'{np.mean(lstm_fp32["epoch_times"]):.1f}',
         f'{lstm_fp32["peak_memory_mb"]:.1f}', f'{lstm_fp32["best_val_acc"]:.2f}'],
        ['Bi-LSTM', 'AMP', f'{lstm_amp["total_time"]:.1f}',
         f'{np.mean(lstm_amp["epoch_times"]):.1f}',
         f'{lstm_amp["peak_memory_mb"]:.1f}', f'{lstm_amp["best_val_acc"]:.2f}'],
        ['BERT', 'FP32', f'{bert_fp32["total_time"]:.1f}',
         f'{np.mean(bert_fp32["epoch_times"]):.1f}',
         f'{bert_fp32["peak_memory_mb"]:.1f}', f'{bert_fp32["best_val_acc"]:.2f}'],
        ['BERT', 'AMP', f'{bert_amp["total_time"]:.1f}',
         f'{np.mean(bert_amp["epoch_times"]):.1f}',
         f'{bert_amp["peak_memory_mb"]:.1f}', f'{bert_amp["best_val_acc"]:.2f}'],
    ]

    table = ax.table(cellText=data, colLabels=columns, loc='center',
                     cellLoc='center', colColours=['#E8E8E8'] * len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # 高亮AMP行
    for i in [1, 3]:
        for j in range(len(columns)):
            table[i+1, j].set_facecolor('#E3F2FD')

    ax.set_title('GPU Profile: AMP vs FP32 对比表 (RTX 4060)', fontsize=13, pad=20)

    save_path = os.path.join(save_dir, 'gpu_profile_table.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  GPU Profile表已保存: {save_path}")
    return save_path


# ==================== 超参数对比表 ====================
def plot_hyperparam_table(experiment_results, model_name='LSTM', save_dir=OUTPUT_DIR):
    """绘制超参数实验对比表"""
    fig, ax = plt.subplots(figsize=(10, 3 + len(experiment_results) * 0.5))
    ax.axis('off')

    columns = ['实验名称', '学习率', '批大小', '优化器', '最佳Val Acc(%)', '总时间(s)']
    data = []
    for exp in experiment_results:
        data.append([
            exp['name'],
            f'{exp["learning_rate"]:.0e}',
            str(exp['batch_size']),
            exp.get('optimizer', 'AdamW'),
            f'{exp["best_val_acc"]:.2f}',
            f'{exp["total_time"]:.1f}',
        ])

    table = ax.table(cellText=data, colLabels=columns, loc='center',
                     cellLoc='center', colColours=['#E8E8E8'] * len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # 高亮最佳行
    best_idx = max(range(len(experiment_results)),
                   key=lambda i: experiment_results[i]['best_val_acc'])
    for j in range(len(columns)):
        table[best_idx + 1, j].set_facecolor('#C8E6C9')

    ax.set_title(f'{model_name} 超参数对比实验结果', fontsize=13, pad=20)

    save_path = os.path.join(save_dir, f'{model_name.lower()}_hyperparam_table.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  超参数对比表已保存: {save_path}")
    return save_path


# ==================== 综合对比表 ====================
def plot_final_comparison(lstm_results, bert_results, save_dir=OUTPUT_DIR):
    """绘制LSTM vs BERT最终对比表"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    columns = ['指标', 'Bi-LSTM', 'BERT']

    speedup_str = ''
    lstm_time = lstm_results['total_time']
    bert_time = bert_results['total_time']
    if bert_time > 0:
        speedup_str = ''

    data = [
        ['测试集准确率', f'{lstm_results.get("test_acc", 0)*100:.2f}%',
         f'{bert_results.get("test_acc", 0)*100:.2f}%'],
        ['测试集F1', f'{lstm_results.get("test_f1", 0)*100:.2f}%',
         f'{bert_results.get("test_f1", 0)*100:.2f}%'],
        ['最佳验证准确率', f'{lstm_results["best_val_acc"]:.2f}%',
         f'{bert_results["best_val_acc"]:.2f}%'],
        ['总训练时间', f'{lstm_time:.1f}s', f'{bert_time:.1f}s'],
        ['显存峰值', f'{lstm_results["peak_memory_mb"]:.1f}MB',
         f'{bert_results["peak_memory_mb"]:.1f}MB'],
        ['模型参数量', f'{lstm_results.get("total_params", 0)/1e6:.1f}M',
         f'{bert_results.get("total_params", 0)/1e6:.1f}M'],
    ]

    table = ax.table(cellText=data, colLabels=columns, loc='center',
                     cellLoc='center', colColours=['#E8E8E8'] * len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    ax.set_title('Bi-LSTM vs BERT 最终对比', fontsize=14, pad=20)

    save_path = os.path.join(save_dir, 'final_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  最终对比表已保存: {save_path}")
    return save_path


# ==================== 生成实验报告 ====================
def generate_report(lstm_results, bert_results, lstm_test, bert_test, save_dir=OUTPUT_DIR):
    """生成文本实验报告"""
    report = f"""
{'='*70}
        深度学习期末综合实践 - 方向二 实验报告
        融合LSTM与Transformer/BERT的序列系统
{'='*70}

一、实验概述
  任务：电影简介情感分析（二分类）
  数据集：IMDB Movies ({lstm_results.get('data_size', 'N/A')}条)
  硬件：{lstm_results.get('gpu', 'N/A')}
  框架：PyTorch + HuggingFace Transformers

二、模型对比结果
  {'指标':<20} {'Bi-LSTM':<20} {'BERT':<20}
  {'-'*60}
  {'测试集准确率':<20} {lstm_test['accuracy']*100:<20.2f} {bert_test['accuracy']*100:<20.2f}
  {'测试集精确率':<20} {lstm_test['precision']*100:<20.2f} {bert_test['precision']*100:<20.2f}
  {'测试集召回率':<20} {lstm_test['recall']*100:<20.2f} {bert_test['recall']*100:<20.2f}
  {'测试集F1值':<20} {lstm_test['f1']*100:<20.2f} {bert_test['f1']*100:<20.2f}
  {'最佳验证准确率':<20} {lstm_results['best_val_acc']:<20.2f} {bert_results['best_val_acc']:<20.2f}

三、训练效率对比
  {'指标':<20} {'Bi-LSTM':<20} {'BERT':<20}
  {'-'*60}
  {'总训练时间(s)':<20} {lstm_results['total_time']:<20.1f} {bert_results['total_time']:<20.1f}
  {'平均每Epoch(s)':<20} {np.mean(lstm_results['epoch_times']):<20.1f} {np.mean(bert_results['epoch_times']):<20.1f}
  {'显存峰值(MB)':<20} {lstm_results['peak_memory_mb']:<20.1f} {bert_results['peak_memory_mb']:<20.1f}

四、AMP混合精度对比
  LSTM FP32 vs AMP: 显存 {lstm_results.get('fp32_mem', 0):.1f}MB vs {lstm_results.get('amp_mem', 0):.1f}MB
  BERT FP32 vs AMP: 显存 {bert_results.get('fp32_mem', 0):.1f}MB vs {bert_results.get('amp_mem', 0):.1f}MB

五、分析与结论
  (1) BERT作为预训练模型，在文本分类任务上通常优于Bi-LSTM，
      因为它通过大规模语料预训练获得了丰富的语言知识。
  (2) Bi-LSTM训练速度更快，参数量更少，适合资源受限场景。
  (3) AMP混合精度训练可以在几乎不损失精度的情况下显著减少
      显存占用和训练时间。
  (4) 学习率调度策略（Warmup + Cosine Decay）对BERT微调至关重要。

报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
"""

    save_path = os.path.join(save_dir, 'experiment_report.txt')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n  实验报告已保存: {save_path}")
    print(report)
    return save_path

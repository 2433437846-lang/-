#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练引擎模块：
- 通用训练/验证函数（支持AMP混合精度）
- Bi-LSTM训练器
- BERT训练器
- TensorBoard集成
- 学习率调度（Warmup + Cosine Decay / ReduceLROnPlateau）
- GPU Profile（AMP vs FP32 耗时/显存对比）
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from config import DEVICE, RUNS_DIR, CHECKPOINT_DIR, GPU_NAME


# ==================== 训练一个epoch ====================
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler=None,
                    device=DEVICE, use_amp=False, scaler=None, writer=None,
                    epoch=0, model_type='lstm'):
    """训练一个epoch，返回平均loss和准确率"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        if model_type == 'lstm':
            inputs, lengths, labels = batch
            inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)
        else:  # bert
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                if model_type == 'lstm':
                    outputs = model(inputs, lengths)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    outputs = outputs.logits
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if model_type == 'lstm':
                outputs = model(inputs, lengths)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                outputs = outputs.logits
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # TensorBoard: 记录step级别的loss
        global_step = epoch * len(dataloader) + batch_idx
        if writer is not None and batch_idx % 10 == 0:
            writer.add_scalar(f'Train/loss_step', loss.item(), global_step)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    elapsed = time.time() - start_time

    # TensorBoard: 记录epoch级别的指标
    if writer is not None:
        writer.add_scalar('Train/loss_epoch', avg_loss, epoch)
        writer.add_scalar('Train/accuracy_epoch', accuracy, epoch)
        if scheduler is not None:
            writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)

    return avg_loss, accuracy, elapsed


# ==================== 验证一个epoch ====================
def validate(model, dataloader, criterion, device=DEVICE, model_type='lstm'):
    """验证集评估，返回平均loss和准确率"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            if model_type == 'lstm':
                inputs, lengths, labels = batch
                inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)
                outputs = model(inputs, lengths)
            else:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                outputs = outputs.logits

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ==================== Bi-LSTM训练器 ====================
def train_lstm(model, train_loader, val_loader, config=None, use_amp=False):
    """完整训练Bi-LSTM模型"""
    from config import LSTM_TRAIN_CONFIG

    if config is None:
        config = LSTM_TRAIN_CONFIG

    print("\n" + "=" * 60)
    print("模块3A：Bi-LSTM 训练")
    print("=" * 60)

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # 优化器
    if config.get('optimizer', 'AdamW') == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(),
                                       lr=config['learning_rate'],
                                       weight_decay=config.get('weight_decay', 1e-4))
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                      lr=config['learning_rate'])

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=1
    )

    # TensorBoard
    log_dir = os.path.join(RUNS_DIR, f'lstm_{time.strftime("%Y%m%d_%H%M%S")}')
    writer = SummaryWriter(log_dir)
    print(f"  TensorBoard日志: {log_dir}")

    # AMP
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("  使用AMP混合精度训练")

    # 训练配置
    epochs = config['epochs']
    patience = config.get('patience', 3)
    best_val_acc = 0.0
    no_improve = 0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    epoch_times = []

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"\n  训练轮数: {epochs}")
    print(f"  批大小: {config['batch_size']}")
    print(f"  学习率: {config['learning_rate']}")
    print(f"  优化器: {config.get('optimizer', 'AdamW')}")
    print(f"  设备: {DEVICE} ({GPU_NAME})")

    # GPU显存监控
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        # 训练
        train_loss, train_acc, elapsed = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler=None,
            device=DEVICE, use_amp=use_amp, scaler=scaler,
            writer=writer, epoch=epoch, model_type='lstm'
        )

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion,
                                      device=DEVICE, model_type='lstm')

        # 学习率调度（基于验证准确率）
        scheduler.step(val_acc)

        # 记录
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        epoch_times.append(elapsed)

        # TensorBoard
        writer.add_scalar('Val/loss', val_loss, epoch)
        writer.add_scalar('Val/accuracy', val_acc, epoch)

        print(f"  Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"Time: {elapsed:.1f}s")

        # Early Stopping & Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, 'best_lstm_model.pth'))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  早停于 Epoch {epoch+1}")
                break

    # 显存峰值
    peak_memory = 0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        print(f"\n  GPU显存峰值: {peak_memory:.1f} MB")

    writer.close()

    results = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'epoch_times': epoch_times,
        'total_time': sum(epoch_times),
        'peak_memory_mb': peak_memory,
        'log_dir': log_dir,
        'use_amp': use_amp,
    }

    print(f"\n  训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    print(f"  总训练时间: {sum(epoch_times):.1f}s")

    return results


# ==================== BERT训练器 ====================
def train_bert(model, train_loader, val_loader, config=None, use_amp=False, class_weights=None):
    """完整训练BERT模型"""
    from config import BERT_TRAIN_CONFIG
    from transformers import get_cosine_schedule_with_warmup

    if config is None:
        config = BERT_TRAIN_CONFIG

    print("\n" + "=" * 60)
    print("模块3B：BERT Transformer 训练")
    print("=" * 60)

    model = model.to(DEVICE)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
        print(f"  使用加权损失函数（处理类别不平衡）")
    else:
        criterion = nn.CrossEntropyLoss()

    # 优化器（BERT专用：分层学习率）
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=config['learning_rate'],
                                   weight_decay=config.get('weight_decay', 0.01))

    # 学习率调度：Warmup + Cosine Decay
    epochs = config['epochs']
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # TensorBoard
    log_dir = os.path.join(RUNS_DIR, f'bert_{time.strftime("%Y%m%d_%H%M%S")}')
    writer = SummaryWriter(log_dir)
    print(f"  TensorBoard日志: {log_dir}")

    # AMP
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("  使用AMP混合精度训练")

    # 训练配置
    patience = config.get('patience', 2)
    best_val_acc = 0.0
    no_improve = 0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    epoch_times = []

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"\n  训练轮数: {epochs}")
    print(f"  批大小: {config['batch_size']}")
    print(f"  学习率: {config['learning_rate']}")
    print(f"  Warmup步数: {warmup_steps}/{total_steps}")
    print(f"  设备: {DEVICE} ({GPU_NAME})")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        # 训练
        train_loss, train_acc, elapsed = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler=scheduler,
            device=DEVICE, use_amp=use_amp, scaler=scaler,
            writer=writer, epoch=epoch, model_type='bert'
        )

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion,
                                      device=DEVICE, model_type='bert')

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        epoch_times.append(elapsed)

        # TensorBoard
        writer.add_scalar('Val/loss', val_loss, epoch)
        writer.add_scalar('Val/accuracy', val_acc, epoch)

        print(f"  Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, 'best_bert_model.pth'))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  早停于 Epoch {epoch+1}")
                break

    peak_memory = 0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

    writer.close()

    results = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'epoch_times': epoch_times,
        'total_time': sum(epoch_times),
        'peak_memory_mb': peak_memory,
        'log_dir': log_dir,
        'use_amp': use_amp,
    }

    print(f"\n  训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    print(f"  总训练时间: {sum(epoch_times):.1f}s")

    return results


# ==================== GPU Profile：AMP vs FP32对比 ====================
def run_amp_comparison(model_class, model_kwargs, train_loader, val_loader,
                       model_type='lstm', epochs=3):
    """对比AMP和FP32的训练耗时和显存占用"""
    print("\n" + "=" * 60)
    print(f"GPU Profile: AMP vs FP32 对比实验 ({model_type.upper()})")
    print("=" * 60)

    results = {}

    for amp_mode in [False, True]:
        mode_name = 'AMP' if amp_mode else 'FP32'
        print(f"\n--- {mode_name} 模式 ---")

        # 创建新模型
        if model_type == 'lstm':
            from model_lstm import BiLSTMClassifier
            model = BiLSTMClassifier(**model_kwargs)
            config = {'batch_size': 64, 'epochs': epochs, 'learning_rate': 1e-3,
                      'weight_decay': 1e-4, 'patience': epochs, 'optimizer': 'AdamW'}
            res = train_lstm(model, train_loader, val_loader, config=config, use_amp=amp_mode)
        else:
            from model_bert import build_bert_model
            model = build_bert_model()
            config = {'batch_size': 32, 'epochs': epochs, 'learning_rate': 2e-5,
                      'weight_decay': 0.01, 'warmup_ratio': 0.1, 'patience': epochs}
            res = train_bert(model, train_loader, val_loader, config=config, use_amp=amp_mode)

        results[mode_name] = res

    # 打印对比表
    print("\n" + "=" * 60)
    print("AMP vs FP32 对比结果")
    print("=" * 60)
    print(f"  {'指标':<25} {'FP32':<20} {'AMP':<20}")
    print(f"  {'-' * 65}")
    print(f"  {'总训练时间(s)':<25} {results['FP32']['total_time']:<20.1f} {results['AMP']['total_time']:<20.1f}")
    print(f"  {'平均每epoch时间(s)':<25} {np.mean(results['FP32']['epoch_times']):<20.1f} {np.mean(results['AMP']['epoch_times']):<20.1f}")
    print(f"  {'显存峰值(MB)':<25} {results['FP32']['peak_memory_mb']:<20.1f} {results['AMP']['peak_memory_mb']:<20.1f}")
    print(f"  {'最佳验证准确率(%)':<25} {results['FP32']['best_val_acc']:<20.2f} {results['AMP']['best_val_acc']:<20.2f}")

    fp32_time = np.mean(results['FP32']['epoch_times'])
    amp_time = np.mean(results['AMP']['epoch_times'])
    if amp_time > 0:
        speedup = fp32_time / amp_time
        print(f"\n  AMP加速比: {speedup:.2f}x")

    fp32_mem = results['FP32']['peak_memory_mb']
    amp_mem = results['AMP']['peak_memory_mb']
    if fp32_mem > 0:
        mem_save = (1 - amp_mem / fp32_mem) * 100
        print(f"  AMP显存节省: {mem_save:.1f}%")

    return results

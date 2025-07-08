import optuna
import torch
import torch.nn as nn
import numpy as np
import logging
import sys
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

from models.model import Network
from func.functions import train_from_scratch, validate
from ASVDataloader.ASVRawDataset import ASVRawDataset
from utils.utils import Genotype

def objective(trial):
    # ===== 超参数搜索空间 =====
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    class_weight_ratio = trial.suggest_float("class_weight_ratio", 1.0, 5.0)
    drop_path_prob = trial.suggest_float("drop_path_prob", 0.1, 0.4)

    # ===== 固定参数 =====
    pretrained_model = "pre_trained_models/4-16.pth"
    arch = "Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 3), ('avg_pool_3x3', 2), ('avg_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 3), ('dil_conv_3x3', 2), ('avg_pool_3x3', 4)], reduce_concat=range(2, 6))"
    data_path = Path("../../data/test_sample")

    # 设置设备和随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"🚀 Trial {trial.number}: lr={lr:.2e}, wd={weight_decay:.2e}, bs={batch_size}, ratio={class_weight_ratio:.2f}")

    try:
        # ===== 构建数据集 =====
        train_dataset = ASVRawDataset(data_path, 'train', data_path / 'train.txt')
        dev_dataset = ASVRawDataset(data_path, 'dev', data_path / 'dev.txt')
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True,
            drop_last=True
        )
        dev_loader = torch.utils.data.DataLoader(
            dev_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True,
            drop_last=True
        )

        # ===== 构建模型 =====
        genotype = eval(arch)
        
        # 创建args对象 (模拟原始代码中的args)
        class Args:
            def __init__(self):
                self.nfft = 1024
                self.hop = 4
                self.nfilter = 70
                self.num_ceps = 20
                self.sr = 16000
                self.is_log = True
                self.is_mask = False
                self.is_cmvn = False
                self.report_freq = 1000
        
        args = Args()
        
        model = Network(16, 4, args, 2, genotype, 'LFCC').to(device)
        
        # 加载预训练模型
        ckpt = torch.load(pretrained_model, map_location='cpu')
        model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
        
        # 冻结backbone
        for name, param in model.named_parameters():
            if 'classifier' not in name.lower():
                param.requires_grad = False

        # ===== Loss & Optimizer =====
        weight = torch.FloatTensor([1.0, class_weight_ratio]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            verbose=False,
            min_lr=1e-6
        )

        # 创建writer_dict (简化版)
        # 创建一个假的writer对象来避免None错误
        class DummyWriter:
            def add_scalar(self, *args, **kwargs):
                pass
            def close(self):
                pass
        
        writer_dict = {
            'writer': DummyWriter(),  # 使用假的writer对象
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }

        best_eer = float('inf')
        patience = 0
        max_patience = 5  # 提前停止

        for epoch in range(25):  # 减少epoch数以加快搜索
            # 设置drop_path_prob
            model.drop_path_prob = drop_path_prob * epoch / 25
            
            # 训练
            train_acc, train_loss = train_from_scratch(
                args, train_loader, model, optimizer, criterion, epoch, writer_dict
            )
            
            # 验证
            dev_acc, dev_eer, dev_frr = validate(
                dev_loader, model, criterion, epoch, writer_dict, validate_type='dev'
            )

            # 更新学习率
            scheduler.step(dev_eer)
            
            # 记录最佳结果
            if dev_eer < best_eer:
                best_eer = dev_eer
                patience = 0
            else:
                patience += 1
            
            # 报告中间结果给optuna
            trial.report(dev_eer, epoch)
            
            # 检查是否应该剪枝
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            # 早停
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch}, best EER: {best_eer:.4f}")
                break
            
            # 每5个epoch打印一次进度
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: EER={dev_eer:.4f}, best_EER={best_eer:.4f}")

        print(f"✅ Trial {trial.number} completed with best EER: {best_eer:.4f}")
        return best_eer

    except Exception as e:
        print(f"❌ Trial {trial.number} failed: {str(e)}")
        raise optuna.exceptions.TrialPruned()

def run_optuna():
    print("🔍 Starting Optuna hyperparameter optimization...")
    
    # 创建研究
    study = optuna.create_study(
        direction="minimize",
        study_name="asv_finetune_optimization",
        storage="sqlite:///optuna_finetune.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
        ),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # 运行优化
    try:
        study.optimize(
            objective, 
            n_trials=20,  # 可以根据时间调整
            timeout=3600*4,  # 4小时超时
            show_progress_bar=True,
            callbacks=[
                lambda study, trial: print(f"📊 Trials completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
                if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) > 0 
                else print(f"📊 Trial {trial.number} completed")
            ]
        )
    except KeyboardInterrupt:
        print("\n⏹️ Optimization interrupted by user")
    
    # 输出结果
    print("\n" + "="*50)
    print("🎯 OPTIMIZATION RESULTS")
    print("="*50)
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(completed_trials) > 0:
        print(f"Best EER: {study.best_value:.4f}")
        print(f"Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
    else:
        print("No trials completed successfully.")
    
    print(f"Total trials: {len(study.trials)}")
    print(f"Completed trials: {len(completed_trials)}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    
    # 保存结果
    import json
    if len(completed_trials) > 0:
        with open('best_params_finetune.json', 'w') as f:
            json.dump(study.best_params, f, indent=2)
        print(f"💾 Best parameters saved to best_params_finetune.json")
    else:
        print("⚠️ No best parameters to save.")
    
    # 保存试验历史
    df = study.trials_dataframe()
    df.to_csv('optuna_finetune_trials.csv', index=False)
    print(f"📋 Trial history saved to optuna_finetune_trials.csv")

def plot_results():
    """绘制优化结果"""
    try:
        import optuna.visualization as vis
        
        study = optuna.load_study(
            study_name="asv_finetune_optimization",
            storage="sqlite:///optuna_finetune.db"
        )
        
        # 优化历史
        fig1 = vis.plot_optimization_history(study)
        fig1.write_html('finetune_optimization_history.html')
        print("📈 Optimization history saved to finetune_optimization_history.html")
        
        # 参数重要性
        fig2 = vis.plot_param_importances(study)
        fig2.write_html('finetune_param_importances.html')
        print("📊 Parameter importances saved to finetune_param_importances.html")
        
    except ImportError:
        print("⚠️ Install plotly for visualization: pip install plotly")

if __name__ == "__main__":
    # 检查必要文件
    import os
    if not os.path.exists('pre_trained_models/4-16.pth'):
        print("❌ pre_trained_models/4-16.pth not found!")
        exit(1)
    
    if not os.path.exists('../../data/test_sample'):
        print("❌ ../../data/test_sample not found!")
        exit(1)
    
    # 运行优化
    run_optuna()
    
    # 生成可视化
    plot_results()
    
    print("\n🎉 Finetune optimization completed!")
    print("Next steps:")
    print("1. Check best_params_finetune.json for optimal hyperparameters")
    print("2. Run training with best parameters using your original finetune_v2.py")
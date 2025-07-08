# import numpy as np
# import torch
# from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# import argparse
# import os

# def calculate_eer(y_true, y_scores):
#     """计算EER (Equal Error Rate)"""
#     fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
#     fnr = 1 - tpr
    
#     # 找到FPR和FNR最接近的点
#     eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
#     eer_value = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
#     return eer_value, eer_threshold

# def calculate_min_dcf(y_true, y_scores, p_target=0.01, c_miss=1, c_fa=1):
#     """计算最小检测代价函数 (min t-DCF)"""
#     fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
#     fnr = 1 - tpr
    
#     # 计算DCF
#     dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    
#     # 找到最小DCF
#     min_dcf_idx = np.argmin(dcf)
#     min_dcf = dcf[min_dcf_idx]
#     min_dcf_threshold = thresholds[min_dcf_idx]
    
#     return min_dcf, min_dcf_threshold

# def parse_score_file(score_file):
#     """解析分数文件 - 适配PC-DARTS输出格式"""
#     y_true = []
#     y_scores = []
#     file_names = []
    
#     print(f"正在解析分数文件: {score_file}")
    
#     with open(score_file, 'r') as f:
#         lines = f.readlines()
#         print(f"文件总行数: {len(lines)}")
        
#         # 检查前几行来确定格式
#         if len(lines) > 0:
#             first_line = lines[0].strip().split()
#             print(f"第一行格式: {first_line}")
#             print(f"第一行长度: {len(first_line)}")
        
#         for i, line in enumerate(lines):
#             parts = line.strip().split()
#             if len(parts) >= 3:
#                 try:
#                     file_name = parts[0]
                    
#                     # 尝试不同的格式
#                     if len(parts) == 4:
#                         # 格式: filename protocol label score
#                         label = parts[2]
#                         score = float(parts[3])
#                     elif len(parts) == 3:
#                         # 格式: filename label score
#                         label = parts[1]
#                         score = float(parts[2])
#                     else:
#                         # 其他格式，跳过
#                         continue
                    
#                     file_names.append(file_name)
#                     y_scores.append(score)
#                     # bonafide = 1, spoof = 0
#                     y_true.append(1 if label.lower() == 'bonafide' else 0)
                    
#                 except (ValueError, IndexError) as e:
#                     print(f"跳过第{i+1}行，解析错误: {line.strip()}")
#                     continue
    
#     print(f"成功解析 {len(y_true)} 个样本")
#     print(f"Bonafide样本: {sum(y_true)}, Spoof样本: {len(y_true) - sum(y_true)}")
    
#     return np.array(y_true), np.array(y_scores), file_names

# def print_detailed_results(y_true, y_scores):
#     """打印详细的评估结果"""
    
#     # 计算EER
#     eer, eer_threshold = calculate_eer(y_true, y_scores)
    
#     # 计算min t-DCF
#     min_dcf, min_dcf_threshold = calculate_min_dcf(y_true, y_scores)
    
#     # 统计数据集信息
#     bonafide_count = np.sum(y_true == 1)
#     spoof_count = np.sum(y_true == 0)
#     total_count = len(y_true)
    
#     # 在EER阈值下的预测
#     y_pred_eer = (y_scores >= eer_threshold).astype(int)
    
#     # 计算性能指标
#     cm = confusion_matrix(y_true, y_pred_eer)
#     tn, fp, fn, tp = cm.ravel()
    
#     accuracy = accuracy_score(y_true, y_pred_eer)
#     precision = precision_score(y_true, y_pred_eer, zero_division=0)
#     recall = recall_score(y_true, y_pred_eer, zero_division=0)
#     f1 = f1_score(y_true, y_pred_eer, zero_division=0)
    
#     far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Acceptance Rate
#     frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Rejection Rate
    
#     print("=" * 60)
#     print("PC-DARTS ASVspoof2019 反欺诈检测结果")
#     print("=" * 60)
    
#     print(f"\n数据集信息:")
#     print(f"  Bonafide样本数: {bonafide_count}")
#     print(f"  Spoof样本数: {spoof_count}")
#     print(f"  总样本数: {total_count}")
    
#     print(f"\n性能指标:")
#     print(f"  EER: {eer*100:.3f}%")
#     print(f"  最佳阈值: {eer_threshold:.6f}")
#     print(f"  min t-DCF: {min_dcf:.5f}")
    
#     print(f"\n在最佳阈值下的指标:")
#     print(f"  FAR (假阳率): {far*100:.3f}%")
#     print(f"  FRR (假阴率): {frr*100:.3f}%")
#     print(f"  准确率: {accuracy*100:.3f}%")
#     print(f"  精确率: {precision*100:.3f}%")
#     print(f"  召回率: {recall*100:.3f}%")
#     print(f"  F1分数: {f1:.3f}")
    
#     print(f"\n混淆矩阵:")
#     print(f"  TN (正确拒绝spoof): {tn}")
#     print(f"  FP (错误接受spoof): {fp}")
#     print(f"  FN (错误拒绝bonafide): {fn}")
#     print(f"  TP (正确接受bonafide): {tp}")
    
#     # 分数统计
#     bonafide_scores = y_scores[y_true == 1]
#     spoof_scores = y_scores[y_true == 0]
    
#     print(f"\n分数统计:")
#     print(f"  Bonafide分数 - 均值: {np.mean(bonafide_scores):.6f}, 标准差: {np.std(bonafide_scores):.6f}")
#     print(f"  Spoof分数 - 均值: {np.mean(spoof_scores):.6f}, 标准差: {np.std(spoof_scores):.6f}")
#     print(f"  分数范围: [{np.min(y_scores):.6f}, {np.max(y_scores):.6f}]")
    
#     print("=" * 60)
    
#     return {
#         'eer': eer,
#         'eer_threshold': eer_threshold,
#         'min_dcf': min_dcf,
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1,
#         'far': far,
#         'frr': frr,
#         'confusion_matrix': (tn, fp, fn, tp)
#     }

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('PC-DARTS EER计算')
#     parser.add_argument('--score_file', type=str, required=True, 
#                        help='分数文件路径 (PC-DARTS输出)')
    
#     args = parser.parse_args()
    
#     if not os.path.exists(args.score_file):
#         print(f"错误: 分数文件 {args.score_file} 不存在")
#         exit(1)
    
#     # 解析分数文件
#     y_true, y_scores, file_names = parse_score_file(args.score_file)
    
#     if len(y_true) == 0:
#         print("错误: 没有成功解析到任何数据")
#         exit(1)
    
#     # 打印结果
#     results = print_detailed_results(y_true, y_scores)

import numpy as np
import torch
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import argparse
import os

def calculate_eer(y_true, y_scores):
    """计算EER (Equal Error Rate)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    
    # 找到FPR和FNR最接近的点
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_value = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    return eer_value, eer_threshold

def calculate_min_dcf(y_true, y_scores, p_target=0.01, c_miss=1, c_fa=1):
    """计算最小检测代价函数 (min t-DCF)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    
    # 计算DCF
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    
    # 找到最小DCF
    min_dcf_idx = np.argmin(dcf)
    min_dcf = dcf[min_dcf_idx]
    min_dcf_threshold = thresholds[min_dcf_idx]
    
    return min_dcf, min_dcf_threshold

def parse_score_file(score_file):
    """解析分数文件 - 适配PC-DARTS输出格式"""
    y_true = []
    y_scores = []
    file_names = []
    
    print(f"正在解析分数文件: {score_file}")
    
    with open(score_file, 'r') as f:
        lines = f.readlines()
        print(f"文件总行数: {len(lines)}")
        
        # 检查前几行来确定格式
        if len(lines) > 0:
            first_line = lines[0].strip().split()
            print(f"第一行格式: {first_line}")
            print(f"第一行长度: {len(first_line)}")
        
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    file_name = parts[0]
                    
                    # 尝试不同的格式
                    if len(parts) == 4:
                        # 格式: filename protocol label score
                        label = parts[2]
                        score = float(parts[3])
                    elif len(parts) == 3:
                        # 格式: filename label score
                        label = parts[1]
                        score = float(parts[2])
                    else:
                        # 其他格式，跳过
                        continue
                    
                    file_names.append(file_name)
                    y_scores.append(score)
                    # bonafide = 1, spoof = 0
                    y_true.append(1 if label.lower() == 'bonafide' else 0)
                    
                except (ValueError, IndexError) as e:
                    print(f"跳过第{i+1}行，解析错误: {line.strip()}")
                    continue
    
    print(f"成功解析 {len(y_true)} 个样本")
    print(f"Bonafide样本: {sum(y_true)}, Spoof样本: {len(y_true) - sum(y_true)}")
    
    return np.array(y_true), np.array(y_scores), file_names

def print_detailed_results(y_true, y_scores, fixed_threshold=None):
    """打印详细的评估结果"""
    
    # 计算EER（当前数据集的EER，仅供参考）
    eer, eer_threshold = calculate_eer(y_true, y_scores)
    
    # 计算min t-DCF
    min_dcf, min_dcf_threshold = calculate_min_dcf(y_true, y_scores)
    
    # 统计数据集信息
    bonafide_count = np.sum(y_true == 1)
    spoof_count = np.sum(y_true == 0)
    total_count = len(y_true)
    
    # 决定使用哪个阈值
    if fixed_threshold is not None:
        # 使用固定阈值
        threshold_to_use = fixed_threshold
        threshold_source = "固定阈值 (来自dev集)"
        print("=" * 60)
        print("使用固定阈值的测试结果")
        print("=" * 60)
    else:
        # 使用当前数据集的EER阈值
        threshold_to_use = eer_threshold
        threshold_source = "当前数据集的EER阈值"
        print("=" * 60)
        print("PC-DARTS ASVspoof2019 反欺诈检测结果")
        print("=" * 60)
    
    # 在选定阈值下的预测
    y_pred = (y_scores >= threshold_to_use).astype(int)
    
    # 计算性能指标
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Acceptance Rate
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Rejection Rate
    
    print(f"\n数据集信息:")
    print(f"  Bonafide样本数: {bonafide_count}")
    print(f"  Spoof样本数: {spoof_count}")
    print(f"  总样本数: {total_count}")
    
    if fixed_threshold is not None:
        print(f"\n使用固定阈值的性能:")
        print(f"  使用的阈值: {threshold_to_use:.6f} ({threshold_source})")
        print(f"  EER: {((far + frr) / 2)*100:.3f}%")
        print(f"  FAR (假阳率): {far*100:.3f}%")
        print(f"  FRR (假阴率): {frr*100:.3f}%")
        print(f"  准确率: {accuracy*100:.3f}%")
        print(f"  精确率: {precision*100:.3f}%")
        print(f"  召回率: {recall*100:.3f}%")
        print(f"  F1分数: {f1:.3f}")
        
        print(f"\n当前数据集的EER (仅供参考):")
        print(f"  如果在此数据集上优化，EER: {eer*100:.3f}%")
        print(f"  对应的最佳阈值: {eer_threshold:.6f}")
        print(f"  min t-DCF: {min_dcf:.5f}")
    else:
        print(f"\n性能指标:")
        print(f"  EER: {eer*100:.3f}%")
        print(f"  最佳阈值: {eer_threshold:.6f}")
        print(f"  min t-DCF: {min_dcf:.5f}")
        
        print(f"\n在最佳阈值下的指标:")
        print(f"  FAR (假阳率): {far*100:.3f}%")
        print(f"  FRR (假阴率): {frr*100:.3f}%")
        print(f"  准确率: {accuracy*100:.3f}%")
        print(f"  精确率: {precision*100:.3f}%")
        print(f"  召回率: {recall*100:.3f}%")
        print(f"  F1分数: {f1:.3f}")
    
    print(f"\n混淆矩阵:")
    print(f"  TN (正确拒绝spoof): {tn}")
    print(f"  FP (错误接受spoof): {fp}")
    print(f"  FN (错误拒绝bonafide): {fn}")
    print(f"  TP (正确接受bonafide): {tp}")
    
    # 分数统计
    bonafide_scores = y_scores[y_true == 1]
    spoof_scores = y_scores[y_true == 0]
    
    print(f"\n分数统计:")
    print(f"  Bonafide分数 - 均值: {np.mean(bonafide_scores):.6f}, 标准差: {np.std(bonafide_scores):.6f}")
    print(f"  Spoof分数 - 均值: {np.mean(spoof_scores):.6f}, 标准差: {np.std(spoof_scores):.6f}")
    print(f"  分数范围: [{np.min(y_scores):.6f}, {np.max(y_scores):.6f}]")
    
    # 检查阈值是否在合理范围内
    if fixed_threshold is not None:
        if threshold_to_use < np.min(y_scores):
            print(f"\n⚠️  警告: 固定阈值 {threshold_to_use:.6f} 小于最小分数 {np.min(y_scores):.6f}")
            print(f"     所有样本都会被分类为bonafide!")
        elif threshold_to_use > np.max(y_scores):
            print(f"\n⚠️  警告: 固定阈值 {threshold_to_use:.6f} 大于最大分数 {np.max(y_scores):.6f}")
            print(f"     所有样本都会被分类为spoof!")
    
    print("=" * 60)
    
    return {
        'eer': eer,
        'eer_threshold': eer_threshold,
        'min_dcf': min_dcf,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'far': far,
        'frr': frr,
        'confusion_matrix': (tn, fp, fn, tp),
        'threshold_used': threshold_to_use
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PC-DARTS EER计算')
    parser.add_argument('--score_file', type=str, required=True, 
                       help='分数文件路径 (PC-DARTS输出)')
    parser.add_argument('--threshold', type=float, default=None,
                       help='固定阈值 (如果不提供，则使用当前数据集的EER阈值)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.score_file):
        print(f"错误: 分数文件 {args.score_file} 不存在")
        exit(1)
    
    # 解析分数文件
    y_true, y_scores, file_names = parse_score_file(args.score_file)
    
    if len(y_true) == 0:
        print("错误: 没有成功解析到任何数据")
        exit(1)
    
    # 打印结果
    results = print_detailed_results(y_true, y_scores, args.threshold)
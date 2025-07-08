import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from utils.utils import AvgrageMeter, EERMeter, TotalAccuracyMeter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d



def train(args, train_loader, val_loader, model, architect, criterion, lr, optimizer, epoch, writer_dict):
    losses = AvgrageMeter()
    total_acc_meter = TotalAccuracyMeter('Total accuracy')
    writer = writer_dict['writer']
    model.train()

    for step, (input, target) in enumerate(train_loader):
        global_steps = writer_dict['train_global_steps']

        # batch size
        n = input.size(0)

        input_search, target_search = next(iter(val_loader))
        # try:
        #     input_search, target_search = next(val_loader_iter)
        # except StopIteration:
        #     val_loader_iter = iter(val_loader)
        #     input_search, target_search = next(val_loader_iter)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        input_search = input_search.cuda(non_blocking=True)
        target_search = target_search.cuda(non_blocking=True)
    
        if epoch >= args.warm_up_epoch:
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
    

        output, embeddings = model(input, args.is_mask)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), n)
        total_acc_meter.update(target, output)

        if step % args.report_freq == 0:
            logging.info('train %03d %e', step, losses.avg)
        
        writer.add_scalar('train_loss', losses.val, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1
        
    acc_per_epoch = total_acc_meter.get_accuracy()
    loss_per_epoch = losses.avg
    writer.add_scalar('arch_train_accuracy', acc_per_epoch, epoch)
    writer.add_scalar('arch_train_loss', loss_per_epoch, epoch)

    total_loss = losses.sum

    return acc_per_epoch*100, total_loss

def calculate_eer(y_true, y_scores):
    """
    计算Equal Error Rate (EER)
    
    Args:
        y_true: 真实标签 (0=bonafide, 1=spoof)
        y_scores: 预测分数 (spoof的概率)
    
    Returns:
        eer: Equal Error Rate (百分比)
    """
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    
    # EER是FPR=1-TPR的交点
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100  

def validate(dev_loader, model, criterion, epoch, writer_dict, validate_type):
    from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score
    from scipy.interpolate import interp1d
    from scipy.optimize import brentq

    def calculate_eer(y_true, y_scores):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer * 100

    losses = AvgrageMeter()
    total_acc_meter = TotalAccuracyMeter('Total accuracy')

    model.eval()
    all_preds = []
    all_targets = []
    all_scores = []

    with torch.no_grad():
        for step, (input, target) in enumerate(dev_loader):
            input = input.cuda(non_blocking=True).float()
            target = target.cuda(non_blocking=True)

            output = model(input, is_mask=False)
            output = model.forward_classifier(output)

            preds = torch.argmax(output, dim=1)
            scores = torch.softmax(output, dim=1)[:, 1]  # spoof 的概率（正类）

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

            total_acc_meter.update(target, output)

            if criterion:
                loss = criterion(output, target)
                losses.update(loss.item(), input.size(0))

    acc_per_epoch = total_acc_meter.get_accuracy()
    eer_per_epoch = calculate_eer(all_targets, all_scores)

    # 混淆矩阵 & FRR / FAR
    cm = confusion_matrix(all_targets, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
    else:
        frr = 0
        far = 0

    # 输出与记录
    print(f"\n📌 Confusion Matrix (Epoch {epoch}):\n{cm}")
    print(f"📊 Accuracy: {acc_per_epoch * 100:.2f}%")
    print(f"📉 EER: {eer_per_epoch:.2f}%")
    print(f"📉 FRR: {frr * 100:.2f}%")
    print(f"📈 FAR: {far * 100:.2f}%\n")

    if writer_dict:
        writer = writer_dict['writer']
        writer.add_scalar(validate_type + '_accuracy', acc_per_epoch * 100, epoch)
        writer.add_scalar(validate_type + '_loss', losses.avg, epoch)
        writer.add_scalar(validate_type + '_eer', eer_per_epoch, epoch)
        writer.add_scalar(validate_type + '_frr', frr * 100, epoch)

    return acc_per_epoch * 100, eer_per_epoch, frr * 100


def train_from_scratch(args, train_loader, model, optimizer, criterion, epoch, writer_dict):
    losses = AvgrageMeter()
    total_acc_meter = TotalAccuracyMeter('Total accuracy')
    writer = writer_dict['writer']

    model.train()
    for step, (input, target) in enumerate(train_loader):
        global_steps = writer_dict['train_global_steps']
        
        # batch size
        n = input.size(0)
        input = input.cuda(non_blocking=True)

        target = target.cuda(non_blocking=True)

        output, embeddings = model(input, args.is_mask)

        loss = criterion(output, target)
        losses.update(loss.item(), n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_acc_meter.update(target, output)

        if step % args.report_freq == 0:
            logging.info('train loss: %03d %e', step, losses.avg)
        writer.add_scalar('train_loss_step', losses.val, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    acc_per_epoch = total_acc_meter.get_accuracy()
    loss_per_epoch = losses.avg
    writer.add_scalar('train_accuracy', acc_per_epoch*100, epoch)
    writer.add_scalar('train_loss', loss_per_epoch, epoch)

    total_loss = losses.sum

    return acc_per_epoch*100, total_loss


def evaluate(test_loader, model, comment):
    eermeter = EERMeter('EER', round_digits=4)
    total_acc_meter = TotalAccuracyMeter('Total accuracy')

    model.eval()
    fname_list = []
    key_list = []
    att_id_list = []
    key_list = []
    score_list = []

    with torch.no_grad():
        for step, (input, file_name, attack_id, target) in tqdm(enumerate(test_loader)):

            input = input.cuda(non_blocking=True).float()
            target = target.cuda(non_blocking=True)
            output = model(input, is_mask=False)
            output = model.forward_classifier(output)


            eermeter.update(target, output)
            total_acc_meter.update(target, output)

            fname_list.extend(list(file_name))
            key_list.extend(
            ['bonafide' if key == 1 else 'spoof' for key in target.tolist()])
            att_id_list.extend(list(attack_id))
            score_list.extend(output[:,1].tolist())

    save_path = 'score-' + comment + '.txt'
    with open(save_path, 'a') as fh:
        for f, s, k, cm in zip(fname_list, att_id_list, key_list, score_list):
            fh.write('{} {} {} {}\n'.format(f, s, k, cm))





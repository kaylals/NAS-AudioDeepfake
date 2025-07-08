import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from pathlib import Path
from models.model import Network
from func.architect import Architect
from ASVDataloader.AugRawDataset import AugRawDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ASVDataloader.ASVRawDataset import ASVRawDataset

from utils.utils import count_parameters
from func.functions import train_from_scratch, validate
from utils import utils
from utils.utils import Genotype

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ASVSpoof2019 model')
    parser.add_argument('--data', type=str, default='/path/to/your/LA', help='location of the data')                           
    parser.add_argument('--valid_freq', type=int, default=1, help='validate frequency')
    parser.add_argument('--report_freq', type=int, default=1000, help='report frequency in training')
    parser.add_argument('--layers', type=int, default=4, help='number of cells of the network')
    parser.add_argument('--init_channels', type=int, default=16, help='number of the initial channels of the network')
    parser.add_argument('--arch', type=str, help='the searched architecture')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--nfft', type=int, default=1024, help='number of FFT point')
    parser.add_argument('--hop', type=int, default=4, help='number of hop size (nfft//hop)')
    parser.add_argument('--nfilter', type=int, default=70, help='number of linear filter')
    parser.add_argument('--num_ceps', type=int, default=20, help='LFCC dimention before deltas')
    parser.add_argument('--log', dest='is_log', action='store_true', help='whether use log(STFT)')
    parser.add_argument('--no-log', dest='is_log', action='store_false', help='whether use log(STFT)')
    parser.add_argument('--mask', dest='is_mask', action='store_true', help='whether use freq mask')
    parser.add_argument('--no-mask', dest='is_mask', action='store_false', help='whether use freq mask')
    parser.add_argument('--cmvn', dest='is_cmvn', action='store_true', help='whether zero-mean std')
    parser.add_argument('--no-cmvn', dest='is_cmvn', action='store_false', help='whether zero-mean std')
    parser.add_argument('--frontend', type=str, help='select frontend, it can be either spec, lfb or lfcc')
    parser.add_argument('--sr', type=int, default=16000, help='default sampling rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='intial learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-4, help='mininum learning rate')
    # parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--comment', type=str, default='EXP', help='Comment to describe the saved mdoel')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
    parser.add_argument('--pretrained_model', type=str, default=None, help='path to pretrained model for fine-tuning')
    parser.add_argument('--finetune_lr', type=float, default=None, help='specific learning rate for fine-tuning')
    parser.add_argument('--freeze_backbone', action='store_true', help='freeze backbone layers, only train classifier')

    parser.set_defaults(is_log=True)
    parser.set_defaults(is_mask=False)
    parser.set_defaults(is_cmvn=False)

    args = parser.parse_args()
    args.comment = 'train-{}-{}'.format(args.comment, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.comment, scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.comment, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    
    # models will be saved under this path
    model_save_path = os.path.join(args.comment, 'models')
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    train_protocol = '../../data/test_sample/train.txt'
    dev_protocol = '../../data/test_sample/dev.txt'



    if args.frontend == 'spec':
        front_end = 'Spectrogram'
        logging.info('-----Using STFT frontend-----')
    elif args.frontend == 'lfcc':
        front_end = 'LFCC'
        logging.info('-----Using LFCC frontend-----')
    elif args.frontend == 'lfb':
        front_end = 'LFB'
        logging.info('-----Using LFB frontend-----')

    OUTPUT_CLASSES = 2
    
    # set random seed
    if args.seed:
        cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    logging.info("args = %s", args)
    
    device = 'cuda'
    weight = torch.FloatTensor([1.0, 2.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    criterion = criterion.cuda()

    # get the network architecture
    genotype = eval(args.arch)
    # initialise the model
    model = Network(args.init_channels, args.layers, args, OUTPUT_CLASSES, genotype, front_end)
    model = model.to(device)

    # 在模型加载完成后，freeze之前和之后分别统计
    if args.pretrained_model:
        print(f"🔄 Loading pretrained model from {args.pretrained_model}")
        checkpoint = torch.load(args.pretrained_model, map_location='cpu')
        model_state = checkpoint.get('state_dict', checkpoint)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        print(f"✅ Loaded pretrained model with {len(missing)} missing keys and {len(unexpected)} unexpected keys")

        # 统计加载后、冻结前的参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 模型总参数量: {total_params/1e6:.3f}M")
        
        if args.freeze_backbone:
            frozen = 0
            for name, param in model.named_parameters():
                if 'classifier' not in name.lower():
                    param.requires_grad = False
                    frozen += 1
            print(f'🔒 Frozen {frozen} backbone parameters. Only classifier will be trained.')
            
            # 统计冻结后的可训练参数
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"✅ 可训练参数: {trainable_params/1e6:.6f}M")


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total param size = %fM", total_params / 1e6)
    logging.info("Trainable param size = %fM", trainable_params / 1e6)

    train_dataset = AugRawDataset(Path(args.data), 'train', train_protocol)
    dev_dataset = ASVRawDataset(Path(args.data), 'dev', dev_protocol) 

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    dev_loader = torch.utils.data.DataLoader(
        dataset=dev_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )


    if args.pretrained_model:
        if args.finetune_lr:
            lr = args.finetune_lr
        else:
            lr = args.lr * 0.1
    else:
        lr = args.lr

    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr, 
    #     weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False
    )


    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer, float(args.num_epochs), eta_min=args.lr_min)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',      # 监控EER，越小越好
        factor=0.5,      # 学习率减半
        patience=3,      # 3个epoch不改善就降低学习率
        verbose=True,    # 打印信息
        min_lr=args.lr_min
    )
    begin_epoch = 0
    best_acc = 85
    writer_dict = {
        'writer': SummaryWriter(args.comment),
        'train_global_steps': begin_epoch * len(train_loader),
        'valid_global_steps': begin_epoch // args.valid_freq,
    }
    best_eer = float('inf')
    patience = 0
    max_patience = 10

    for epoch in range(args.num_epochs):
        lr = scheduler.get_last_lr()[0]
        logging.info('Epoch: %d lr: %e', epoch, lr)

        model.drop_path_prob = args.drop_path_prob * epoch / args.num_epochs
        
        train_acc, train_loss = train_from_scratch(args, train_loader, model, optimizer, criterion, epoch, writer_dict)
        logging.info('train_loss %f', train_loss)
        logging.info('train_acc %f', train_acc)
        if epoch % args.valid_freq == 0:
            dev_acc, dev_eer, dev_frr = validate(dev_loader, model, criterion, epoch, writer_dict, validate_type='dev')

            print(f'🎯 Epoch {epoch} - Acc: {dev_acc:.2f}%, EER: {dev_eer:.2f}%, FRR: {dev_frr:.2f}%')
            logging.info('dev_frr %f', dev_frr)
            logging.info('dev_acc %f', dev_acc)
            logging.info('dev_eer %f', dev_eer)
            # 可以根据EER或准确率保存最佳模型
            if dev_acc > best_acc:
                best_acc = dev_acc
                torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
                logging.info("✅ Saved new best model (based on dev acc)")
                print('*'*50)
                logging.info('best acc model found')
                print('*'*50)
            
            if dev_eer < best_eer:
                best_eer = dev_eer
                patience = 0
                torch.save(model.state_dict(), os.path.join(model_save_path, 'best_eer_model.pth'))
                logging.info("✅ Saved new best EER model (EER: %.4f)", best_eer)
            else:
                patience += 1
                logging.info("EER did not improve. Patience: %d/%d", patience, max_patience)
            
            scheduler.step(dev_eer)

            if patience >= max_patience:
                logging.info("Early stopping triggered! Best EER: %.4f", best_eer)
                break

        torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
       
import numpy as np
import torch
import torch.utils.data as data
import librosa
from ASVDataloader.Augmentation import Augmentation

class AugRawDataset(data.Dataset):
    def __init__(self, root, partition, protocol_name):
        super(AugRawDataset, self).__init__()  # 修正这里，应该是AugRawDataset
        self.root = root
        self.partition = partition
        
        # 初始化增强器
        self.aug = Augmentation(prob=0.5)

        self.sysid_dict = {
            'bonafide': 1,  
            'spoof': 0, 
        }
        
        protocol_dir = root.joinpath(protocol_name)
        protocol_lines = open(protocol_dir).readlines()

        self.features = []
        
        for protocol_line in protocol_lines:
            tokens = protocol_line.strip().split(' ')
            # 你的协议格式: A0001 tts_001 - - spoof
            # [0]    [1]     [2][3] [4]
            
            # 直接使用flac文件夹，不需要分train/dev子文件夹
            feature_path = self.root.joinpath('flac', tokens[1] + '.flac')
            sys_id = self.sysid_dict[tokens[4]]
            self.features.append((feature_path, sys_id))

    def load_feature(self, feature_path):
        feature, sr = librosa.load(feature_path, sr=16000)
        fix_len = sr*4

        while feature.shape[0] < fix_len:
            feature = np.concatenate((feature, feature))
        feature = feature[:fix_len]

        return feature

    def __getitem__(self, index):
        feature_path, sys_id = self.features[index]
        feature = self.load_feature(feature_path)
        
        # 转换为tensor并应用增强
        feature = torch.from_numpy(feature).float()
        feature = self.aug(feature)
        
        return feature, sys_id

    def __len__(self):
        return len(self.features)
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


class AudioFeatureDataset(Dataset):
    def __init__(self, hubert_dir, pitch_dir, speaker_dir, ppg_dir, mel_dir):
        """
        初始化数据集
        :param hubert_dir: 存储Hubert特征的目录
        :param pitch_dir: 存储pitch特征的目录
        :param speaker_dir: 存储说话人特征的目录
        :param ppg_dir: 存储PPG特征的目录
        """
        self.hubert_files = [os.path.join(hubert_dir, file) for file in os.listdir(hubert_dir)]
        self.pitch_files = [os.path.join(pitch_dir, file) for file in os.listdir(pitch_dir)]
        self.speaker_files = [os.path.join(speaker_dir, file) for file in os.listdir(speaker_dir)]
        self.ppg_files = [os.path.join(ppg_dir, file) for file in os.listdir(ppg_dir)]
        self.mel_files = [os.path.join(mel_dir, file) for file in os.listdir(mel_dir)]

        # 确保文件对齐，这里简单假设文件顺序相同且一一对应
        self.hubert_files.sort()
        self.pitch_files.sort()
        self.speaker_files.sort()
        self.ppg_files.sort()
        self.mel_files.sort()

    def __len__(self):
        return len(self.hubert_files)

    def __getitem__(self, idx):
        # 读取特征，确保在加载时允许pickle
        hubert_feature = np.load(self.hubert_files[idx], allow_pickle=True).astype(np.float32)
        pitch_feature = np.load(self.pitch_files[idx], allow_pickle=True).astype(np.float32)
        speaker_feature = np.load(self.speaker_files[idx], allow_pickle=True).astype(np.float32)
        ppg_feature = np.load(self.ppg_files[idx], allow_pickle=True).astype(np.float32)
        mel_feature = torch.load(self.mel_files[idx])  # 加载.pt格式的Mel频谱

        # 处理特征，确保它们的维度和长度匹配
        # 可能需要调整pitch特征的尺寸，因为它的长度通常是其他特征的两倍
        pitch_feature = pitch_feature[:len(hubert_feature)]

        # 将说话人特征扩展到与其他特征相同的时间维度
        speaker_feature = np.repeat(speaker_feature[np.newaxis, :], hubert_feature.shape[0], axis=0)

        # 转换为tensor
        hubert_feature = torch.from_numpy(hubert_feature)
        pitch_feature = torch.from_numpy(pitch_feature)
        speaker_feature = torch.from_numpy(speaker_feature)
        ppg_feature = torch.from_numpy(ppg_feature)

        # 合并特征（可选）
        # combined_feature = torch.cat([hubert_feature, ppg_feature, speaker_feature, pitch_feature.unsqueeze(1)], dim=1)

        # return combined_feature

        # 返回字典
        return {
            'hubert_feature': hubert_feature,
            'pitch_feature': pitch_feature,
            'speaker_feature': speaker_feature,
            'ppg_feature': ppg_feature,
            'mel_feature': mel_feature
        }

    def collate_fn(self, batch):
        # 手动处理批次中的数据填充
        hubert_features = pad_sequence([item['hubert_feature'] for item in batch], batch_first=True, padding_value=0)
        pitch_features = pad_sequence([item['pitch_feature'].unsqueeze(1) for item in batch], batch_first=True,
                                      padding_value=0).squeeze(2)
        speaker_features = pad_sequence([item['speaker_feature'] for item in batch], batch_first=True, padding_value=0)
        ppg_features = pad_sequence([item['ppg_feature'] for item in batch], batch_first=True, padding_value=0)

        # 找出最大长度
        max_length = max([item['mel_feature'].shape[1] for item in batch])  # 获取最长的时间维度长度

        # 填充mel_feature到最大长度
        padded_mel_features = [F.pad(item['mel_feature'], (0, max_length - item['mel_feature'].shape[1]), "constant", 0)
                               for item in batch]

        mel_features = pad_sequence(padded_mel_features, batch_first=True,
                                    padding_value=0)  # 尽管所有mel_feature都已经是相同长度，仍然使用pad_sequence确保一致的处理

        return {
            'hubert_feature': hubert_features,
            'pitch_feature': pitch_features,
            'speaker_feature': speaker_features,
            'ppg_feature': ppg_features,
            'mel_feature': mel_features
        }


if __name__ == "__main__":
    # 假设你已经设置好了特征文件夹路径
    hubert_dir = 'data_svc/hubert/speaker1'
    pitch_dir = 'data_svc/pitch/speaker1'
    speaker_dir = 'data_svc/speaker/speaker1'
    ppg_dir = 'data_svc/whisper/speaker1'
    mel_dir = 'data_svc/specs/speaker1'

    dataset = AudioFeatureDataset(hubert_dir, pitch_dir, speaker_dir, ppg_dir, mel_dir)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=dataset.collate_fn)

    for batch in data_loader:
        # 打印各个特征
        print("HuBERT Feature:", batch['hubert_feature'].shape, batch['hubert_feature'])
        print("Pitch Feature:", batch['pitch_feature'].shape, batch['pitch_feature'])
        print("Speaker Feature:", batch['speaker_feature'].shape, batch['speaker_feature'])
        print("PPG Feature:", batch['ppg_feature'].shape, batch['ppg_feature'])
        print('mel specs:', batch['mel_feature'].shape, batch['mel_feature'])
        print()  # 添加空行以便于区分不同的批次

        # 这里可以继续添加训练模型的代码
        # 例如：output = model(batch['hubert_feature'], ...)
        pass

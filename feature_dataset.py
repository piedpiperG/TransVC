import os
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torchaudio  # 导入torchaudio处理音频


class AudioFeatureDataset(Dataset):
    def __init__(self, hubert_dir, pitch_dir, speaker_dir, ppg_dir, mel_dir, wav_dir):
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
        self.wav_files = [os.path.join(wav_dir, file) for file in os.listdir(wav_dir)]  # 新增wav文件目录

        # 确保文件对齐，这里简单假设文件顺序相同且一一对应
        self.hubert_files.sort()
        self.pitch_files.sort()
        self.speaker_files.sort()
        self.ppg_files.sort()
        self.mel_files.sort()
        self.wav_files.sort()

    def __len__(self):
        return len(self.hubert_files)

    def __getitem__(self, idx):
        # 读取特征，确保在加载时允许pickle
        hubert_feature = np.load(self.hubert_files[idx], allow_pickle=True).astype(np.float32)
        pitch_feature = np.load(self.pitch_files[idx], allow_pickle=True).astype(np.float32)
        speaker_feature = np.load(self.speaker_files[idx], allow_pickle=True).astype(np.float32)
        ppg_feature = np.load(self.ppg_files[idx], allow_pickle=True).astype(np.float32)
        mel_feature = torch.load(self.mel_files[idx])  # 加载.pt格式的Mel频谱
        wave_tensor, sample_rate = torchaudio.load(self.wav_files[idx])  # 加载wav文件为波形张量
        # wave_tensor = wave_tensor.mean(dim=0)  # 如果是立体声，转换为单声道

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

        # 返回字典
        return {
            'hubert_feature': hubert_feature,
            'pitch_feature': pitch_feature,
            'speaker_feature': speaker_feature,
            'ppg_feature': ppg_feature,
            'mel_feature': mel_feature,
            'waveform': wave_tensor  # 返回wav文件的波形数据
        }

    def collate_fn(self, batch):
        # 手动处理批次中的数据填充
        hubert_features = pad_sequence([item['hubert_feature'] for item in batch], batch_first=True, padding_value=0)
        pitch_features = pad_sequence([item['pitch_feature'].unsqueeze(1) for item in batch], batch_first=True,
                                      padding_value=0).squeeze(2)
        speaker_features = pad_sequence([item['speaker_feature'] for item in batch], batch_first=True, padding_value=0)
        ppg_features = pad_sequence([item['ppg_feature'] for item in batch], batch_first=True, padding_value=0)

        # 找出批次中所有特征时间维度的最小值
        min_time_length = min(hubert_features.shape[1], pitch_features.shape[1], speaker_features.shape[1])

        # 裁剪所有特征到最小时间维度
        hubert_features = hubert_features[:, :min_time_length, :]
        pitch_features = pitch_features[:, :min_time_length]
        speaker_features = speaker_features[:, :min_time_length, :]
        # 对PPG特征进行填充或裁剪
        ppg_features = pad_sequence([item['ppg_feature'] for item in batch], batch_first=True, padding_value=0)
        ppg_features = ppg_features[:, :min_time_length, :]

        # 找出最大长度
        max_length = max([item['mel_feature'].shape[1] for item in batch])  # 获取最长的时间维度长度

        # 填充mel_feature到最大长度
        padded_mel_features = [F.pad(item['mel_feature'], (0, max_length - item['mel_feature'].shape[1]), "constant", 0)
                               for item in batch]

        mel_features = pad_sequence(padded_mel_features, batch_first=True,
                                    padding_value=0)  # 尽管所有mel_feature都已经是相同长度，仍然使用pad_sequence确保一致的处理

        # 找出波形数据中的最小长度
        min_waveform_length = min([item['waveform'].shape[1] for item in batch])

        # 剪切波形到最小长度
        waveforms = pad_sequence([item['waveform'][:, :min_waveform_length] for item in batch], batch_first=True,
                                 padding_value=0)

        return {
            'hubert_feature': hubert_features,
            'pitch_feature': pitch_features,
            'speaker_feature': speaker_features,
            'ppg_feature': ppg_features,
            'mel_feature': mel_features,
            'waveform': waveforms  # 返回处理后的波形批次数据
        }


if __name__ == "__main__":
    # 假设你已经设置好了特征文件夹路径
    hubert_dir = 'data_svc/hubert/speaker1'
    pitch_dir = 'data_svc/pitch/speaker1'
    speaker_dir = 'data_svc/speaker/speaker1'
    ppg_dir = 'data_svc/whisper/speaker1'
    mel_dir = 'data_svc/specs/speaker1'
    wav_dir = 'data_svc/waves-32k/speaker1'

    dataset = AudioFeatureDataset(hubert_dir, pitch_dir, speaker_dir, ppg_dir, mel_dir, wav_dir)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)

    for batch in data_loader:
        # 打印各个特征
        print("HuBERT Feature:", batch['hubert_feature'].shape)
        print("Pitch Feature:", batch['pitch_feature'].shape)
        print("Speaker Feature:", batch['speaker_feature'].shape)
        print("PPG Feature:", batch['ppg_feature'].shape)
        print('mel specs:', batch['mel_feature'].shape)
        print('waveforms:', batch['waveform'].shape)
        print()  # 添加空行以便于区分不同的批次

        # 这里可以继续添加训练模型的代码
        # 例如：output = model(batch['hubert_feature'], ...)
        pass

import os, sys

import librosa
import soundfile as sf
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam
import argparse
from omegaconf import OmegaConf
import torch.nn.functional as F

from feature_dataset import AudioFeatureDataset
from Transformer_encoder import SpeechFeatureTransformer
from Transformer_decoder import SpeechFeatureDecoder
from vits import commons
from vits_extend.stft import TacotronSTFT

# 数据加载
hubert_dir = 'data_svc/hubert/speaker1'
pitch_dir = 'data_svc/pitch/speaker1'
speaker_dir = 'data_svc/speaker/speaker1'
ppg_dir = 'data_svc/whisper/speaker1'
mel_dir = 'data_svc/specs/speaker1'
wav_dir = 'data_svc/waves-16k/speaker1'

dataset = AudioFeatureDataset(hubert_dir, pitch_dir, speaker_dir, ppg_dir, mel_dir, wav_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=False, default='configs/base.yaml',
                    help="yaml file for configuration")

args = parser.parse_args()

hp = OmegaConf.load(args.config)
with open(args.config, 'r') as f:
    hp_str = ''.join(f.readlines())

# 模型定义
feature_sizes = {'hubert': 256, 'ppg': 1280, 'speaker': 256, 'pitch': 1, 'combined': 192}
encoder = SpeechFeatureTransformer(feature_sizes, nhead=8, num_encoder_layers=12, dim_feedforward=2048)
decoder = SpeechFeatureDecoder(feature_size=192, nhead=8, num_decoder_layers=12, dim_feedforward=2048, mel_bins=513,
                               hp=hp)

# 将模型放到GPU上（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
encoder.to(device)
decoder.to(device)

encoder_optimizer = Adam(encoder.parameters(), lr=0.0001)
decoder_optimizer = Adam(decoder.parameters(), lr=0.0001)

loss_function = torch.nn.MSELoss()  # 定义损失函数

stft = TacotronSTFT(filter_length=hp.data.filter_length,
                    hop_length=hp.data.hop_length,
                    win_length=hp.data.win_length,
                    n_mel_channels=hp.data.mel_channels,
                    sampling_rate=hp.data.sampling_rate,
                    mel_fmin=hp.data.mel_fmin,
                    mel_fmax=hp.data.mel_fmax,
                    center=False,
                    device=device)

# 创建目录来保存音频文件（如果不存在的话）
output_dir = 'output_audio'
os.makedirs(output_dir, exist_ok=True)

# 训练循环
num_epochs = 30
for epoch in range(num_epochs):
    epoch_loss = 0
    losses = []  # 列表来存储每个epoch的平均损失
    for batch in dataloader:
        hubert = batch['hubert_feature'].to(device)
        pitch = batch['pitch_feature'].to(device)
        speaker = batch['speaker_feature'].to(device)  # 源说话人特征
        ppg = batch['ppg_feature'].to(device)
        mel = batch['mel_feature'].to(device)
        wav = batch['waveform'].to(device)

        target_speaker = speaker  # 目标说话人特征，需要修改为实际的目标说话人特征

        # 编码阶段
        encoder_optimizer.zero_grad()
        encoded_features = encoder(hubert, pitch, ppg)
        encoded_features = encoded_features.permute(1, 0, 2)

        # print(f'encoded_features shape: {encoded_features.shape}')
        # 解码阶段
        decoder_optimizer.zero_grad()
        fake_audio, ids_str = decoder(encoded_features, target_speaker, pitch, wav.size(2))

        # print(f'fake_audio shape: {fake_audio.shape}')
        # print(f'ids_str: {ids_str}')
        # print(f'wav shape: {wav.shape}')
        # print(wav.size(2))

        audio = commons.slice_segments(
            wav, ids_str * hp.data.hop_length, hp.data.segment_size)  # slice

        # print(f'audio shape: {audio.shape}')

        mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))
        mel_real = stft.mel_spectrogram(audio.squeeze(1))

        # print(f'mel_fake shape:{mel_fake.shape}')
        # print(f'mel_real shape:{mel_real.shape}')

        # print()

        mel_loss = F.l1_loss(mel_fake, mel_real) * hp.train.c_mel

        # 反向传播更新模型
        mel_loss.backward()  # 反向传播计算梯度
        encoder_optimizer.step()  # 更新编码器参数
        decoder_optimizer.step()  # 更新解码器参数

        # 累积损失以便于观察
        epoch_loss += mel_loss.item()

        average_loss = epoch_loss / len(dataloader)
        losses.append(average_loss)

    print(f'Epoch {epoch + 1}, Average Loss: {epoch_loss / len(dataloader)}')

print("Training complete.")

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'training_loss_curve.png'))  # 保存损失曲线图像
plt.show()  # 显示图像

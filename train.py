import os, sys

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import argparse
from omegaconf import OmegaConf

from feature_dataset import AudioFeatureDataset
from Transformer_encoder import SpeechFeatureTransformer
from Transformer_decoder import SpeechFeatureDecoder

# 数据加载
hubert_dir = 'data_svc/hubert/speaker1'
pitch_dir = 'data_svc/pitch/speaker1'
speaker_dir = 'data_svc/speaker/speaker1'
ppg_dir = 'data_svc/whisper/speaker1'
mel_dir = 'data_svc/specs/speaker1'

dataset = AudioFeatureDataset(hubert_dir, pitch_dir, speaker_dir, ppg_dir, mel_dir)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=False, default='configs/base.yaml',
                    help="yaml file for configuration")

args = parser.parse_args()

hp = OmegaConf.load(args.config)
with open(args.config, 'r') as f:
    hp_str = ''.join(f.readlines())

# 模型定义
feature_sizes = {'hubert': 256, 'ppg': 1280, 'speaker': 256, 'pitch': 1, 'combined': 192}
encoder = SpeechFeatureTransformer(feature_sizes, nhead=8, num_encoder_layers=6, dim_feedforward=2048)
decoder = SpeechFeatureDecoder(feature_size=192, nhead=8, num_decoder_layers=6, dim_feedforward=2048, mel_bins=513,
                               hp=hp)

# 将模型放到GPU上（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
encoder.to(device)
decoder.to(device)

encoder_optimizer = Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = Adam(decoder.parameters(), lr=0.001)

loss_function = torch.nn.MSELoss()  # 定义损失函数

# 训练循环
num_epochs = 30
for epoch in range(num_epochs):
    for batch in dataloader:
        hubert = batch['hubert_feature'].to(device)
        pitch = batch['pitch_feature'].to(device)
        speaker = batch['speaker_feature'].to(device)  # 源说话人特征
        ppg = batch['ppg_feature'].to(device)
        mel = batch['mel_feature'].to(device)

        target_speaker = speaker  # 目标说话人特征，需要修改为实际的目标说话人特征

        # 编码阶段
        encoder_optimizer.zero_grad()
        encoded_features = encoder(hubert, pitch, ppg)
        encoded_features = encoded_features.permute(1, 0, 2)

        # print(f'encoded_features shape: {encoded_features.shape}')
        # 解码阶段
        decoder_optimizer.zero_grad()
        audio, ids_str = decoder(encoded_features, target_speaker, pitch)

        print(audio)
        print(ids_str)


print("Training complete.")

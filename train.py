import os

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

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
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn)

# 模型定义
feature_sizes = {'hubert': 256, 'ppg': 1280, 'speaker': 256, 'pitch': 1, 'combined': 512}
encoder = SpeechFeatureTransformer(feature_sizes, nhead=8, num_encoder_layers=6, dim_feedforward=2048)
decoder = SpeechFeatureDecoder(feature_size=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048, mel_bins=513)

# 将模型放到GPU上（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        encoded_features = encoder(hubert, pitch, speaker, ppg)
        encoded_features = encoded_features.permute(1, 0, 2)

        # print(f'target_speaker: {target_speaker.shape}')
        # print(f'encoded_features: {encoded_features.shape}')
        # 解码阶段
        decoder_optimizer.zero_grad()
        decoded_features = decoder(encoded_features, target_speaker)
        # 使用torch的插值方法扩展decoded_features
        decoded_expanded = torch.nn.functional.interpolate(decoded_features, size=mel.shape[2], mode='linear',
                                                           align_corners=False)

        # print(f'decoded_features shape: {decoded_features.shape}')
        print(f'decoded_expanded shape: {decoded_expanded.shape}')
        # print(f'mel_features shape: {mel.shape}')

        # 计算损失
        loss = loss_function(decoded_expanded, mel)
        loss.backward()  # 反向传播
        encoder_optimizer.step()
        decoder_optimizer.step()

        print(f'Epoch: {epoch}, Loss: {loss.item()}')

        # 如果是最后一个epoch，则保存生成的mel频谱
        # if epoch == num_epochs - 1:
        #     decoded_np = decoded_expanded.detach().cpu().numpy()  # 先转换为numpy数组
        #     resampled_mels = np.empty((decoded_expanded.shape[0], 80, decoded_expanded.shape[2]))
        #
        #     for i in range(decoded_expanded.shape[0]):
        #         for t in range(decoded_expanded.shape[2]):
        #             resampled_mels[i, :, t] = librosa.resample(decoded_np[i, :, t], orig_sr=513, target_sr=80)
        #
        #     # 为每个mel频谱创建独立的保存路径，并保存
        #     for i in range(resampled_mels.shape[0]):
        #         save_path = os.path.join('saved_mels', f'mel_{i}.npy')
        #         os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #         np.save(save_path, resampled_mels[i:i + 1, :, :])  # 保存单个mel频谱为.npy
        #         print(f'Saved mel spectrum {i} with shape: {resampled_mels[i:i + 1, :, :].shape}')

        # 如果是最后一个epoch，则保存生成的mel频谱
        if epoch == num_epochs - 1:
            decoded_np = mel.detach().cpu().numpy()  # 先转换为numpy数组
            resampled_mels = np.empty((mel.shape[0], 80, mel.shape[2]))

            for i in range(mel.shape[0]):
                for t in range(mel.shape[2]):
                    resampled_mels[i, :, t] = librosa.resample(decoded_np[i, :, t], orig_sr=513, target_sr=80)

            # 为每个mel频谱创建独立的保存路径，并保存
            for i in range(resampled_mels.shape[0]):
                save_path = os.path.join('saved_mels', f'mel_{i}.npy')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, resampled_mels[i:i + 1, :, :])  # 保存单个mel频谱为.npy
                print(f'Saved mel spectrum {i} with shape: {resampled_mels[i:i + 1, :, :].shape}')

print("Training complete.")

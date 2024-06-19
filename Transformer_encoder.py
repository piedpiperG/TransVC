import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SpeechFeatureTransformer(nn.Module):
    def __init__(self, feature_sizes, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(SpeechFeatureTransformer, self).__init__()
        self.hubert_proj = nn.Linear(feature_sizes['hubert'], feature_sizes['combined'])
        self.ppg_proj = nn.Linear(feature_sizes['ppg'], feature_sizes['combined'])
        self.speaker_proj = nn.Linear(feature_sizes['speaker'], feature_sizes['combined'])
        self.pitch_proj = nn.Linear(1, feature_sizes['combined'])  # Pitch is a single value per timestep

        self.pos_encoder = PositionalEncoding(feature_sizes['combined'])
        encoder_layers = TransformerEncoderLayer(d_model=feature_sizes['combined'], nhead=nhead,
                                                 dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

    def forward(self, hubert, pitch, speaker, ppg):
        # Project all features to the same dimension
        hubert = self.hubert_proj(hubert)
        pitch = self.pitch_proj(pitch.unsqueeze(-1))  # Add an extra dimension to pitch for linear layer
        pitch = torch.squeeze(pitch, dim=2)  # Remove the extra dimension
        speaker = self.speaker_proj(speaker)
        ppg = self.ppg_proj(ppg)

        # print(f'hubert: {hubert.shape}')
        # print(f'pitch: {pitch.shape}')
        # print(f'speaker: {speaker.shape}')
        # print(f'ppg: {ppg.shape}')

        # Combine features
        combined_features = hubert + pitch + speaker + ppg  # Element-wise addition
        combined_features = self.pos_encoder(combined_features)
        output = self.transformer_encoder(combined_features)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


if __name__ == '__main__':
    # 设置输入的特征大小和其他参数
    feature_sizes = {
        'hubert': 256,
        'ppg': 1280,
        'speaker': 256,
        'pitch': 1,
        'combined': 512  # 投影后的维度大小
    }

    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型实例
    model = SpeechFeatureTransformer(feature_sizes, nhead=8, num_encoder_layers=6, dim_feedforward=2048)
    model = model.to(device)

    # 创建合成数据
    batch_size = 2
    seq_length = 510

    hubert_input = torch.randn(batch_size, seq_length, feature_sizes['hubert']).to(device)
    ppg_input = torch.randn(batch_size, seq_length, feature_sizes['ppg']).to(device)
    speaker_input = torch.randn(batch_size, seq_length, feature_sizes['speaker']).to(device)
    pitch_input = torch.randn(batch_size, seq_length, 1).to(device)

    # 前向传播
    output = model(hubert_input, pitch_input, speaker_input, ppg_input)
    # output = output.permute(1, 0, 2)

    # 输出结果
    print("Output shape:", output.shape)

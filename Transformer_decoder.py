from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.nn as nn
from Transformer_encoder import PositionalEncoding
import torch


class SpeechFeatureDecoder(nn.Module):
    def __init__(self, feature_size, nhead, num_decoder_layers, dim_feedforward, mel_bins, dropout=0.1):
        super(SpeechFeatureDecoder, self).__init__()
        # Positional Encoding for adding temporal dynamics
        self.pos_decoder = PositionalEncoding(feature_size)

        # Decoder layers
        decoder_layer = TransformerDecoderLayer(d_model=feature_size, nhead=nhead,
                                                dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        # Projection layers to potentially transform the decoded features into a different feature space
        self.output_proj = nn.Linear(feature_size,
                                     feature_size)  # Adjust according to the output dimensionality required
        # Adjust dimensionality of target speaker features to match encoded features
        self.target_speaker_proj = nn.Linear(256, feature_size)  # new line here

        # Mel Spectrogram Generator
        self.mel_generator = nn.Sequential(
            nn.Linear(feature_size, feature_size * 2),
            nn.ReLU(),
            nn.Linear(feature_size * 2, mel_bins)  # 假设输出Mel频谱的带数为mel_bins
        )

    def forward(self, encoded_features, target_speaker_features):
        target_speaker_features = target_speaker_features.permute(1, 0, 2)
        # Adjust dimensions
        target_speaker_features = self.target_speaker_proj(target_speaker_features)

        # Now we can safely add since both tensors match in all dimensions
        decoder_input = encoded_features + target_speaker_features

        # 添加位置编码
        decoder_input = self.pos_decoder(decoder_input)

        # 正确调用解码器，提供tgt和memory
        output = self.transformer_decoder(tgt=decoder_input, memory=encoded_features)
        output = self.output_proj(output)

        # Generate Mel spectrogram
        mel_spectrogram = self.mel_generator(output)
        return mel_spectrogram.permute(1, 2, 0)  # 调整为 [batch_size, mel_bins, time_steps]


if __name__ == '__main__':
    # 假设 encoded_features 和 target_speaker_features 是已经准备好的张量
    batch_size = 2
    time_steps = 510
    feature_size = 512
    mel_bins = 513

    encoded_features = torch.randn(time_steps, batch_size,
                                   feature_size)  # 10 time steps, batch size 32, feature size 512
    target_speaker_features = torch.randn(batch_size, time_steps,
                                          256)  # batch size 32, feature size 512 (no time dimension)

    decoder = SpeechFeatureDecoder(feature_size=feature_size, nhead=8, num_decoder_layers=6, dim_feedforward=2048,
                                   mel_bins=mel_bins)
    mel_spectrograms = decoder(encoded_features, target_speaker_features)  # [batch_size, time_steps, mel_bins]
    print(mel_spectrograms.shape)

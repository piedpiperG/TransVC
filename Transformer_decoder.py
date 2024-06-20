from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.nn as nn
from Transformer_encoder import PositionalEncoding
import torch
from vits_decoder.generator import Generator
from vits import commons


class SpeechFeatureDecoder(nn.Module):
    def __init__(self, feature_size, nhead, num_decoder_layers, dim_feedforward, mel_bins, hp, dropout=0.1):
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

        self.dec = Generator(hp=hp)
        self.segment_size = hp.data.segment_size // hp.data.hop_length

        self.hp = hp

    def forward(self, encoded_features, target_speaker_features, pitch_features, wav_len):
        target_speaker_features = target_speaker_features.permute(1, 0, 2)
        # Adjust dimensions
        adjusted_speaker_features = self.target_speaker_proj(target_speaker_features)

        # Now we can safely add since both tensors match in all dimensions
        decoder_input = encoded_features + adjusted_speaker_features

        # 添加位置编码
        decoder_input = self.pos_decoder(decoder_input)

        # 正确调用解码器，提供tgt和memory
        output = self.transformer_decoder(tgt=decoder_input, memory=encoded_features)
        output = self.output_proj(output)
        output = output.permute(1, 2, 0)

        target_speaker_features = target_speaker_features.permute(1, 0, 2)
        target_speaker_features = torch.mean(target_speaker_features, dim=1)

        # print(f'target_speaker_features shape:{target_speaker_features.shape}')
        # print(f'output shape:{output.shape}')
        # print(f'pitch_features shape:{pitch_features.shape}')

        x_slice, p_slice, ids_str = commons.rand_slice_segments_with_pitch(x=output, pitch=pitch_features,
                                                                           x_lengths=wav_len / self.hp.data.hop_length,
                                                                           segment_size=self.segment_size)

        # print(f'x_slice shape: {x_slice.shape}')
        # print(f'p_slice shape: {p_slice.shape}')
        # print(f'ids_str_before: {ids_str}')

        audio = self.dec(target_speaker_features, x_slice, p_slice)

        # print(f'audio shape:{audio.shape}')
        # print(f'ids_str_after:{ids_str}')
        # print()

        return audio, ids_str


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

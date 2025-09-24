"""
DTLN (Dual-signal Transformation LSTM Network) for real-time speech enhancement.

Reference: Westhausen & Meyer, "Dual-Signal Transformation LSTM Network for
Real-Time Noise Suppression", Interspeech 2020.

Architecture in one sentence: two cascaded mask-estimation stages, each an
LSTM stack, the first operating on an FFT magnitude representation (good at
suppressing stationary noise) and the second on a learned encoder representation
of the time-domain signal (good at cleaning up what the first stage missed).
Both stages are *causal* (no future context) so the model can run frame-by-frame
in real time.

For TRAINING we process whole utterances with batched STFT (fast, parallel).
For REAL-TIME INFERENCE the same weights are run frame-by-frame with LSTM
hidden state carried across frames (see src/streaming.py) - this is what
gets exported to ONNX and what actually matters for the "real-time on edge
devices" requirement.
"""
import torch
import torch.nn as nn


class InstantLayerNorm(nn.Module):
    """Normalizes across the feature dimension only (not time) so it can be
    applied causally, one frame at a time, without looking at future frames."""

    def __init__(self, channels, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # x: (batch, time, channels)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta


class SeparationCore(nn.Module):
    """Shared building block for both DTLN stages: 2-layer LSTM -> FC -> sigmoid mask."""

    def __init__(self, feature_dim, hidden_dim=128, dropout=0.25):
        super().__init__()
        self.norm = InstantLayerNorm(feature_dim)
        self.lstm1 = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x, h1=None, h2=None):
        # x: (batch, time, feature_dim)
        x = self.norm(x)
        x, h1 = self.lstm1(x, h1)
        x = self.dropout(x)
        x, h2 = self.lstm2(x, h2)
        x = self.dropout(x)
        mask = torch.sigmoid(self.fc(x))
        return mask, h1, h2


class DTLN(nn.Module):
    """
    Args:
        frame_len: analysis frame length in samples (512 @ 16kHz = 32ms)
        frame_hop: hop size in samples (128 @ 16kHz = 8ms, 75% overlap)
        encoder_dim: number of learned filters in the stage-2 encoder
        hidden_dim: LSTM hidden size in each stage
    """

    def __init__(self, frame_len=512, frame_hop=128, encoder_dim=256, hidden_dim=128):
        super().__init__()
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.encoder_dim = encoder_dim
        n_freq = frame_len // 2 + 1

        # Stage 1: operates on FFT magnitude
        self.stage1 = SeparationCore(n_freq, hidden_dim)

        # Stage 2: learned conv encoder/decoder ("basis" representation),
        # operating on the time-domain output of stage 1
        self.encoder = nn.Conv1d(1, encoder_dim, kernel_size=frame_len,
                                  stride=frame_hop, bias=False)
        self.decoder = nn.ConvTranspose1d(encoder_dim, 1, kernel_size=frame_len,
                                           stride=frame_hop, bias=False)
        self.stage2 = SeparationCore(encoder_dim, hidden_dim)

        self.register_buffer("window", torch.hann_window(frame_len))

    def forward(self, noisy_wav):
        """
        noisy_wav: (batch, num_samples) raw waveform at 16kHz
        returns: enhanced waveform, same shape (modulo frame padding)
        """
        batch = noisy_wav.shape[0]

        # ---- Stage 1: FFT-domain masking ----
        stft = torch.stft(
            noisy_wav, n_fft=self.frame_len, hop_length=self.frame_hop,
            window=self.window, return_complex=True, center=True,
        )  # (batch, freq, time)
        mag = torch.abs(stft).transpose(1, 2)   # (batch, time, freq)
        phase = torch.angle(stft)

        mask1, _, _ = self.stage1(mag)
        est_mag = mag * mask1

        est_complex = (est_mag.transpose(1, 2) * torch.exp(1j * phase))
        stage1_out = torch.istft(
            est_complex, n_fft=self.frame_len, hop_length=self.frame_hop,
            window=self.window, center=True, length=noisy_wav.shape[-1],
        )  # (batch, num_samples)

        # ---- Stage 2: learned-encoder masking ----
        x = stage1_out.unsqueeze(1)             # (batch, 1, samples)
        encoded = self.encoder(x)                # (batch, encoder_dim, frames)
        encoded_t = encoded.transpose(1, 2)       # (batch, frames, encoder_dim)

        mask2, _, _ = self.stage2(encoded_t)
        masked = encoded * mask2.transpose(1, 2)

        enhanced = self.decoder(masked).squeeze(1)  # (batch, samples)

        # pad/trim to match input length (conv/transpose-conv edge effects)
        if enhanced.shape[-1] < noisy_wav.shape[-1]:
            enhanced = nn.functional.pad(enhanced, (0, noisy_wav.shape[-1] - enhanced.shape[-1]))
        else:
            enhanced = enhanced[..., :noisy_wav.shape[-1]]

        return enhanced


if __name__ == "__main__":
    # quick sanity check
    model = DTLN()
    dummy = torch.randn(2, 16000)  # 2 utterances, 1 second @ 16kHz
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters:   {n_params:,}")

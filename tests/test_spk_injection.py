import torch
from dia.model import DiaModel
from tools.speaker_encoder import SpeakerEncoder

# Mock config
class DummyConfig:
    class decoder:
        d_model = 256
    model = decoder()

# Initialize model
model = DiaModel(config=DummyConfig())
model.eval()

# Simulate input
B, T, C = 2, 16, 256
dummy_audio_tokens = torch.randint(0, 100, (B, T, C)).float()
dummy_spk_embed = torch.randn(B, 192)

# Forward pass
with torch.no_grad():
    out = model.decoder_forward(dummy_audio_tokens, spk_embed=dummy_spk_embed)

print("✅ Decoder output shape:", out.shape)
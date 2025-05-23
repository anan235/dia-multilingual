import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

class SpeakerEncoder:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = EncoderClassifier.from_hparams(
            # Conformer‑Based Multilingual Speaker Encoder
            # Model: tflite-hub/conformer-speaker-encoder
            # Architecture: Conformer | Loss: GE2E | Languages: 100+
            source="tflite-hub/conformer-speaker-encoder",
            run_opts={"device": device}
        )

    def encode(self, wav_path):
        signal, fs = torchaudio.load(wav_path)
        if fs != 16000:
            signal = torchaudio.functional.resample(signal, fs, 16000)
        embedding = self.model.encode_batch(signal).squeeze(0)
        return embedding.detach()

# Usage:
# enc = SpeakerEncoder()
# spk_vec = enc.encode("sample.wav")
import torchaudio
import torchaudio.transforms as T
from pathlib import Path

def normalize_and_trim(path, out_path, target_db=-20):
    wav, sr = torchaudio.load(path)
    vol = T.Vol(target_db, gain_type='db')
    wav = vol(wav)
    trimmed = T.Vad(sample_rate=sr)(wav)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(out_path, trimmed, sr)

# Example usage:
# normalize_and_trim("input.wav", "cleaned/input.wav")
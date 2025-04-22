import argparse
import json
import torch
import torchaudio
from pathlib import Path
from dia.model import DiaModel
from subprocess import check_output

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
parser.add_argument("--lang_vocab")
parser.add_argument("--output_dir")
parser.add_argument("--text")
parser.add_argument("--lang")
parser.add_argument("--reference_wav", default=None)
parser.add_argument("--sample_rate", type=int, default=22050)
args = parser.parse_args()

lang_vocab = json.load(open(args.lang_vocab))
model = DiaModel.load_from_checkpoint(args.model_path, map_location="cpu")
model.eval()

lang_token = f"<{args.lang}>"
lang_token_id = lang_vocab.get(lang_token, 0)

ipa = check_output(f"echo '{args.text}' | espeak-ng -v {args.lang} --ipa -q", shell=True, text=True).strip()
print("Phonemes:", ipa)

char_vocab = {c: i+10 for i, c in enumerate(sorted(set(ipa)))}
char_vocab["<pad>"] = 0
char_vocab["<unk>"] = 1
phoneme_ids = [char_vocab.get(c, 1) for c in ipa]
input_ids = torch.tensor([lang_token_id] + phoneme_ids).unsqueeze(0)

ref = None
if args.reference_wav:
    wav, sr = torchaudio.load(args.reference_wav)
    if sr != args.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, args.sample_rate)
    ref = wav.squeeze(0).unsqueeze(0)

with torch.no_grad():
    audio = model.infer(input_ids=input_ids, ref_audio=ref)

Path(args.output_dir).mkdir(parents=True, exist_ok=True)
outpath = Path(args.output_dir) / f"tts_{args.lang}.wav"
torchaudio.save(str(outpath), audio.cpu(), args.sample_rate)
print("Saved:", outpath)
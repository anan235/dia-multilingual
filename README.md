# DIA-Multilingual (StyleTTS2-based)

This is a fork of the original DIA model extended for **multilingual TTS**, with support for 30+ languages (same as ElevenLabs). Built on top of StyleTTS2 with language token injection, espeak-ng phonemization, and support for reference audio-based style transfer.

---

## 🧠 Supported Languages

Supports over 30 languages via `<lang>` token injection, including:

`en`, `es`, `de`, `fr`, `it`, `pt`, `pl`, `ro`, `nl`, `tr`, `sv`, `cs`, `el`, `hu`, `fi`, `da`, `sk`, `bg`, `hi`, `ar`, `zh`, `ja`, `ko`, ...

---

## 🚀 Quickstart (RunPod)

**Build your container:**
```bash
docker build -t dia-multilang -f docker/Dockerfile .
```

**Launch training (inside container):**
```bash
bash docker/launch.sh
```

This will:
- Load `train_manifest.json` and `lang_vocab.json`
- Start training from scratch using espeak-based phoneme inputs
- Save checkpoints in `/workspace/checkpoints`

---

## 🧾 File Structure

```bash
├── dataset/
│   └── multilang_tts_dataset.py     # Dataset + phonemizer + collate_fn
├── scripts/
│   ├── train_dia.py                 # Main training loop
│   ├── validate.py                  # Eval script (loss only)
│   └── infer_dia.py                 # Generate audio from text
├── docker/
│   ├── Dockerfile                   # GPU-enabled training container
│   └── launch.sh                    # Entrypoint script
├── lang_vocab.json                  # Maps <lang> → token_id
├── train_manifest.json              # Manifest (audio, text, lang)
```

---

## 🎙️ Inference

Generate audio from text using:

```bash
python3 scripts/infer_dia.py \
  --model_path checkpoints/epoch49.pt \
  --lang_vocab lang_vocab.json \
  --text "Ciao, come stai?" \
  --lang it \
  --output_dir samples/
```

To use reference audio (zero-shot style cloning):

```bash
  --reference_wav samples/italian_female.wav
```

---

## 💡 Notes

- Uses espeak-ng to phonemize all input text (per-language IPA)
- Pretrained `xlm-roberta-base` recommended for phoneme encodings
- Output speech is high-fidelity and respects cross-language style transfer

---

## 🧠 Credits
Built on top of:
- [DIA](https://github.com/nari-labs/dia)
- [StyleTTS2](https://github.com/yl4579/StyleTTS2)
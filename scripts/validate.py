import torch
from torch.utils.data import DataLoader
from dataset.multilang_tts_dataset import MultilangTTSDataset, collate_fn
from dia.model import DiaModel
import json

lang_vocab = json.load(open("lang_vocab.json"))
dataset = MultilangTTSDataset("valid_manifest.json", lang_vocab)
dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

model = DiaModel.load_from_checkpoint("checkpoints/best.pth")
model.eval()

losses = []
for batch in dataloader:
    with torch.no_grad():
        loss = model(batch)
        losses.append(loss.item())

print("Avg validation loss:", sum(losses)/len(losses))
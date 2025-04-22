import argparse, json
from torch.utils.data import DataLoader
from dataset.multilang_tts_dataset import MultilangTTSDataset, collate_fn
from dia.model import DiaModel
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--manifest")
parser.add_argument("--lang_vocab")
parser.add_argument("--data_root")
parser.add_argument("--output_dir")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

lang_vocab = json.load(open(args.lang_vocab))
dataset = MultilangTTSDataset(args.manifest, lang_vocab)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)

model = DiaModel(lang_vocab=lang_vocab)  # Adjust if constructor differs
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

os.makedirs(args.output_dir, exist_ok=True)

for epoch in range(args.epochs):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    ckpt_path = os.path.join(args.output_dir, f"epoch{epoch}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print("Saved:", ckpt_path)
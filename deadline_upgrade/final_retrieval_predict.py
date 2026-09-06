import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from geoguesser.checkpoint import load_checkpoint


class HoldoutDataset(Dataset):
    def __init__(self, root, size):
        self.root = Path(root)
        self.names = sorted(
            p.name for p in self.root.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        self.tf = T.Compose([
            T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        with Image.open(self.root / self.names[i]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            x = self.tf(im)
        return self.names[i], x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda")
    model, _, _, bundle = load_checkpoint(args.checkpoint, device)
    model.eval()

    size = int(bundle["metadata"]["image_size"])

    with np.load(args.bank, allow_pickle=False) as z:
        bank_emb = z["embedding"].astype(np.float32)
        bank_coord = z["truth"].astype(np.float32)

    bank_emb /= np.clip(
        np.linalg.norm(bank_emb, axis=1, keepdims=True),
        1e-12,
        None,
    )
    bank = torch.from_numpy(bank_emb).to(device)

    ds = HoldoutDataset(
        Path(args.data) / "holdout_public",
        size,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    names = []
    predictions = []

    with torch.inference_mode():
        for batch_names, images in dl:
            output = model(images.to(device))
            q = F.normalize(output["embedding"].float(), dim=1)

            nearest = (q @ bank.T).argmax(dim=1).cpu().numpy()

            names.extend(batch_names)
            predictions.append(bank_coord[nearest])

    predictions = np.concatenate(predictions)

    out = pd.DataFrame({
        "filename": names,
        "pred_lat": predictions[:, 0],
        "pred_lng": predictions[:, 1],
    })

    assert len(out) == 2400
    assert out.filename.nunique() == 2400

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print("Saved:", args.out)
    print("Rows:", len(out))


if __name__ == "__main__":
    main()

import argparse
import os

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from lightmnet3 import LightMNet


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_label_dir(root):
    for dir_name in ("label", "OUT", "out", "mask", "masks"):
        candidate = os.path.join(root, dir_name)
        if os.path.isdir(candidate):
            return candidate
    return ""


def build_transform():
    return A.Compose(
        [
            A.Normalize(),
            ToTensorV2(),
        ],
        additional_targets={"image_b": "image"},
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no_pretrained_backbone", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def infer(model, transform, device, img_a, img_b):
    height, width = img_a.shape[:2]
    augmented = transform(image=img_a, image_b=img_b, mask=np.zeros((height, width), dtype=np.uint8))
    a_tensor = augmented["image"].unsqueeze(0).to(device, memory_format=torch.channels_last)
    b_tensor = augmented["image_b"].unsqueeze(0).to(device, memory_format=torch.channels_last)
    with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
        logits = model(a_tensor, b_tensor)
    pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    return pred * 255


def main():
    args = parse_args()
    device = resolve_device()
    transform = build_transform()
    model = LightMNet(pretrained=not args.no_pretrained_backbone).to(device)
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(memory_format=torch.channels_last)
    model.eval()

    root = os.path.join(args.data_root, args.split)
    a_dir = os.path.join(root, "A")
    b_dir = os.path.join(root, "B")
    label_dir = resolve_label_dir(root)
    os.makedirs(args.save_dir, exist_ok=True)

    for name in args.names:
        a_path = os.path.join(a_dir, name)
        b_path = os.path.join(b_dir, name)
        if not (os.path.exists(a_path) and os.path.exists(b_path)):
            raise FileNotFoundError(f"Missing A/B image: {a_path} / {b_path}")
        img_a = np.array(Image.open(a_path).convert("RGB"))
        img_b = np.array(Image.open(b_path).convert("RGB"))
        mask = infer(model, transform, device, img_a, img_b)
        out_path = os.path.join(args.save_dir, f"{os.path.splitext(name)[0]}_pred.png")
        Image.fromarray(mask, mode="L").save(out_path)
        print(f"saved: {out_path}", flush=True)
        if label_dir:
            label_path = os.path.join(label_dir, name)
            if os.path.exists(label_path):
                gt = np.array(Image.open(label_path).convert("L"))
                gt = (gt > 127).astype(np.uint8) * 255
                gt_path = os.path.join(args.save_dir, f"{os.path.splitext(name)[0]}_gt.png")
                Image.fromarray(gt, mode="L").save(gt_path)
                print(f"saved: {gt_path}", flush=True)


if __name__ == "__main__":
    main()


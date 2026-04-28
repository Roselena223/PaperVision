"""
inference.py - PaperVision: Paper Detection Inference Script
============================================================
Usage:
    python3 inference.py --model_path sessions/best_model.pth --test_dir test_images/
    python3 inference.py --model_path sessions/best_model.pth --test_dir test_images/ --threshold 0.7
"""

import os
import argparse
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import time

# ─────────────────────────────────────────────────────────
# 1. Argument Parser
# ─────────────────────────────────────────────────────────
def get_infer_args():
    parser = argparse.ArgumentParser(description="PaperVision Inference")
    parser.add_argument('--model_path',  type=str, default='sessions/best_model.pth')
    parser.add_argument('--backbone',    type=str, default='fasterrcnn_resnet50_fpn',
                        choices=['fasterrcnn_resnet50_fpn', 'fasterrcnn_mobilenet_v3'])
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--image_size',  type=int, default=512)
    parser.add_argument('--threshold',   type=float, default=0.5)
    parser.add_argument('--test_dir',    type=str, default='test_images/')
    parser.add_argument('--out_dir',     type=str, default='test_results/')
    return parser.parse_args()


# ─────────────────────────────────────────────────────────
# 2. Model Builder
# ─────────────────────────────────────────────────────────
def build_model(backbone: str, num_classes: int):
    if backbone == 'fasterrcnn_resnet50_fpn':
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model   = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    else:
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model   = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ─────────────────────────────────────────────────────────
# 3. Preprocess
# ─────────────────────────────────────────────────────────
def preprocess(image_path: str, image_size: int):
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    orig_w, orig_h = img.size
    img_resized = img.resize((image_size, image_size))
    tensor = torchvision.transforms.functional.to_tensor(img_resized)
    return tensor, img, orig_w, orig_h


# ─────────────────────────────────────────────────────────
# 4. Run inference
# ─────────────────────────────────────────────────────────
def run_inference(model, tensor, device):
    model.eval()
    with torch.no_grad():
        tensor = tensor.to(device, dtype=torch.float32)
        outputs = model([tensor])
    return outputs[0]


# ─────────────────────────────────────────────────────────
# 5. Draw results and save
# ─────────────────────────────────────────────────────────
def draw_and_save(orig_img, output, threshold, image_size, save_path, image_name):
    orig_w, orig_h = orig_img.size
    scale_x = orig_w / image_size
    scale_y = orig_h / image_size

    boxes  = output['boxes'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    labels = output['labels'].cpu().numpy()

    keep   = scores >= threshold
    boxes  = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"PaperVision — {image_name}", fontsize=15, fontweight='bold', y=1.01)

    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(orig_img)
    axes[1].set_title(
        f"Detections (threshold={threshold})  |  Found: {len(boxes)} paper(s)",
        fontsize=12
    )
    axes[1].axis("off")

    colors = ['#FF3333', '#FF8800', '#22DD22', '#3399FF', '#AA00FF',
              '#FF66AA', '#00CCCC', '#FFDD00', '#FF4400', '#00FF88']

    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1 = box[0] * scale_x
        y1 = box[1] * scale_y
        x2 = box[2] * scale_x
        y2 = box[3] * scale_y
        w  = x2 - x1
        h  = y2 - y1

        color = colors[idx % len(colors)]
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=3, edgecolor=color, facecolor='none'
        )
        axes[1].add_patch(rect)
        axes[1].text(
            x1, max(y1 - 8, 0),
            f"Paper  {score:.2f}",
            fontsize=11, fontweight='bold', color='white',
            bbox=dict(facecolor=color, alpha=0.85, pad=3, edgecolor='none')
        )

    if len(boxes) == 0:
        axes[1].text(
            0.5, 0.05, "⚠️  No paper detected above threshold",
            transform=axes[1].transAxes,
            fontsize=12, color='red', ha='center',
            bbox=dict(facecolor='white', alpha=0.8)
        )
    else:
        avg_conf = scores.mean() if len(scores) > 0 else 0.0
        axes[1].text(
            0.5, 0.03,
            f"✅  {len(boxes)} detection(s)   |   Avg confidence: {avg_conf:.2f}",
            transform=axes[1].transAxes,
            fontsize=11, color='darkgreen', ha='center',
            bbox=dict(facecolor='white', alpha=0.8)
        )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────
# 6. Summary grid
# ─────────────────────────────────────────────────────────
def save_summary_grid(result_paths, out_dir):
    n = len(result_paths)
    if n == 0:
        return

    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5))
    fig.suptitle("PaperVision — Full Test Results Summary", fontsize=16, fontweight='bold')

    if rows == 1 and cols == 1:
        axes_flat = [axes]
    elif rows == 1:
        axes_flat = list(axes)
    else:
        axes_flat = [ax for row in axes for ax in row]

    for i, path in enumerate(result_paths):
        img = Image.open(path).convert("RGB")
        axes_flat[i].imshow(img)
        axes_flat[i].axis("off")
        axes_flat[i].set_title(Path(path).stem, fontsize=9)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    summary_path = os.path.join(out_dir, "_SUMMARY_ALL_RESULTS.jpg")
    plt.savefig(summary_path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    print(f"\n📊 Summary grid saved: {summary_path}")
    return summary_path


# ─────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────
def main():
    args = get_infer_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*55}")
    print(f"  PaperVision — Inference")
    print(f"{'='*55}")
    print(f"  Device     : {device}")
    print(f"  Model      : {args.model_path}")
    print(f"  Backbone   : {args.backbone}")
    print(f"  Threshold  : {args.threshold}")
    print(f"  Test dir   : {args.test_dir}")
    print(f"  Output dir : {args.out_dir}")
    print(f"{'='*55}\n")

    print("⏳  Loading model weights...")
    model = build_model(args.backbone, args.num_classes)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("✅  Model loaded successfully!\n")

    supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    test_images = sorted([
        p for p in Path(args.test_dir).iterdir()
        if p.suffix.lower() in supported_ext
    ])

    if not test_images:
        print(f"❌  No images found in '{args.test_dir}'. Please add test images.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"📂  Found {len(test_images)} test image(s)\n")
    print("-" * 55)

    result_paths = []
    total_detections = 0

    for i, img_path in enumerate(test_images):
        print(f"[{i+1:02d}/{len(test_images)}]  Processing: {img_path.name}")

        t0 = time.time()
        tensor, orig_img, orig_w, orig_h = preprocess(str(img_path), args.image_size)
        output = run_inference(model, tensor, device)
        elapsed = time.time() - t0

        n_det = int((output['scores'].cpu().numpy() >= args.threshold).sum())
        total_detections += n_det

        save_name = f"result_{i+1:02d}_{img_path.stem}.jpg"
        save_path = os.path.join(args.out_dir, save_name)
        draw_and_save(orig_img, output, args.threshold,
                      args.image_size, save_path, img_path.name)
        result_paths.append(save_path)

        print(f"         → {n_det} paper(s) detected  |  {elapsed:.2f}s  |  saved: {save_name}")

    save_summary_grid(result_paths, args.out_dir)

    print("\n" + "="*55)
    print(f"  ✅  Inference complete!")
    print(f"  Images processed : {len(test_images)}")
    print(f"  Total detections : {total_detections}")
    print(f"  Results saved in : {args.out_dir}/")
    print("="*55 + "\n")


if __name__ == '__main__':
    main()